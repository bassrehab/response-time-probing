#!/usr/bin/env python3
"""Build the C1 marker-decorrelated training spec for the ICLR v2 probe.

The paper's probe trains on (40 harmful, all with a compliance-marker prefill)
vs (50 benign, no prefill). "Is there a marker in the window" therefore
separates the classes perfectly -> the probe learns the marker, not the harm
(workshop audit, App. mech_discriminators).

C1 decorrelates the marker from the label by emitting FOUR quadrants so the
compliance marker appears with BOTH labels and harmful content appears with and
without the marker:

  quadrant            prefill                                   label
  harm_with_marker    "Sure, here's how to" (compliance marker) 1
  harm_no_marker      marker-STRIPPED harmful opener            1 (judge-verified)
  benign_with_marker  compliance marker + benign query          0
  benign_no_marker    none (paper's benign style)               0

`label` is EXPECTED. Harm quadrants set needs_judge=True: on GPU the harness
generates the first-N tokens under the given prefill and KEEPS the sample as
label=1 only if the judge rates the continuation harmful; refusals are dropped.
This is why harm_no_marker prefills are model-agnostic here (built from Mistral's
cached harmful bodies + AdvBench text) and re-judged per model in the harness --
Llama has too few cached harmful continuations (~7) to build a per-model set.

Output: data/c1_marker_decorrelated/train_spec.json  (+ summary.json)
No GPU. Run: python scripts/build_c1_dataset.py
"""
import json
import re
import glob
from pathlib import Path

import sys
sys.path.insert(0, ".")
from src.attacks.datasets.prompts import ADVBENCH_BEHAVIORS, BENIGN_QUERIES, BENIGN_OPENENDED

OUT = Path("data/c1_marker_decorrelated")
OUT.mkdir(parents=True, exist_ok=True)

# Compliance markers (paper's prefill set) used for the *_with_marker quadrants.
COMPLIANCE_MARKERS = [
    "Sure, here is", "Sure, here's how to", "Of course! Here are the detailed steps:",
    "Absolutely, I can help with that.", "Great question! Here's what you need to know:",
    "Happy to help! Let me walk you through this.", "Here's a step-by-step guide:",
    "No problem, I'll explain.",
]

# Leading compliance phrasing to strip when recovering a marker-free harmful
# opener from a cached response. Longest-first so prefixes match greedily.
_STRIP_PREFIXES = sorted(set(COMPLIANCE_MARKERS + [
    "Of course!", "Of course,", "Of course.", "Certainly!", "Certainly,", "Sure thing,",
    "Sure!", "Sure,", "Sure.", "Absolutely!", "Absolutely,", "Here is", "Here's",
    "I'd be happy to help.", "I can help with that.", "Let me help", "Here are the",
]), key=len, reverse=True)

_MARKER_RE = re.compile("|".join(re.escape(m) for m in _STRIP_PREFIXES), re.IGNORECASE)


def strip_marker(text: str) -> str:
    """Remove leading compliance phrasing + punctuation/newlines, iteratively."""
    t = text.strip()
    for _ in range(4):
        before = t
        m = _MARKER_RE.match(t)
        if m:
            t = t[m.end():]
        t = t.lstrip(" \t\n:.,!-–—\r")
        if t == before:
            break
    return t


def first_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n]).strip()


def load_cached_harmful_bodies():
    """prompt -> marker-free harmful opener, harvested from cached Mistral runs.
    Only keep continuations judged harmful with enough post-marker content."""
    bodies = {}
    def is_harm(r):
        j = r.get("judge_result")
        if isinstance(j, dict):
            return bool(j.get("is_harmful"))
        return bool(r.get("is_harmful_baseline"))
    for path in glob.glob("results/**/mistral-7b/*.jsonl", recursive=True):
        try:
            rows = [json.loads(l) for l in open(path)]
        except Exception:
            continue
        for r in rows:
            resp = r.get("response") or ""
            prompt = r.get("prompt") or ""
            at = str(r.get("attack_type") or r.get("distribution") or "")
            if not prompt or len(resp) < 150:
                continue
            if not (is_harm(r) and ("prefill" in at.lower() or r.get("prefill"))):
                continue
            body = strip_marker(resp)
            if len(body.split()) < 8:            # nothing left after the marker
                continue
            opener = first_words(body, 14)        # marker-free harmful opener
            # prefer the longest/cleanest opener per prompt
            if prompt not in bodies or len(opener) > len(bodies[prompt]):
                bodies[prompt] = opener
    return bodies


def match_body(prompt, bodies):
    """Exact prompt match, else fuzzy by shared prefix (prompts are truncated in some caches)."""
    if prompt in bodies:
        return bodies[prompt]
    key = prompt[:40].lower()
    for p, o in bodies.items():
        if p[:40].lower() == key:
            return o
    return None


def build():
    bodies = load_cached_harmful_bodies()
    records = []
    rid = 0

    def add(prompt, prefill, label, quadrant, needs_judge):
        nonlocal rid
        records.append({
            "id": rid, "prompt": prompt, "prefill": prefill,
            "label": label, "quadrant": quadrant, "needs_judge": needs_judge,
        })
        rid += 1

    n_bodies_matched = 0
    for i, beh in enumerate(ADVBENCH_BEHAVIORS):
        marker = COMPLIANCE_MARKERS[i % len(COMPLIANCE_MARKERS)]
        # Q1 harm_with_marker (paper style)
        add(beh, marker, 1, "harm_with_marker", True)
        # Q2 harm_no_marker: marker-free harmful opener (from cache if available)
        opener = match_body(beh, bodies)
        if opener:
            n_bodies_matched += 1
            add(beh, opener, 1, "harm_no_marker", True)

    for i, q in enumerate(BENIGN_QUERIES):
        marker = COMPLIANCE_MARKERS[i % len(COMPLIANCE_MARKERS)]
        # Q3 benign_with_marker: compliance marker on a benign query
        add(q, marker, 0, "benign_with_marker", False)
        # Q4 benign_no_marker (paper style)
        add(q, None, 0, "benign_no_marker", False)

    # Balance target: for the marker to be *uninformative* we want equal counts
    # in all four quadrants (P(harm|marker) == P(harm|no-marker) == 0.5).
    # harm_no_marker is short (limited by cache); the harness tops it up to
    # TARGET on GPU by applying the marker-free openers to further behaviours
    # and re-judging. benign quadrants are already >= TARGET.
    TARGET = 40

    # C5 diverse negatives kept SEPARATE (marker-free), for OOD-FPR eval, not
    # for the decorrelation training balance.
    diverse = [{"id": j, "prompt": q, "prefill": None, "label": 0,
                "quadrant": "diverse_benign", "needs_judge": False}
               for j, q in enumerate(BENIGN_OPENENDED[:60])]
    (OUT / "diverse_benign.json").write_text(json.dumps(diverse, indent=2))

    from collections import Counter
    quad_counts = Counter(r["quadrant"] for r in records)
    label_counts = Counter(r["label"] for r in records)
    summary = {
        "n_records": len(records),
        "quadrants": dict(quad_counts),
        "labels": dict(label_counts),
        "harm_no_marker_from_cache": n_bodies_matched,
        "cached_bodies_available": len(bodies),
        "target_per_quadrant": TARGET,
        "gpu_todo": {
            "augment_harm_no_marker_to": TARGET,
            "subsample_benign_each_to": TARGET,
            "reason": "balance quadrants so the compliance marker carries no label information",
        },
        "diverse_benign_file": "diverse_benign.json (60, C5 OOD-FPR eval, not training)",
        "decorrelation_check": {
            "p_harm_given_marker": round(quad_counts["harm_with_marker"] /
                max(1, quad_counts["harm_with_marker"] + quad_counts["benign_with_marker"]), 3),
            "p_harm_given_no_marker_current": round(quad_counts["harm_no_marker"] /
                max(1, quad_counts["harm_no_marker"] + quad_counts["benign_no_marker"]), 3),
            "note": "these converge to 0.5/0.5 once the GPU balances the quadrants to TARGET",
        },
        "note": "harm_* quadrants re-judged per model on GPU; label is expected. "
                "harm_no_marker prefills are marker-free openers from Mistral cache, "
                "applied to both models (Llama has too few cached harmful bodies).",
    }
    (OUT / "train_spec.json").write_text(json.dumps(records, indent=2))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    # show 2 examples per quadrant
    for quad in ["harm_with_marker", "harm_no_marker", "benign_with_marker", "benign_no_marker"]:
        ex = [r for r in records if r["quadrant"] == quad][:2]
        print(f"\n[{quad}]")
        for r in ex:
            print(f"  prompt: {r['prompt'][:55]!r}")
            print(f"  prefill: {str(r['prefill'])[:55]!r}  label={r['label']}")


if __name__ == "__main__":
    build()
