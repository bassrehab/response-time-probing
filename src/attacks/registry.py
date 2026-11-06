"""Attack registry — load attack sets by name."""

from src.attacks.prefilling import generate_prefilling_attacks
from src.attacks.intent_laundering import generate_intent_laundering_attacks
from src.attacks.deep_inception import generate_deep_inception_attacks
from src.attacks.gcg import generate_gcg_attacks
from src.attacks.autodan import generate_autodan_attacks
from src.attacks.datasets.prompts import BENIGN_QUERIES


ATTACK_GENERATORS = {
    "prefilling": generate_prefilling_attacks,
    "intent_laundering": generate_intent_laundering_attacks,
    "deep_inception": generate_deep_inception_attacks,
    "gcg": generate_gcg_attacks,
    "autodan": generate_autodan_attacks,
}


def load_attack_set(
    attack_name: str,
    n_attacks: int = 20,
    include_benign: bool = True,
    n_benign: int = 20,
    seed: int = 42,
) -> list[dict]:
    """Load an attack set by name, optionally including benign prompts for FPR.

    Args:
        attack_name: Name of the attack ("prefilling", "intent_laundering").
        n_attacks: Number of attack prompts.
        include_benign: Whether to append benign prompts.
        n_benign: Number of benign prompts to include.
        seed: Random seed.

    Returns:
        List of dicts with keys: prompt, attack_type, and attack-specific fields.
    """
    if attack_name not in ATTACK_GENERATORS:
        raise ValueError(f"Unknown attack: {attack_name}. Available: {list(ATTACK_GENERATORS.keys())}")

    attacks = ATTACK_GENERATORS[attack_name](n_attacks=n_attacks, seed=seed)

    if include_benign:
        for i in range(min(n_benign, len(BENIGN_QUERIES))):
            attacks.append({
                "prompt": BENIGN_QUERIES[i],
                "attack_type": "benign",
            })

    return attacks
