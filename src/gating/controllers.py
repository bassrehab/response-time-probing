"""Controllers that map probe scores to steering strengths.

Each controller takes a probe score in [0, 1] and outputs a steering
strength in [0, max_strength]. The controller defines the relationship
between detected threat level and intervention intensity.
"""

from dataclasses import dataclass


@dataclass
class BangBangController:
    """Binary on/off controller. Steers at full strength above threshold, zero below.

    Simple and interpretable. Good baseline. May cause abrupt behavior changes
    near the threshold boundary.
    """

    threshold: float = 0.5
    on_strength: float = 1.0

    def __call__(self, score: float) -> float:
        return self.on_strength if score >= self.threshold else 0.0


@dataclass
class ProportionalController:
    """Linear proportional controller. Steering strength scales with probe score.

    strength = clamp(gain * (score - threshold), 0, max_strength)

    Provides smooth intervention that scales with threat confidence.
    Preserves more utility on borderline cases than bang-bang.
    """

    threshold: float = 0.5
    max_strength: float = 1.0
    gain: float = 2.0

    def __call__(self, score: float) -> float:
        if score < self.threshold:
            return 0.0
        raw = self.gain * (score - self.threshold)
        return min(raw, self.max_strength)
