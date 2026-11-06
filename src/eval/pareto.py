"""Pareto frontier computation for safety-utility tradeoff analysis."""

from dataclasses import dataclass


@dataclass
class ParetoPoint:
    """A single point on the safety-utility plane."""

    model: str
    defense: str
    dsr: float          # Defense Success Rate (higher = safer)
    utility: float      # MMLU accuracy (higher = more helpful)
    fpr: float
    latency_ms: float
    config: dict        # Full config (strength, threshold, etc.)


def compute_pareto_frontier(points: list[ParetoPoint]) -> list[ParetoPoint]:
    """Compute the Pareto frontier from a set of points.

    A point is Pareto-optimal if no other point has both higher DSR AND
    higher utility. We want to maximize both axes.

    Args:
        points: List of ParetoPoints.

    Returns:
        List of Pareto-optimal points, sorted by DSR ascending.
    """
    if not points:
        return []

    # Sort by DSR descending
    sorted_points = sorted(points, key=lambda p: p.dsr, reverse=True)

    frontier = []
    max_utility = -1.0

    for p in sorted_points:
        if p.utility > max_utility:
            frontier.append(p)
            max_utility = p.utility

    # Return sorted by DSR ascending for plotting
    return sorted(frontier, key=lambda p: p.dsr)


def check_pareto_dominance(
    candidate: ParetoPoint,
    baseline: ParetoPoint,
    dsr_margin: float = 5.0,
    utility_tolerance: float = 2.0,
) -> bool:
    """Check if candidate Pareto-dominates baseline.

    Dominance criteria:
      - candidate DSR >= baseline DSR + dsr_margin (in percentage points)
      - candidate utility >= baseline utility - utility_tolerance

    Args:
        candidate: The defense being evaluated (e.g., probe-gated).
        baseline: The defense being compared against (e.g., AlphaSteer).
        dsr_margin: Required DSR improvement in percentage points.
        utility_tolerance: Allowed utility degradation in percentage points.

    Returns:
        True if candidate dominates baseline.
    """
    dsr_improvement = (candidate.dsr - baseline.dsr) * 100  # Convert to pp
    utility_delta = (candidate.utility - baseline.utility) * 100

    return (dsr_improvement >= dsr_margin) and (utility_delta >= -utility_tolerance)
