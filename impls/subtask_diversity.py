"""Keyword-based subtask diversity utilities.

Shared by best-of-N candidate sampling (main_carla.py) and the interactive
candidate picker (main_carla_teleop.py) -- both need to tell candidates apart
by their VLA-generated subtask text well enough to reject near-duplicates
without needing real NLP.
"""

from __future__ import annotations

MAX_SAMPLE_ATTEMPTS_PER_CANDIDATE = 6
# A candidate must share less than this fraction of its category tags (Jaccard) with
# every already-accepted candidate to count as "diverse enough" and stop early.
DIVERSITY_JACCARD_THRESHOLD = 0.5

SUBTASK_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "accelerate": ("accelerat",),
    "decelerate": ("decelerat", "brak", "slow"),
    "stop": ("stop", "stopped", "remain"),
    "turn_right": ("turn right", "turns right", "right turn", "rightward turn"),
    "turn_left": ("turn left", "turns left", "left turn", "leftward turn"),
    "adjust_right": ("adjustment to the right", "right adjustment", "adjusts right", "rightward adjustment"),
    "adjust_left": ("adjustment to the left", "left adjustment", "adjusts left", "leftward adjustment"),
    "maintain": ("maintain", "steady course"),
    "reverse": ("revers",),
}


def subtask_categories(text: str) -> frozenset[str]:
    t = text.lower()
    return frozenset(
        cat for cat, keywords in SUBTASK_CATEGORY_KEYWORDS.items() if any(kw in t for kw in keywords)
    )


def category_jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 1.0  # can't tell them apart from keywords alone -> treat as maximally similar
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def diversity_score(cats: frozenset[str], accepted_cats: list[frozenset[str]]) -> float:
    """Higher is better: how different ``cats`` is from every already-accepted candidate."""
    if not accepted_cats:
        return 1.0
    return 1.0 - max(category_jaccard(cats, existing) for existing in accepted_cats)
