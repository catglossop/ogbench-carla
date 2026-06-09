"""Soft-penalty reward infrastructure (ca5be36).

Provides compute_soft_penalty_reward(), which shapes progress reward by
multiplying with factors for outside-lanes, speeding (speed-limit penalty),
time-to-collision, and comfort, then adds a terminal reward on episode end.

Speed-limit penalty: enabled by setting speeding_infraction=true in the CARLA
config (automatically forced on when use_soft_penalty_reward=true).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

_TERMINATION_CRASH_MESSAGES: Dict[str, str] = {
    "collision": "Collision",
    "run_red_light": "RunningRedLight",
    "run_stop_sign": "RunningStop",
    "off_road": "OffRoad",
    "route_deviation": "RouteDeviation",
    "blocked": "Blocked",
}


def compute_soft_penalty_reward(
    *,
    route_completion_delta: float,
    soft_penalties: Dict[str, Any],
    terminal_state: Dict[str, Any],
    use_perc_progress: bool,
) -> Tuple[float, bool, Dict[str, Any]]:
    """Compute the soft-penalty reward from commit ca5be36.

    Progress reward = route_completion_delta, multiplied down by speeding /
    TTC / comfort factors when the ego is within lane bounds.  Optionally
    multiplied by lane_center_factor when use_perc_progress is True.
    Terminal reward is added from terminal_state["terminal_reward"].

    Returns (reward, terminated, info_updates).  The caller is responsible
    for side effects such as _finalize_route.
    """
    progress_reward = route_completion_delta
    if soft_penalties["outside_lanes"]:
        progress_reward = 0.0
    else:
        if soft_penalties["overspeed_kmh"] > 0.0:
            progress_reward *= soft_penalties["speeding_factor"]
        if soft_penalties["ttc_factor"] < 1.0:
            progress_reward *= soft_penalties["ttc_factor"]
        if soft_penalties["comfort_factor"] < 1.0:
            progress_reward *= soft_penalties["comfort_factor"]
    if use_perc_progress:
        progress_reward *= soft_penalties["lane_center_factor"]
    reward = progress_reward + terminal_state["terminal_reward"]

    info: Dict[str, Any] = {
        "progress_reward": float(progress_reward),
        "soft_penalty_product": float(soft_penalties["penalty_product"]),
        "soft_penalty_outside_lanes": float(soft_penalties["outside_lanes_factor"]),
        "soft_penalty_lane_center": float(soft_penalties["lane_center_factor"]),
        "soft_penalty_speeding": float(soft_penalties["speeding_factor"]),
        "soft_penalty_ttc": float(soft_penalties["ttc_factor"]),
        "soft_penalty_comfort": float(soft_penalties["comfort_factor"]),
        "outside_lanes": bool(soft_penalties["outside_lanes"]),
        "overspeed_kmh": float(soft_penalties["overspeed_kmh"]),
        "ttc_violated_now": bool(soft_penalties["ttc_violated_now"]),
        "comfort_violated_now": bool(soft_penalties["comfort_violated_now"]),
        "comfort_violations": list(soft_penalties["comfort_metrics"]["violations"]),
        "comfort_metrics": {
            k: v for k, v in soft_penalties["comfort_metrics"].items() if k != "violations"
        },
        "collision": bool(terminal_state["collision"]),
        "off_road": bool(terminal_state["off_road"]),
        "run_red_light": bool(terminal_state["run_red_light"]),
        "run_stop_sign": bool(terminal_state["run_stop_sign"]),
        "route_deviation": bool(terminal_state["route_deviation"]),
        "route_deviation_distance_m": float(terminal_state["route_deviation_distance_m"]),
        "blocked": bool(terminal_state["blocked"]),
        "left_route": bool(terminal_state["left_route"]),
        "in_route_ok": bool(terminal_state["in_route_ok"]),
        "route_completed": bool(terminal_state.get("route_completed", False)),
        "success": bool(terminal_state["success"]),
        "termination_reason": str(terminal_state["termination_reason"]),
        "reward_terminal": float(terminal_state["terminal_reward"]),
        "reward_total": float(reward),
    }
    return float(reward), bool(terminal_state["terminated"]), info


def termination_crash_message(termination_reason: str) -> str:
    """Map a termination_reason string to the Bench2Drive crash-message string."""
    return _TERMINATION_CRASH_MESSAGES.get(termination_reason, "")
