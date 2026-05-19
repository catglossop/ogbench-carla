"""Expert language commentary computation from live CARLA world state.

Labels match the non-augmented vocabulary from the simlingo commentary generator
(simlingo/dataset_generation/language_labels/commentary/carla_commentary_generator.py).
No simlingo dependency required — all queries go through the CARLA Python API.

Commentary vocabulary (COMMENTARY_VOCAB) covers all unique words across the 929
non-augmented commentary templates in simlingo/data/augmented_templates/commentary_augmented.json.
Each generated commentary is bag-of-words encoded into a binary vector of length
NUM_COMMENTARY_WORDS (= 119).

The generated commentary text follows the same structure as simlingo's generator:
  "{action_route} {action_speed}{reason_speed}."

For backward compatibility the categorical one-hot interface is still exported:
  NUM_LABELS, SPEED_LABEL_NAMES, ROUTE_LABEL_NAMES, ALL_LABEL_NAMES.
"""

from __future__ import annotations

import re
import numpy as np

# ---------------------------------------------------------------------------
# Categorical label constants (kept for backward compat)
# ---------------------------------------------------------------------------

NUM_SPEED_LABELS = 6
NUM_ROUTE_LABELS = 6
NUM_LABELS = NUM_SPEED_LABELS + NUM_ROUTE_LABELS

SPEED_LABEL_NAMES = [
    "remain_stopped",
    "stop_now",
    "maintain_speed",
    "maintain_reduced_speed",
    "accelerate",
    "decelerate",
]
ROUTE_LABEL_NAMES = [
    "follow_route",
    "turn_left",
    "turn_right",
    "prepare_lane_change",
    "lane_change",
    "exit_parking_lot",
]
ALL_LABEL_NAMES = SPEED_LABEL_NAMES + ROUTE_LABEL_NAMES

# ---------------------------------------------------------------------------
# Commentary vocabulary (bag-of-words encoding)
# ---------------------------------------------------------------------------

# All 119 unique words extracted from the 929 non-augmented commentary templates
# in simlingo/data/augmented_templates/commentary_augmented.json (sorted).
COMMENTARY_VOCAB: list[str] = [
    "a", "accelerate", "accident", "according", "after", "and", "are", "around",
    "as", "at", "attention", "avoid", "avoiding", "away", "because", "before",
    "behind", "big", "bikes", "bit", "brake", "but", "change", "changing",
    "clear", "cleared", "closer", "collision", "coming", "cones", "construction",
    "crossing", "current", "decelerate", "distance", "do", "door", "down", "drive",
    "due", "emergency", "enough", "entering", "exit", "follow", "for", "gap",
    "give", "go", "green", "if", "in", "intersecting", "invades", "is", "its",
    "junction", "lane", "lanes", "left", "light", "limit", "lot", "maintain",
    "make", "moving", "necessary", "neighbouring", "now", "object", "obstacle",
    "of", "on", "oncoming", "opening", "original", "other", "overtake", "parked",
    "parking", "path", "pay", "pedestrian", "prepare", "reach", "red", "reduced",
    "remain", "return", "right", "road", "route", "shift", "since", "site",
    "slowing", "space", "speed", "stationary", "stay", "steer", "stop", "stopped",
    "target", "that", "the", "through", "to", "towards", "traffic", "turn",
    "vehicle", "vehicles", "wait", "walker", "way", "with", "you", "your",
]
NUM_COMMENTARY_WORDS: int = len(COMMENTARY_VOCAB)  # 119
_WORD_TO_IDX: dict[str, int] = {w: i for i, w in enumerate(COMMENTARY_VOCAB)}

# Separate vocabulary for critic delta-language feedback. This is intentionally
# compact and action-oriented; unlike COMMENTARY_VOCAB it is designed to encode
# corrective supervision such as "adjust right" or "decelerate more heavily".
DELTA_COMMENTARY_VOCAB: list[str] = [
    "accelerate",
    "adjust",
    "brake",
    "current",
    "decelerate",
    "follow",
    "heavily",
    "left",
    "maintain",
    "more",
    "now",
    "right",
    "route",
    "speed",
    "stop",
    "the",
    "your",
]
NUM_DELTA_COMMENTARY_WORDS: int = len(DELTA_COMMENTARY_VOCAB)
_DELTA_WORD_TO_IDX: dict[str, int] = {w: i for i, w in enumerate(DELTA_COMMENTARY_VOCAB)}


def commentary_to_bow(text: str) -> np.ndarray:
    """Multi-hot bag-of-words encoding of ``text`` over ``COMMENTARY_VOCAB``."""
    bow = np.zeros(NUM_COMMENTARY_WORDS, dtype=np.float32)
    for w in re.findall(r"[a-z']+", text.lower()):
        idx = _WORD_TO_IDX.get(w)
        if idx is not None:
            bow[idx] = 1.0
    return bow


def delta_commentary_to_bow(text: str) -> np.ndarray:
    """Multi-hot bag-of-words encoding of delta-commentary text."""
    bow = np.zeros(NUM_DELTA_COMMENTARY_WORDS, dtype=np.float32)
    for w in re.findall(r"[a-z']+", text.lower()):
        idx = _DELTA_WORD_TO_IDX.get(w)
        if idx is not None:
            bow[idx] = 1.0
    return bow


def delta_commentary_from_critic_actions(
    expert_first: np.ndarray,
    agent_first: np.ndarray,
    *,
    speed_tol: float = 0.10,
    speed_strong_tol: float = 0.35,
    lateral_tol: float = 0.10,
    lateral_strong_tol: float = 0.35,
) -> tuple[str, np.ndarray]:
    """Generate corrective critic language from expert-vs-agent action delta.

    Inputs are expected to be in critic action space, i.e. first-step 4-D
    SteerVLA action `[dx_speed, dy_speed, dx_route, dy_route]`.
    """
    expert_first = np.asarray(expert_first, dtype=np.float32).reshape(-1)
    agent_first = np.asarray(agent_first, dtype=np.float32).reshape(-1)
    if expert_first.shape[0] < 4 or agent_first.shape[0] < 4:
        text = "Follow the route. Maintain your current speed."
        return text, delta_commentary_to_bow(text)

    expert_speed = float(np.linalg.norm(expert_first[:2]))
    agent_speed = float(np.linalg.norm(agent_first[:2]))
    speed_delta = expert_speed - agent_speed

    route_lat_delta = float(expert_first[3] - agent_first[3])

    if route_lat_delta >= lateral_strong_tol:
        route_text = "Adjust left more."
    elif route_lat_delta >= lateral_tol:
        route_text = "Adjust left."
    elif route_lat_delta <= -lateral_strong_tol:
        route_text = "Adjust right more."
    elif route_lat_delta <= -lateral_tol:
        route_text = "Adjust right."
    else:
        route_text = "Follow the route."

    if expert_speed < 0.05 and agent_speed > 0.10:
        speed_text = "Stop now."
    elif speed_delta >= speed_strong_tol:
        speed_text = "Accelerate more heavily."
    elif speed_delta >= speed_tol:
        speed_text = "Accelerate."
    elif speed_delta <= -speed_strong_tol:
        speed_text = "Decelerate more heavily."
    elif speed_delta <= -speed_tol:
        speed_text = "Decelerate."
    else:
        speed_text = "Maintain your current speed."

    text = f"{route_text} {speed_text}".replace("..", ".").strip()
    if not text.endswith("."):
        text += "."
    return text, delta_commentary_to_bow(text)


def collision_override_delta_commentary() -> tuple[str, np.ndarray]:
    """Forced corrective commentary while the agent is pushing during contact."""
    text = "Brake more heavily."
    return text, delta_commentary_to_bow(text)


# ---------------------------------------------------------------------------
# Commentary text generation phrases (words must all be in COMMENTARY_VOCAB)
# ---------------------------------------------------------------------------

_ROUTE_PHRASE: dict[int, str] = {
    0: "Follow the route.",
    1: "Turn left.",
    2: "Turn right.",
    3: "Prepare to do a lane change.",
    4: "Do a lane change.",
    5: "Exit the parking lot.",
}

_SPEED_PHRASE: dict[int, str] = {
    0: "Remain stopped",
    1: "Stop now",
    2: "Maintain your current speed",
    3: "Maintain the reduced speed",
    4: "Accelerate",
    5: "Decelerate",
}


class ExpertLabelComputer:
    """Compute expert language commentary from the live CARLA world state.

    Instantiate once per episode (or reuse across episodes) and call
    :meth:`compute_full` after each ``env.step()`` / ``env.reset()`` to get
    the commentary text and bag-of-words encoding.

    Parameters
    ----------
    speed_limit_ratio : float
        Expert's target speed as a fraction of the posted speed limit.
    maintain_threshold : float
        m/s band within which speed is considered "at target".
    look_ahead_dist : float
        m, radius for vehicle/walker ahead detection.
    safe_follow_dist : float
        m, stop if a vehicle is closer than this on the forward path.
    turn_dist : float
        m, announce turn/lane-change within this distance.
    lane_change_dist : float
        m, label as "lane_change" (vs "prepare") within this distance.
    maintain_ratio : float
        Fraction of speed limit above which the label is maintain_speed vs
        maintain_reduced_speed (matches commentary generator 0.71 threshold).
    """

    def __init__(
        self,
        speed_limit_ratio: float = 0.9,
        maintain_threshold: float = 0.5,
        look_ahead_dist: float = 30.0,
        safe_follow_dist: float = 8.0,
        turn_dist: float = 20.0,
        lane_change_dist: float = 10.0,
        maintain_ratio: float = 0.71,
    ):
        self.speed_limit_ratio = speed_limit_ratio
        self.maintain_threshold = maintain_threshold
        self.look_ahead_dist = look_ahead_dist
        self.safe_follow_dist = safe_follow_dist
        self.turn_dist = turn_dist
        self.lane_change_dist = lane_change_dist
        self.maintain_ratio = maintain_ratio

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_full(
        self,
        ego_actor,
        world,
        world_map,
        route_world_coord,
        *,
        scenario_type: str = "",
        expert_action: np.ndarray | None = None,
    ) -> tuple[int, int, str, np.ndarray]:
        """Return ``(speed_idx, route_idx, commentary_text, bow_vector)``.

        Parameters
        ----------
        ego_actor : carla.Actor  — the ego vehicle.
        world : carla.World     — current CARLA world.
        world_map : carla.Map   — cached map (avoids repeated RPC calls).
        route_world_coord : list[(carla.Transform, RoadOption)]
        scenario_type : str     — used for parking-exit detection.
        """
        try:
            speed_idx, speed_ctx = self._speed_info(ego_actor, world, world_map)
        except Exception:
            speed_idx, speed_ctx = 4, {}

        try:
            route_idx = self._route_label(ego_actor, route_world_coord, scenario_type)
        except Exception:
            route_idx = 0

        # Match SimLingo's commentary pipeline more closely:
        # use the fixed expert trajectory to determine the speed/action intent,
        # and keep live-world hazard checks for the explanatory reason.
        if expert_action is not None:
            try:
                speed_idx, traj_ctx = self._trajectory_speed_info(
                    ego_actor,
                    np.asarray(expert_action, dtype=np.float32),
                    float(speed_ctx.get("speed_limit", 8.33)),
                )
                speed_ctx = {**speed_ctx, **traj_ctx}
            except Exception:
                pass

            try:
                route_idx = self._trajectory_route_label(
                    ego_actor,
                    np.asarray(expert_action, dtype=np.float32),
                    scenario_type=scenario_type,
                    fallback_route_idx=route_idx,
                )
            except Exception:
                pass

        commentary_text = self._generate_commentary_text(speed_idx, route_idx, speed_ctx)
        bow = commentary_to_bow(commentary_text)
        return speed_idx, route_idx, commentary_text, bow

    def compute(
        self,
        ego_actor,
        world,
        world_map,
        route_world_coord,
        *,
        scenario_type: str = "",
    ) -> tuple[int, int, np.ndarray]:
        """Backward-compatible interface: return ``(speed_idx, route_idx, one_hot[12])``."""
        speed_idx, route_idx, _text, _bow = self.compute_full(
            ego_actor, world, world_map, route_world_coord, scenario_type=scenario_type
        )
        one_hot = np.zeros(NUM_LABELS, dtype=np.float32)
        one_hot[speed_idx] = 1.0
        one_hot[NUM_SPEED_LABELS + route_idx] = 1.0
        return speed_idx, route_idx, one_hot

    # ------------------------------------------------------------------
    # Speed info (index + context for commentary generation)
    # ------------------------------------------------------------------

    def _speed_info(self, ego, world, world_map) -> tuple[int, dict]:
        """Return ``(speed_idx, context_dict)``."""
        import carla  # noqa: F401

        ego_loc = ego.get_location()
        wp = world_map.get_waypoint(ego_loc)
        # carla.Waypoint has no speed_limit in CARLA 0.9.16; read from the vehicle actor instead.
        try:
            speed_limit = max(ego.get_speed_limit() / 3.6, 1.0)
        except Exception:
            speed_limit = 8.33  # 30 km/h default
        target_speed = min(speed_limit * self.speed_limit_ratio, 20.0)

        red_light = False
        try:
            tl = ego.get_traffic_light()
            if tl is not None and str(tl.get_state()) == "Red":
                target_speed = 0.0
                red_light = True
        except Exception:
            pass

        vehicle_ahead = False
        walker_ahead = False
        ego_tf = ego.get_transform()
        ego_fwd = ego_tf.get_forward_vector()
        try:
            actors = world.get_actors()
            for v in actors.filter("*vehicle*"):
                if v.id == ego.id:
                    continue
                rel = v.get_location() - ego_loc
                dist = (rel.x ** 2 + rel.y ** 2) ** 0.5
                if dist > self.look_ahead_dist:
                    continue
                dot = rel.x * ego_fwd.x + rel.y * ego_fwd.y
                if dot <= 0:
                    continue
                v_vel = v.get_velocity()
                v_speed = (v_vel.x ** 2 + v_vel.y ** 2) ** 0.5
                vehicle_ahead = True
                if dist < self.safe_follow_dist:
                    target_speed = 0.0
                else:
                    target_speed = min(target_speed, v_speed)
            for w in actors.filter("*walker*"):
                rel = w.get_location() - ego_loc
                dist = (rel.x ** 2 + rel.y ** 2) ** 0.5
                if dist > self.safe_follow_dist * 2:
                    continue
                dot = rel.x * ego_fwd.x + rel.y * ego_fwd.y
                if dot <= 0:
                    continue
                walker_ahead = True
                target_speed = 0.0
        except Exception:
            pass

        v = ego.get_velocity()
        current_speed = (v.x ** 2 + v.y ** 2) ** 0.5

        if current_speed < 0.2 and target_speed < 0.5:
            speed_idx = 0  # remain_stopped
        elif abs(current_speed - target_speed) < self.maintain_threshold:
            if target_speed < 0.2:
                speed_idx = 1  # stop_now
            elif target_speed / speed_limit > self.maintain_ratio:
                speed_idx = 2  # maintain_speed
            else:
                speed_idx = 3  # maintain_reduced_speed
        elif current_speed < target_speed:
            speed_idx = 4  # accelerate
        else:
            speed_idx = 5  # decelerate

        ctx = {
            "red_light": red_light,
            "vehicle_ahead": vehicle_ahead,
            "walker_ahead": walker_ahead,
            "current_speed": current_speed,
            "target_speed": target_speed,
            "speed_limit": speed_limit,
        }
        return speed_idx, ctx

    def _speed_label(self, ego, world, world_map) -> int:
        speed_idx, _ = self._speed_info(ego, world, world_map)
        return speed_idx

    def _trajectory_speed_info(
        self,
        ego,
        expert_action: np.ndarray,
        speed_limit: float,
    ) -> tuple[int, dict]:
        """SimLingo-style speed label from future expert target speeds.

        SimLingo commentary uses the average of the next few future target
        speeds, not only the current target speed. Our fixed expert chunk is a
        delta trajectory in ego frame, so we estimate those future target speeds
        from the first speed deltas.
        """
        chunk = np.asarray(expert_action, dtype=np.float32).reshape(-1, 4)
        if chunk.shape[0] == 0:
            raise ValueError("empty expert_action")

        dt = 5.0 / 20.0
        future_target_speeds = np.linalg.norm(chunk[:, :2], axis=1) / max(dt, 1e-6)
        avg_future_target_speed = float(np.mean(future_target_speeds[: min(5, len(future_target_speeds))]))

        v = ego.get_velocity()
        current_speed = float((v.x ** 2 + v.y ** 2) ** 0.5)

        if current_speed < 0.2 and abs(current_speed - avg_future_target_speed) < 0.5:
            speed_idx = 0  # remain_stopped
        elif abs(current_speed - avg_future_target_speed) < self.maintain_threshold:
            if avg_future_target_speed < 0.2:
                speed_idx = 1  # stop_now
            elif avg_future_target_speed / max(speed_limit, 1e-3) > self.maintain_ratio:
                speed_idx = 2  # maintain_speed
            else:
                speed_idx = 3  # maintain_reduced_speed
        elif current_speed < avg_future_target_speed:
            speed_idx = 4  # accelerate
        else:
            speed_idx = 5  # decelerate

        return speed_idx, {
            "current_speed": current_speed,
            "target_speed": avg_future_target_speed,
            "speed_limit": speed_limit,
            "expert_future_target_speed": avg_future_target_speed,
        }

    # ------------------------------------------------------------------
    # Route label
    # ------------------------------------------------------------------

    def _route_label(self, ego, route_world_coord, scenario_type: str) -> int:
        if scenario_type and "parking" in scenario_type.lower():
            ego_loc = ego.get_location()
            wp = None
            try:
                from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
                world_map = CarlaDataProvider.get_map()
                wp = world_map.get_waypoint(ego_loc)
            except Exception:
                pass
            if wp is not None and str(getattr(wp, "lane_type", "")) == "Parking":
                return 5  # exit_parking_lot

        if not route_world_coord:
            return 0

        try:
            from agents.navigation.local_planner import RoadOption
        except ImportError:
            return 0

        ego_loc = ego.get_location()
        ego_tf = ego.get_transform()
        ego_fwd = ego_tf.get_forward_vector()

        upcoming: list[tuple[float, object]] = []
        for transform, road_option in route_world_coord:
            rel = transform.location - ego_loc
            dist = (rel.x ** 2 + rel.y ** 2) ** 0.5
            if dist > 50.0:
                continue
            dot = rel.x * ego_fwd.x + rel.y * ego_fwd.y
            if dot <= 0:
                continue
            upcoming.append((dist, road_option))

        upcoming.sort(key=lambda x: x[0])

        for dist, cmd in upcoming:
            if cmd == RoadOption.LEFT:
                if dist < self.turn_dist:
                    return 1  # turn_left
            elif cmd == RoadOption.RIGHT:
                if dist < self.turn_dist:
                    return 2  # turn_right
            elif cmd in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
                if dist < self.lane_change_dist:
                    return 4  # lane_change
                elif dist < self.turn_dist:
                    return 3  # prepare_lane_change

        return 0  # follow_route

    def _trajectory_route_label(
        self,
        ego,
        expert_action: np.ndarray,
        *,
        scenario_type: str,
        fallback_route_idx: int,
    ) -> int:
        """Infer route intent from the expert future path, following SimLingo's future-path logic."""
        if scenario_type and "parking" in scenario_type.lower():
            ego_loc = ego.get_location()
            try:
                from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

                world_map = CarlaDataProvider.get_map()
                wp = world_map.get_waypoint(ego_loc)
                if wp is not None and str(getattr(wp, "lane_type", "")) == "Parking":
                    return 5
            except Exception:
                pass

        chunk = np.asarray(expert_action, dtype=np.float32).reshape(-1, 4)
        if chunk.shape[0] == 0:
            return fallback_route_idx

        route_pts = np.cumsum(chunk[:, 2:4], axis=0)
        if route_pts.shape[0] == 0:
            return fallback_route_idx

        final_x = float(route_pts[-1, 0])
        final_y = float(route_pts[-1, 1])
        first_y = float(route_pts[0, 1])
        mean_x = float(np.mean(route_pts[:, 0]))

        # SimLingo uses current command + future changed-route state. We do not
        # have those discrete labels here, so infer them from future path shape.
        abs_final_y = abs(final_y)
        heading = float(np.arctan2(final_y, max(final_x, 1e-3)))

        # Large lateral displacement with mostly forward motion behaves like a lane change.
        if abs_final_y > 2.0 and mean_x > 4.0 and abs(heading) < 0.35:
            if abs(first_y) > 0.75:
                return 4  # lane_change
            return 3  # prepare_lane_change

        if heading > 0.30:
            return 1  # turn_left
        if heading < -0.30:
            return 2  # turn_right

        return fallback_route_idx

    # ------------------------------------------------------------------
    # Commentary text generation
    # ------------------------------------------------------------------

    def _generate_commentary_text(self, speed_idx: int, route_idx: int, ctx: dict) -> str:
        """Generate a natural language commentary string using template-vocabulary words."""
        action_route = _ROUTE_PHRASE.get(route_idx, "Follow the route.")
        action_speed = _SPEED_PHRASE.get(speed_idx, "Accelerate")

        red_light = ctx.get("red_light", False)
        vehicle_ahead = ctx.get("vehicle_ahead", False)
        walker_ahead = ctx.get("walker_ahead", False)

        if walker_ahead and speed_idx in (0, 1, 5):
            reason = " due to the pedestrian crossing your path"
        elif red_light and speed_idx in (0, 1, 5):
            reason = " due to the red traffic light"
        elif vehicle_ahead and speed_idx == 4:
            reason = " to follow the vehicle"
        elif vehicle_ahead and speed_idx in (1, 5):
            reason = " to stay behind the vehicle"
        elif vehicle_ahead and speed_idx == 0:
            reason = " to stay behind the vehicle"
        elif speed_idx in (4, 5):
            reason = " to drive with the target speed"
        else:
            reason = ""

        commentary = f"{action_route} {action_speed}{reason}."
        return commentary.replace("..", ".").replace("  ", " ")
