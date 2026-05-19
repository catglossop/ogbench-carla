from __future__ import annotations


class ScenarioLogger:
    def __init__(self, *args, **kwargs):
        self.ego_vehicle = None
        self.world = None

    def log_step(self, *args, **kwargs):
        return None

    def dump_to_json(self):
        return None
