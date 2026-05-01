"""Leaderboard-style agent modules used by ``CarlaBench2DriveWrapper``.

These are *not* RL agents - they are just the thin AutonomousAgent objects the
leaderboard requires so that sensors get registered and the scenario can run.
The actual control comes from the gym wrapper's ``step(action)`` (see
``ogbench.carla.carla_utils.SteppableScenarioManager``).
"""
