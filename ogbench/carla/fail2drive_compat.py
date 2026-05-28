"""Make the ``fail2drive`` Python package work with the bench2drive srunner fork.

Fail2Drive ships a forked ``scenario_runner`` (with extra scenario classes
like ``ImageOnObject`` / ``ObscuredStopSign``, plus modified versions of
shared files such as ``route_obstacles``). The Fail2Drive routes reference
those classes by ``<scenario type=...>``. ogbench-carla already runs the
bench2drive srunner — so to honor Fail2Drive routes faithfully we need to
**add Fail2Drive's scenario files to the leaderboard's class discovery**
without replacing the rest of the bench2drive srunner.

We pursue three runtime adjustments, all of which leave Fail2Drive's own
source byte-for-byte (the source lives in the ``fail2drive`` package):

1. ``DeactivateBrakeLights`` is injected into
   ``srunner.scenariomanager.scenarioatomics.atomic_behaviors`` so that
   ``fail2drive/scenarios/hard_break.py``'s ``HardBrakeNoLights`` resolves
   its ``from ... import DeactivateBrakeLights``.

2. ``CarlaDataProvider`` gains an ``active_scenarios = []`` class attribute
   (Fail2Drive's scenarios append to this list to advertise spawned actors
   to their Expert; bench2drive's fork doesn't define it).

3. ``leaderboard.scenarios.route_scenario.RouteScenario.get_all_scenario_classes``
   is monkey-patched to also walk :data:`fail2drive.SCENARIOS_DIR`. Files
   from Fail2Drive are processed **first** so that for shared filenames
   (e.g. ``route_obstacles.py``) Fail2Drive's version lands in
   ``sys.modules`` and wins.

The patch is applied with :func:`apply` which is idempotent and safe to
call before any route is loaded. Call it once after CARLA's PythonAPI is
on ``sys.path`` but before the first ``RouteScenario`` is built.
"""

from __future__ import annotations

import glob
import importlib
import inspect
import logging
import os
import sys
from typing import Optional


_LOG = logging.getLogger(__name__)
_APPLIED = False


def _inject_atomic() -> Optional[str]:
    """Add ``DeactivateBrakeLights`` to srunner's atomic_behaviors if absent."""
    try:
        import srunner.scenariomanager.scenarioatomics.atomic_behaviors as ab
    except ImportError as e:
        return f'srunner.atomic_behaviors unavailable ({e})'

    if hasattr(ab, 'DeactivateBrakeLights'):
        return None

    try:
        from fail2drive.atomics import DeactivateBrakeLights
    except ImportError as e:
        return f'fail2drive.atomics unavailable ({e})'

    ab.DeactivateBrakeLights = DeactivateBrakeLights
    return None


def _add_active_scenarios_attr() -> Optional[str]:
    """Give ``CarlaDataProvider`` the ``active_scenarios`` list Fail2Drive expects."""
    try:
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
    except ImportError as e:
        return f'CarlaDataProvider unavailable ({e})'

    if not hasattr(CarlaDataProvider, 'active_scenarios'):
        CarlaDataProvider.active_scenarios = []
    return None


def _patch_scenario_discovery() -> Optional[str]:
    """Make ``RouteScenario.get_all_scenario_classes`` also walk fail2drive scenarios.

    Fail2Drive entries are processed first; for shared module names that
    ensures Fail2Drive's version of the file wins in ``sys.modules``.
    """
    try:
        from leaderboard.scenarios.route_scenario import RouteScenario
    except ImportError as e:
        return f'RouteScenario unavailable ({e})'

    try:
        import fail2drive
    except ImportError as e:
        return f'fail2drive package unavailable ({e})'

    f2d_dir = str(fail2drive.SCENARIOS_DIR)
    if getattr(RouteScenario.get_all_scenario_classes, '_fail2drive_patched', False):
        return None

    original = RouteScenario.get_all_scenario_classes

    def get_all_scenario_classes(self):
        """Discover scenario classes from both the fail2drive package and the
        normal srunner location next to the leaderboard install."""
        # Fail2Drive's location FIRST — so any shared module name (e.g.
        # ``route_obstacles``) gets imported from fail2drive's file and
        # cached in sys.modules before the bench2drive version is reached.
        f2d_files = sorted(glob.glob(os.path.join(f2d_dir, '*.py')))
        # bench2drive's location is what the upstream method walks; mirror it
        # so we don't depend on which file's get_all_scenario_classes ran.
        leaderboard_scenarios_dir = os.path.dirname(os.path.abspath(
            sys.modules[type(self).__module__].__file__
        ))
        srunner_files = sorted(glob.glob(
            os.path.join(leaderboard_scenarios_dir, '../../srunner/scenarios/*.py')
        ))

        all_scenario_classes = {}
        for scenario_file in f2d_files + srunner_files:
            module_name = os.path.basename(scenario_file).split('.')[0]
            if module_name == '__init__':
                continue
            scenario_dir = os.path.dirname(scenario_file)
            if scenario_dir not in sys.path:
                sys.path.insert(0, scenario_dir)
            try:
                scenario_module = importlib.import_module(module_name)
            except Exception as e:
                _LOG.warning(
                    'Failed to import scenario module %r from %s: %s',
                    module_name, scenario_file, e,
                )
                continue
            for member in inspect.getmembers(scenario_module, inspect.isclass):
                all_scenario_classes[member[0]] = member[1]
        return all_scenario_classes

    get_all_scenario_classes._fail2drive_patched = True  # type: ignore[attr-defined]
    RouteScenario.get_all_scenario_classes = get_all_scenario_classes
    return None


def apply() -> None:
    """Apply all three shims. Idempotent.

    Call after CARLA's ``PythonAPI/carla`` is on ``sys.path`` (so srunner
    can import) and before the first ``RouteScenario`` is instantiated.
    """
    global _APPLIED
    if _APPLIED:
        return

    errors = []
    for step in (_inject_atomic, _add_active_scenarios_attr, _patch_scenario_discovery):
        err = step()
        if err:
            errors.append(f'{step.__name__}: {err}')

    if errors:
        _LOG.info(
            'fail2drive compat shim skipped some steps: %s',
            '; '.join(errors),
        )
    _APPLIED = True
