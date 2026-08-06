"""Make the Fail2Drive scenario_runner work with the bench2drive srunner fork.

Fail2Drive ships a forked ``scenario_runner`` (with extra scenario classes
like ``ImageOnObject`` / ``ObscuredStopSign``, plus modified versions of
shared files such as ``route_obstacles``). The Fail2Drive routes reference
those classes by ``<scenario type=...>``. ogbench-carla already runs the
bench2drive srunner — so to honor Fail2Drive routes faithfully we need to
**add Fail2Drive's scenario files to the leaderboard's class discovery**
without replacing the rest of the bench2drive srunner.

We pursue three runtime adjustments:

1. ``DeactivateBrakeLights`` is injected into
   ``srunner.scenariomanager.scenarioatomics.atomic_behaviors`` so that
   ``fail2drive/scenarios/hard_break.py``'s ``HardBrakeNoLights`` resolves
   its ``from ... import DeactivateBrakeLights``.

2. ``CarlaDataProvider`` gains an ``active_scenarios = []`` class attribute
   (Fail2Drive's scenarios append to this list to advertise spawned actors
   to their Expert; bench2drive's fork doesn't define it).

3. ``leaderboard.scenarios.route_scenario.RouteScenario.get_all_scenario_classes``
   is monkey-patched to also walk Fail2Drive's scenarios directory. Files
   from Fail2Drive are processed **first** so that for shared filenames
   (e.g. ``route_obstacles.py``) Fail2Drive's version lands in
   ``sys.modules`` and wins.

The patch is applied with :func:`apply` which is idempotent and safe to
call before any route is loaded. Call it once after CARLA's PythonAPI is
on ``sys.path`` but before the first ``RouteScenario`` is built.

Scenario directory resolution order (first one that exists wins):
  1. ``FAIL2DRIVE_SCENARIOS_DIR`` env var
     e.g. ``~/fail2drive/scenario_runner/srunner/scenarios``
  2. ``fail2drive`` Python package's ``SCENARIOS_DIR`` attribute
     (only if the package is pip-installed)
"""

from __future__ import annotations

import glob
import importlib
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Optional


_LOG = logging.getLogger(__name__)
_APPLIED = False


def _resolve_scenarios_dir() -> Optional[str]:
    """Return the path to Fail2Drive's scenarios directory, or None."""
    # 1. Explicit env var — works with a cloned repo (no pip install needed).
    env_dir = os.environ.get("FAIL2DRIVE_SCENARIOS_DIR", "")
    if env_dir:
        p = Path(env_dir).expanduser().resolve()
        if p.is_dir():
            return str(p)

    # 2. Installed package fallback.
    try:
        import fail2drive  # type: ignore
        d = Path(getattr(fail2drive, "SCENARIOS_DIR", ""))
        if d.is_dir():
            return str(d)
    except ImportError:
        pass

    return None


def _inject_atomic() -> Optional[str]:
    """Add ``DeactivateBrakeLights`` to srunner's atomic_behaviors if absent.

    ``hard_break.py`` imports it from ``srunner.scenariomanager.scenarioatomics``.
    bench2drive's fork doesn't ship it, so we load it from Fail2Drive's own
    atomic_behaviors.py (derived from FAIL2DRIVE_SCENARIOS_DIR or the package).
    """
    try:
        import srunner.scenariomanager.scenarioatomics.atomic_behaviors as ab
    except ImportError as e:
        return f'srunner.atomic_behaviors unavailable ({e})'

    if hasattr(ab, 'DeactivateBrakeLights'):
        return None  # already present (bench2drive fork ships it, or already injected)

    # Try the installed package first.
    try:
        from fail2drive.atomics import DeactivateBrakeLights  # type: ignore
        ab.DeactivateBrakeLights = DeactivateBrakeLights
        return None
    except (ImportError, AttributeError):
        pass

    # Fall back: load DeactivateBrakeLights from the cloned repo's atomic_behaviors.py.
    # It lives two levels up from the scenarios dir:
    #   <root>/scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_behaviors.py
    scenarios_dir = _resolve_scenarios_dir()
    if not scenarios_dir:
        return 'fail2drive scenarios dir not found (set FAIL2DRIVE_SCENARIOS_DIR)'

    # Derive atomic_behaviors.py path relative to scenarios dir.
    # scenarios_dir = .../srunner/scenarios  →  parent = .../srunner
    # then .../srunner/scenariomanager/scenarioatomics/atomic_behaviors.py
    srunner_root = Path(scenarios_dir).parent
    ab_path = srunner_root / "scenariomanager" / "scenarioatomics" / "atomic_behaviors.py"
    if not ab_path.exists():
        return f'DeactivateBrakeLights source not found at {ab_path}'

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_f2d_atomic_behaviors", str(ab_path)
        )
        f2d_ab = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(f2d_ab)  # type: ignore[union-attr]
        ab.DeactivateBrakeLights = f2d_ab.DeactivateBrakeLights
        return None
    except Exception as e:
        return f'Failed to load DeactivateBrakeLights from {ab_path}: {e}'


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

    f2d_dir = _resolve_scenarios_dir()
    if not f2d_dir:
        return 'fail2drive scenarios dir not found (set FAIL2DRIVE_SCENARIOS_DIR)'

    if getattr(RouteScenario.get_all_scenario_classes, '_fail2drive_patched', False):
        return None

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
        # Patch here rather than in apply(): discovery imports these modules under
        # their bare names, so this dict holds the class objects actually used to
        # build scenarios.
        _patch_dynamic_object_crossing_walker(all_scenario_classes)
        return all_scenario_classes

    get_all_scenario_classes._fail2drive_patched = True  # type: ignore[attr-defined]
    RouteScenario.get_all_scenario_classes = get_all_scenario_classes
    return None


def _patch_dynamic_object_crossing_walker(scenario_classes: dict) -> None:
    """Make ``DynamicObjectCrossing`` honor the route's ``<walker value=.../>``.

    bench2drive's ``srunner/scenarios/object_crash_vehicle.py`` hardcodes
    ``request_new_actor('walker.*', ...)`` for the adversary and never reads a
    ``walker`` parameter, so ``CarlaDataProvider.create_blueprint`` picks a
    *random* walker. On the Fail2Drive CARLA build that pool is 51 pedestrians
    + 18 animals, so ``generalization-animals-*`` routes get a human ~74% of the
    time even though the route XML asks for e.g. ``walker.animal.1006``.

    Fail2Drive ships no ``object_crash_vehicle.py``, so the file overlay in
    :func:`_patch_scenario_discovery` doesn't cover this one. Patch the class
    object that discovery actually returned (imported under its bare module
    name, so it is *not* the same object as
    ``srunner.scenarios.object_crash_vehicle.DynamicObjectCrossing``) and
    rewrite only the generic ``'walker.*'`` lookup. ``_replace_walker``
    re-spawns by concrete ``type_id`` and passes through untouched.
    """
    cls = scenario_classes.get('DynamicObjectCrossing')
    if cls is None:
        return
    original = getattr(cls, '_initialize_actors', None)
    if original is None or getattr(original, '_fail2drive_walker_patched', False):
        return

    def _initialize_actors(self, config):
        try:
            model = str(config.other_parameters['walker']['value'])
        except (AttributeError, KeyError, TypeError):
            return original(self, config)

        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

        real = CarlaDataProvider.request_new_actor

        def request_new_actor(model_arg, *args, **kwargs):
            return real(model if model_arg == 'walker.*' else model_arg, *args, **kwargs)

        CarlaDataProvider.request_new_actor = staticmethod(request_new_actor)
        try:
            return original(self, config)
        finally:
            CarlaDataProvider.request_new_actor = staticmethod(real)

    _initialize_actors._fail2drive_walker_patched = True  # type: ignore[attr-defined]
    cls._initialize_actors = _initialize_actors


def apply() -> None:
    """Apply all four shims. Idempotent.

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
