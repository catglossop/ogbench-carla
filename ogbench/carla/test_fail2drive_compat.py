"""CPU regression checks: animal diagnostics must preserve scenario creation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import py_trees

spec = importlib.util.spec_from_file_location(
    'fail2drive_compat', Path(__file__).with_name('fail2drive_compat.py'))
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)


def test_animal_monitor_supports_both_scenario_transforms():
    for class_name, attribute in (
        ('DynamicObjectCrossing', '_adversary_transform'),
        ('VehicleTurningRoutePedestrian', '_spawn_transform'),
    ):
        class Scenario:
            def _initialize_actors(self, config):
                pass

            def _create_behavior(self):
                return self.original_behavior

        compat._patch_configured_walker_models({class_name: Scenario})
        scenario = Scenario()
        scenario.original_behavior = py_trees.composites.Sequence(name='OriginalCrossing')
        scenario.other_actors = [SimpleNamespace(
            type_id='walker.animal.1006', id=1, is_alive=True,
            get_location=lambda: SimpleNamespace(x=1., y=2., z=8.))]
        setattr(scenario, attribute, SimpleNamespace(location=SimpleNamespace(z=8.)))
        behavior = scenario._create_behavior()
        assert behavior.children[0] is scenario.original_behavior
        monitor = behavior.children[1]
        assert monitor.update() == py_trees.common.Status.RUNNING
        assert monitor._last_state == 'surface'
        # If a future scenario renames its transform, diagnostics must still
        # return its original tree instead of causing the hazard to be skipped.
        delattr(scenario, attribute)
        assert scenario._create_behavior() is scenario.original_behavior


if __name__ == '__main__':
    test_animal_monitor_supports_both_scenario_transforms()
    print('[ok] animal scenario construction survives lifecycle diagnostics')
