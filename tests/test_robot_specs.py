import pytest

from experiments.presets import RECORD_ROBOTS
from one_policy_to_run_them_all.environments.multi_robot.robot_helper import (
    ROBOTS,
    ROBOTS_BY_CLASS,
    ROBOTS_BY_LONG_NAME,
    ROBOTS_BY_SHORT_NAME,
    get_robot_class,
    get_robot_long_names,
    get_robot_short_name,
    get_robot_spec,
    get_robot_spec_by_class,
)
from one_policy_to_run_them_all.environments.robot_spec import BaseMujocoRobotEnv, RobotSpec
from one_policy_to_run_them_all.environments.unitree_a1.environment import UnitreeA1


def test_robot_specs_have_unique_names_and_matching_class_constants():
    assert len(ROBOTS) == 20
    assert len(ROBOTS_BY_LONG_NAME) == len(ROBOTS)
    assert len(ROBOTS_BY_SHORT_NAME) == len(ROBOTS)
    assert len(ROBOTS_BY_CLASS) == len(ROBOTS)

    for robot in ROBOTS:
        assert robot.long_name == robot.cls.LONG_NAME
        assert robot.short_name == robot.cls.SHORT_NAME
        assert robot.data_dir.name == "data"


def test_unitree_a1_uses_shared_base_env_contract():
    assert issubclass(UnitreeA1, BaseMujocoRobotEnv)
    assert UnitreeA1.robot_spec().long_name == "unitree_a1"


def test_robot_registry_resolves_names_and_classes():
    spec = get_robot_spec("unitree_a1")

    assert spec.cls is UnitreeA1
    assert get_robot_class("unitree_a1") is UnitreeA1
    assert get_robot_short_name("unitree_a1") == "a1"
    assert get_robot_spec_by_class(UnitreeA1) is spec
    assert "unitree_a1" in get_robot_long_names()


def test_robot_registry_reports_available_names_for_unknown_robot():
    with pytest.raises(ValueError, match="Available"):
        get_robot_spec("missing_robot")


def test_record_robot_presets_resolve_to_known_specs():
    for robot_name in RECORD_ROBOTS:
        assert get_robot_spec(robot_name).long_name == robot_name


def test_base_mujoco_robot_env_can_build_default_spec_from_subclass_constants():
    class ToyRobot(BaseMujocoRobotEnv):
        LONG_NAME = "toy_robot"
        SHORT_NAME = "toy"

    spec = ToyRobot.robot_spec()

    assert spec.long_name == "toy_robot"
    assert spec.short_name == "toy"
    assert spec.cls is ToyRobot


def test_robot_spec_validation_catches_mismatched_constants():
    with pytest.raises(ValueError, match="long_name"):
        RobotSpec(cls=UnitreeA1, long_name="wrong", short_name=UnitreeA1.SHORT_NAME).validate()
