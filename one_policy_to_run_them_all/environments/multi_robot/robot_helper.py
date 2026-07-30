from one_policy_to_run_them_all.environments.anymal_b.environment import AnymalB
from one_policy_to_run_them_all.environments.anymal_c.environment import AnymalC
from one_policy_to_run_them_all.environments.atlas.environment import Atlas
from one_policy_to_run_them_all.environments.badger.environment import Badger
from one_policy_to_run_them_all.environments.badger_locked.environment import BadgerLocked
from one_policy_to_run_them_all.environments.barkour_v0.environment import BarkourV0
from one_policy_to_run_them_all.environments.barkour_vb.environment import BarkourVB
from one_policy_to_run_them_all.environments.bittle.environment import Bittle
from one_policy_to_run_them_all.environments.cassie.environment import Cassie
from one_policy_to_run_them_all.environments.hexapod.environment import Hexapod
from one_policy_to_run_them_all.environments.honey_badger.environment import HoneyBadger
from one_policy_to_run_them_all.environments.nao_v5.environment import NaoV5
from one_policy_to_run_them_all.environments.robotis_op3.environment import RobotisOP3
from one_policy_to_run_them_all.environments.sea_snake.environment import SEASnake
from one_policy_to_run_them_all.environments.talos.environment import Talos
from one_policy_to_run_them_all.environments.unitree_a1.environment import UnitreeA1
from one_policy_to_run_them_all.environments.unitree_g1.environment import UnitreeG1
from one_policy_to_run_them_all.environments.unitree_go1.environment import UnitreeGo1
from one_policy_to_run_them_all.environments.unitree_go2.environment import UnitreeGo2
from one_policy_to_run_them_all.environments.unitree_h1.environment import UnitreeH1
from one_policy_to_run_them_all.environments.vr_m3_1_12dof.environment import VRM31_12DOF
from one_policy_to_run_them_all.environments.vr_m3_1_full.environment import VRM31Full
from one_policy_to_run_them_all.environments.robot_spec import RobotSpec


# Backwards-compatible name for older callers that imported Robot from this module.
Robot = RobotSpec


def _spec(cls):
    return RobotSpec(cls=cls, long_name=cls.LONG_NAME, short_name=cls.SHORT_NAME).validate()


ROBOTS = [
    _spec(UnitreeA1),
    _spec(UnitreeGo1),
    _spec(UnitreeGo2),
    _spec(UnitreeH1),
    _spec(VRM31_12DOF),
    _spec(VRM31Full),
    _spec(UnitreeG1),
    _spec(Badger),
    _spec(BadgerLocked),
    _spec(HoneyBadger),
    _spec(Hexapod),
    _spec(Talos),
    _spec(AnymalB),
    _spec(AnymalC),
    _spec(RobotisOP3),
    _spec(BarkourV0),
    _spec(BarkourVB),
    _spec(Cassie),
    _spec(NaoV5),
    _spec(Bittle),
    _spec(Atlas),
    _spec(SEASnake),
]

ROBOTS_BY_LONG_NAME = {robot.long_name: robot for robot in ROBOTS}
ROBOTS_BY_SHORT_NAME = {robot.short_name: robot for robot in ROBOTS}
ROBOTS_BY_CLASS = {robot.cls: robot for robot in ROBOTS}

if len(ROBOTS_BY_LONG_NAME) != len(ROBOTS):
    raise ValueError("Robot long names must be unique.")
if len(ROBOTS_BY_SHORT_NAME) != len(ROBOTS):
    raise ValueError("Robot short names must be unique.")


def _available_long_names():
    return sorted(ROBOTS_BY_LONG_NAME)


def get_robot_spec(robot_type: str) -> RobotSpec:
    try:
        return ROBOTS_BY_LONG_NAME[robot_type]
    except KeyError as exc:
        raise ValueError(f"Unknown robot type '{robot_type}'. Available: {_available_long_names()}") from exc


def get_robot_spec_by_class(env_class) -> RobotSpec:
    try:
        return ROBOTS_BY_CLASS[env_class]
    except KeyError as exc:
        raise ValueError(
            f"Unknown robot environment class '{env_class}'. Available: {_available_long_names()}"
        ) from exc


def get_robot_class(robot_type: str):
    return get_robot_spec(robot_type).cls


def get_robot_short_name(robot_type: str) -> str:
    return get_robot_spec(robot_type).short_name


def get_robot_long_names() -> tuple[str, ...]:
    return tuple(ROBOTS_BY_LONG_NAME)
