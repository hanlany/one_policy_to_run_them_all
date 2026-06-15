from setuptools import find_packages

from setup import find_environment_assets


def test_setup_package_discovery_includes_nested_packages():
    packages = set(find_packages(include=["one_policy_to_run_them_all", "one_policy_to_run_them_all.*"]))

    assert "one_policy_to_run_them_all" in packages
    assert "one_policy_to_run_them_all.environments.multi_robot" in packages
    assert "one_policy_to_run_them_all.algorithms.uni_ppo.ppo" in packages


def test_environment_assets_are_included_in_package_data():
    assets = set(find_environment_assets())

    assert "environments/unitree_a1/data/unitree_a1.xml" in assets
    assert any(asset.startswith("environments/anymal_b/data/assets/") for asset in assets)


def test_multi_robot_helper_import_smoke():
    from one_policy_to_run_them_all.environments.multi_robot.robot_helper import ROBOTS

    assert ROBOTS
    assert {robot.long_name for robot in ROBOTS} >= {"unitree_a1", "anymal_b"}
