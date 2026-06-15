from pathlib import Path

import pytest

from experiments import run
from experiments.presets import PRESETS, RECORD_ROBOTS


def _command_options(command):
    return {part.split("=", 1)[0]: part.split("=", 1)[1] for part in command if part.startswith("--") and "=" in part}


def test_build_command_resolves_every_preset():
    for preset_name in PRESETS:
        command = run.build_command(preset_name, extra_args=[])
        assert command[0]
        assert Path(command[1]).name == "experiment.py"
        assert "--algorithm.name=uni_ppo.ppo" in command
        assert "--environment.name=multi_robot" in command


def test_build_command_preserves_extra_args_order():
    command = run.build_command("test_default", extra_args=["--runner.seed=123", "--custom=value"])

    assert command[-2:] == ["--runner.seed=123", "--custom=value"]


def test_record_robot_index_maps_to_single_robot_command():
    command = run.build_command("test_default", extra_args=[], record_robot_index=3)
    options = _command_options(command)

    assert options["--algorithm.record"] == "True"
    assert options["--algorithm.record_robot_index"] == "0"
    assert options["--environment.nr_envs"] == "1"
    assert options["--environment.multi_render"] == "False"
    assert options["--environment.train_robot_types"] == "('anymal_b',)"


def test_record_robot_index_validation_message():
    with pytest.raises(ValueError, match=f"0 to {len(RECORD_ROBOTS) - 1}"):
        run.build_command("test_default", extra_args=[], record_robot_index=len(RECORD_ROBOTS))
