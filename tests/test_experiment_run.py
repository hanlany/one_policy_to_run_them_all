import json
from pathlib import Path

import pytest

from experiments import run
from experiments.presets import PRESETS, RECORD_ROBOTS, load_presets


def _command_options(command):
    return {part.split("=", 1)[0]: part.split("=", 1)[1] for part in command if part.startswith("--") and "=" in part}


def test_build_command_resolves_every_preset():
    for preset_name in PRESETS:
        command = run.build_command(preset_name, extra_args=[])
        assert command[0]
        assert Path(command[1]).name == "experiment.py"
        assert "--algorithm.name=uni_ppo.ppo" in command
        assert any(part.startswith("--environment.name=") for part in command)


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
    assert options["--environment.eval_robot_types"] == str(RECORD_ROBOTS)


def test_record_student_records_student_stage():
    command = run.build_command("record_student", extra_args=[])
    options = _command_options(command)

    assert options["--algorithm.use_student"] == "True"
    assert options["--algorithm.rollout_policy_stage"] == "student"


def test_record_ssnv2_records_trained_long_bootstrap_snn():
    command = run.build_command("record_ssnv2", extra_args=[])
    options = _command_options(command)

    assert options["--algorithm.student_backend"] == "bootstrap"
    assert options["--algorithm.rollout_policy_stage"] == "snn"
    assert options["--algorithm.student_model_path"].endswith("/snn/Trained_long/network.pt")
    assert options["--algorithm.bootstrap_timesteps"] == "3"
    assert options["--algorithm.bootstrap_readout"] == "mean"
    assert options["--algorithm.bootstrap_neuron_threshold"] == "0.5"
    assert options["--algorithm.bootstrap_current_decay"] == "0.3"
    assert options["--algorithm.bootstrap_voltage_decay"] == "0.02"
    assert options["--algorithm.bootstrap_input_strategy"] == "signed_split"
    assert options["--algorithm.bootstrap_input_weight"] == "2.0"
    assert options["--algorithm.bootstrap_input_bias"] == "0.0"
    assert options["--algorithm.record"] == "True"
    assert options["--environment.nr_envs"] == "1"
    assert options["--environment.multi_render"] == "False"


def test_record_robot_index_validation_message():
    with pytest.raises(ValueError, match=f"0 to {len(RECORD_ROBOTS) - 1}"):
        run.build_command("test_default", extra_args=[], record_robot_index=len(RECORD_ROBOTS))


def test_yaml_backed_presets_include_shell_wrapper_targets():
    assert {
        "collect_data",
        "show_config_h1",
        "show_model_h1",
        "train_full",
    }.issubset(PRESETS)


def test_robot_type_lists_load_as_tuples():
    assert PRESETS["record_default"]["environment.train_robot_types"] == ("unitree_h1",)
    assert PRESETS["record_default"]["algorithm.rollout_policy_stage"] == "teacher"


def test_invalid_preset_config_fails_with_clear_error(tmp_path):
    config_path = tmp_path / "presets.yaml"
    config_path.write_text(
        """
record_robots: [unitree_a1]
base_experiment:
  algorithm.name: uni_ppo.ppo
  environment.name: multi_robot
  runner.mode: test
  environment.mode: test
presets:
  bad_backend:
    algorithm.student_backend: lava
""".strip()
    )

    with pytest.raises(ValueError, match="student_backend"):
        load_presets(config_path)


def test_build_resolved_config_contains_options_and_extra_args():
    resolved = run.build_resolved_config("test_default", ["--runner.seed=123"], force_record=True)

    assert resolved["preset"] == "test_default"
    assert resolved["options"]["algorithm.record"] == "True"
    assert resolved["options"]["algorithm.name"] == "uni_ppo.ppo"
    assert resolved["extra_args"] == ["--runner.seed=123"]


def test_write_resolved_config_round_trip(tmp_path):
    resolved = run.build_resolved_config("test_default", [])
    output_path = run.write_resolved_config(resolved, tmp_path / "resolved.json")

    loaded = json.loads(output_path.read_text())
    assert loaded["preset"] == "test_default"
    assert loaded["options"]["environment.name"] == "multi_robot"



def test_experiment_entrypoint_adds_repo_root_to_import_path(monkeypatch):
    import sys
    from experiments import experiment

    root = str(experiment.ROOT)
    student_dir = str(experiment.STUDENT_DIR)
    monkeypatch.setattr(sys, "path", [path for path in sys.path if path not in {root, student_dir}])

    experiment.ensure_local_import_paths()

    assert root in sys.path
    assert student_dir in sys.path
