from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_shell_wrappers_use_module_entrypoint():
    wrapper_presets = {
        "collect_data.sh": "collect_data",
        "collect_data_dagger.sh": "collect_data_dagger",
        "show_config.sh": "show_config_h1",
        "show_model.sh": "show_model_h1",
        "test.sh": "test_default",
        "test_snn.sh": "test_snn",
        "test_student.sh": "test_student",
        "train_dagger.sh": "train_dagger",
        "train_dagger_snn.sh": "train_dagger_snn",
    }

    for script_name, preset in wrapper_presets.items():
        script = (REPO_ROOT / "experiments" / script_name).read_text()
        assert "python3 -m experiments.run" in script
        assert f"--preset {preset}" in script
        assert '"$@"' in script


def test_slurm_wrapper_delegates_to_train_preset():
    script = (REPO_ROOT / "experiments" / "experiment.sh").read_text()

    assert "#SBATCH" in script
    assert "python3 -m experiments.run" in script
    assert "--preset train_full" in script
