from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import run
from snn import train_v2
from student import bootstrap_backend


def small_model_config(scope="all"):
    return train_v2.ModelConfig(
        input_dim=3,
        output_dim=2,
        hidden_dims=[4],
        time_steps=2,
        input_strategy="signed_split",
        input_weight=1.0,
        input_bias=0.0,
        init_policy="none",
        neuron=train_v2.NeuronConfig(
            threshold=0.2,
            current_decay=0.3,
            voltage_decay=0.02,
        ),
        weight_quantization=train_v2.WeightQuantizationConfig(
            mode="decomposed",
            target_bits=24,
            chunk_bits=8,
            sign_mode="mixed",
            scope=scope,
        ),
    )


def selected_hooks(model):
    return [
        bool(
            getattr(
                block.synapse.pre_hook_fx,
                "is_decomposed_weight_quantizer",
                False,
            )
        )
        for block in model.blocks
        if hasattr(block, "synapse")
    ]


def test_quantrick_config_and_cli_override():
    config = train_v2.load_config(
        Path("snn/train_v2_config_quantrick.yaml"),
        ["model.weight_quantization.scope=first"],
    )
    assert config.model.weight_quantization.mode == "decomposed"
    assert config.model.weight_quantization.target_bits == 24
    assert config.model.weight_quantization.chunk_bits == 8
    assert config.model.weight_quantization.sign_mode == "mixed"
    assert config.model.weight_quantization.scope == "first"
    assert config.runtime.export_hdf5 is False


@pytest.mark.parametrize(
    "override, match",
    [
        ("model.weight_quantization.mode=unknown", "mode"),
        ("model.weight_quantization.scope=unknown", "scope"),
        ("model.weight_quantization.sign_mode=unknown", "sign mode"),
        ("model.weight_quantization.chunk_bits=7", "divisible"),
        ("runtime.export_hdf5=true", "legacy_8bit"),
    ],
)
def test_invalid_quantrick_config_is_rejected(override, match):
    with pytest.raises(ValueError, match=match):
        train_v2.load_config(
            Path("snn/train_v2_config_quantrick.yaml"), [override]
        )


def test_scope_installs_only_selected_synapses():
    assert selected_hooks(train_v2.Network(small_model_config("all"))) == [
        True,
        True,
    ]
    assert selected_hooks(train_v2.Network(small_model_config("first"))) == [
        True,
        False,
    ]


def test_metadata_checkpoint_precedence_and_rollout_parity(tmp_path):
    torch.manual_seed(7)
    trainer_model = train_v2.Network(small_model_config()).eval()
    checkpoint_path = tmp_path / "metadata.pt"
    torch.save(trainer_model.checkpoint_payload(), checkpoint_path)

    rollout_model, metadata = (
        bootstrap_backend.load_bootstrap_policy_from_checkpoint(
            checkpoint_path,
            device="cpu",
            weight_quantization_mode="legacy_8bit",
        )
    )
    states = torch.randn(8, 3)
    trainer_output = train_v2.pure_snn_rate(trainer_model, states)
    rollout_output = rollout_model(states, mode="snn")

    torch.testing.assert_close(
        trainer_output, rollout_output, atol=1e-6, rtol=0
    )
    assert metadata["weight_quantization"] == {
        "mode": "decomposed",
        "target_bits": 24,
        "chunk_bits": 8,
        "sign_mode": "mixed",
        "scope": "all",
    }
    assert selected_hooks(rollout_model) == [True, True]
    assert torch.isfinite(rollout_output).all()


def test_raw_checkpoint_uses_explicit_quantization_compatibility(tmp_path):
    source_model = bootstrap_backend.BootstrapStudentPolicy(
        input_dim=3,
        output_dim=2,
        hidden_dims=[4],
        timesteps=2,
    )
    checkpoint_path = tmp_path / "raw.pt"
    torch.save(source_model.state_dict(), checkpoint_path)

    loaded_model, metadata = (
        bootstrap_backend.load_bootstrap_policy_from_checkpoint(
            checkpoint_path,
            input_dim=3,
            output_dim=2,
            hidden_dims=[4],
            timesteps=2,
            weight_quantization_mode="decomposed",
            weight_quantization_target_bits=24,
            weight_quantization_chunk_bits=8,
            weight_quantization_sign_mode="mixed",
            weight_quantization_scope="first",
        )
    )
    assert metadata["weight_quantization"]["scope"] == "first"
    assert selected_hooks(loaded_model) == [True, False]


def test_one_epoch_cpu_optimization_smoke(tmp_path):
    rng = np.random.default_rng(11)
    dataset_path = tmp_path / "dataset.npz"
    np.savez(
        dataset_path,
        states=rng.normal(size=(12, 3)).astype(np.float32),
        actions=rng.normal(size=(12, 2)).astype(np.float32),
    )
    config = train_v2.Config(
        paths=train_v2.PathConfig(
            dataset=str(dataset_path),
            ann_checkpoint=str(tmp_path / "unused.pt"),
            output_dir=str(tmp_path / "output"),
            checkpoint_name="network.pt",
            history_name="history.json",
        ),
        model=small_model_config(),
        training=train_v2.TrainingConfig(
            epochs=1,
            train_batch_size=4,
            val_batch_size=4,
            val_eval_samples=4,
            full_val_interval=1,
            seed=0,
            val_subset_seed=1,
        ),
        lr_scheduler=train_v2.LRSchedulerConfig(enabled=False),
        bootstrap_training=train_v2.BootstrapTrainingConfig(mode="pure_snn"),
        runtime=train_v2.RuntimeConfig(
            device="cpu", export_hdf5=False, save_plot=False
        ),
    )

    result = train_v2.train(config)
    history = result["history"]
    assert np.isfinite(history["train_snn_loss"]).all()
    assert np.isfinite(history["val_snn_loss"]).all()
    assert history["gradient_norm"][0] > 0
    final_layers = history["quantization_diagnostics"][-1]["layers"]
    assert all(layer["raw_finite"] for layer in final_layers)
    assert all(layer["quantized_finite"] for layer in final_layers)
    assert all(layer["saturation_count"] == 0 for layer in final_layers)

    checkpoint = torch.load(result["checkpoint_path"], map_location="cpu")
    assert "state_dict" in checkpoint
    assert checkpoint["weight_quantization"]["mode"] == "decomposed"
    assert checkpoint["architecture"]["hidden_dims"] == [4]


def test_quantrick_recording_preset_and_wrapper():
    command = run.build_command(
        "record_snnv2_quantrick", extra_args=[]
    )
    options = {
        part.split("=", 1)[0]: part.split("=", 1)[1]
        for part in command
        if part.startswith("--") and "=" in part
    }
    assert options["--algorithm.bootstrap_timesteps"] == "20"
    assert (
        options["--algorithm.bootstrap_weight_quantization_mode"]
        == "decomposed"
    )
    assert (
        options["--algorithm.bootstrap_weight_quantization_target_bits"]
        == "24"
    )
    assert (
        options["--algorithm.bootstrap_weight_quantization_chunk_bits"]
        == "8"
    )
    assert (
        options["--algorithm.bootstrap_weight_quantization_sign_mode"]
        == "mixed"
    )
    assert options["--algorithm.bootstrap_weight_quantization_scope"] == "all"
    assert options["--algorithm.student_model_path"].endswith(
        "/snn/experiments/snn_v2_quantrick/20260722T053851Z/"
        "accepted/quantrick_network.pt"
    )

    wrapper = Path(
        "validate/record_snnv2_quantrick_all_robots.sh"
    ).read_text()
    assert "--preset record_snnv2_quantrick --list-record-robots" in wrapper
    assert "--preset record_snnv2_quantrick --record-robot" in wrapper
    assert "before_all_videos.txt" in wrapper
    assert "video_mapping.tsv" in wrapper
    assert 'comm -13 "${BEFORE_PATH}" "${AFTER_PATH}"' in wrapper
    assert '"${#NEW_VIDEOS[@]}" -ne 1' in wrapper
    assert "validate_quantrick_videos.py" in wrapper
    assert 'video_validation.json' in wrapper
    assert 'contact_sheets' in wrapper
    assert 'verify_quantrick_accepted_bundle.py' in wrapper
    assert 'bundle_verification.json' in wrapper

