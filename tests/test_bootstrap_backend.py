import numpy as np
import pytest
import torch

from student import bootstrap_backend as backend


pytestmark = pytest.mark.skipif(
    backend.lava_bootstrap_block is None or backend.lava_bootstrap_routine is None,
    reason="lava.lib.dl.bootstrap is not available",
)


def test_raw_train_v2_state_dict_loads_and_pure_snn_training_runs(tmp_path):
    input_dim = 3
    output_dim = 2
    hidden_dims = [4]

    source_model = backend.BootstrapStudentPolicy(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        timesteps=2,
        readout="mean",
        neuron_threshold=0.2,
        current_decay=0.3,
        voltage_decay=0.02,
        input_strategy="signed_split",
        input_weight=1.0,
        input_bias=0.0,
    )
    raw_checkpoint = tmp_path / "network.pt"
    torch.save(source_model.state_dict(), raw_checkpoint)

    loaded_model, architecture = backend.load_bootstrap_policy_from_checkpoint(
        raw_checkpoint,
        device="cpu",
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        timesteps=2,
        readout="mean",
        neuron_threshold=0.2,
        current_decay=0.3,
        voltage_decay=0.02,
        input_strategy="signed_split",
        input_weight=1.0,
        input_bias=0.0,
    )

    assert architecture["input_dim"] == input_dim
    assert architecture["output_dim"] == output_dim
    assert architecture["hidden_dims"] == hidden_dims

    states = np.random.default_rng(0).normal(size=(4, input_dim)).astype(np.float32)
    actions = np.random.default_rng(1).normal(size=(4, output_dim)).astype(np.float32)
    config = backend.BootstrapTrainingConfig(
        batch_size=2,
        learning_rate=1e-3,
        epochs=1,
        hidden_dims=hidden_dims,
        val_split=0.5,
        checkpoint_dir=str(tmp_path),
        timesteps=2,
        readout="mean",
        neuron_threshold=0.2,
        current_decay=0.3,
        voltage_decay=0.02,
        input_strategy="signed_split",
        input_weight=1.0,
        input_bias=0.0,
        training_mode="pure_snn",
        lr_scheduler_enabled=True,
    )
    trainer = backend.BootstrapStudentTrainer(config=config, device="cpu")

    artifacts = trainer.train_on_arrays(states, actions, iteration=1, initial_model=loaded_model)

    assert artifacts.last_train_loss >= 0.0
    assert artifacts.last_val_loss >= 0.0
    assert artifacts.iteration_checkpoint_path is not None
    latest_checkpoint = torch.load(artifacts.latest_checkpoint_path, map_location="cpu")
    assert latest_checkpoint["training_mode"] == "pure_snn"
