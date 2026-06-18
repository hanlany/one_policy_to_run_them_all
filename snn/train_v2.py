#!/usr/bin/env python3
"""Train the URMA bootstrap student as a pure SNN.

This is the script version of train_v2.ipynb. Hyperparameters live in a YAML
config so architecture, neuron dynamics, optimization, validation cadence, and
output paths can be changed without editing code.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Iterable

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import yaml

SNN_DIR = Path(__file__).resolve().parent
LAVA_SRC = Path("/app/lava-dl/src")
for path in (SNN_DIR, LAVA_SRC):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import lava.lib.dl.bootstrap as bootstrap  # noqa: E402


@dataclass
class PathConfig:
    dataset: str = "teacher_student_dagger_dataset.npz"
    ann_checkpoint: str = "student_model_latest.pth"
    output_dir: str = "Trained"
    checkpoint_name: str = "network.pt"
    export_name: str = "network.net"
    plot_name: str = "snn_training_curves.png"


@dataclass
class NeuronConfig:
    threshold: float = 0.5
    current_decay: float = 0.3
    voltage_decay: float = 0.02
    tau_grad: float = 1.0
    scale_grad: float = 1.0


@dataclass
class ModelConfig:
    input_dim: int = 668
    output_dim: int = 24
    hidden_dims: list[int] = field(default_factory=lambda: [1024, 1024, 1024, 1024, 1024])
    time_steps: int = 5
    input_strategy: str = "signed_split"
    input_weight: float = 2.0
    input_bias: float = 0.0
    weight_norm: bool = False
    weight_scale: float = 1.0
    delay_shift: bool = False
    neuron: NeuronConfig = field(default_factory=NeuronConfig)


@dataclass
class TrainingConfig:
    epochs: int = 100
    train_batch_size: int = 1024
    val_batch_size: int = 1024
    learning_rate: float = 1e-3
    val_split: float = 0.2
    val_eval_samples: int = 10_000
    full_val_interval: int = 10
    seed: int = 0
    val_subset_seed: int = 1
    num_workers: int = 0
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    pin_memory: bool | None = None


@dataclass
class LRSchedulerConfig:
    enabled: bool = True
    factor: float = 0.5
    patience: int = 5
    threshold: float = 1e-4
    min_lr: float = 1e-5


@dataclass
class RuntimeConfig:
    device: str = "auto"
    export_hdf5: bool = True
    save_plot: bool = True


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def dataclass_to_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: dataclass_to_dict(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, list):
        return [dataclass_to_dict(item) for item in value]
    return value


def build_config(data: dict[str, Any]) -> Config:
    neuron = NeuronConfig(**data.get("model", {}).get("neuron", {}))
    model_data = dict(data.get("model", {}))
    model_data.pop("neuron", None)
    return Config(
        paths=PathConfig(**data.get("paths", {})),
        model=ModelConfig(**model_data, neuron=neuron),
        training=TrainingConfig(**data.get("training", {})),
        lr_scheduler=LRSchedulerConfig(**data.get("lr_scheduler", {})),
        runtime=RuntimeConfig(**data.get("runtime", {})),
    )


def parse_override(raw: str) -> tuple[list[str], Any]:
    if "=" not in raw:
        raise ValueError(f"Override must be key=value, got {raw!r}")
    key, value = raw.split("=", 1)
    return key.split("."), yaml.safe_load(value)


def apply_overrides(data: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    for raw in overrides:
        keys, value = parse_override(raw)
        cursor = data
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[keys[-1]] = value
    return data


def load_config(path: Path, overrides: Iterable[str]) -> Config:
    defaults = dataclass_to_dict(Config())
    loaded = yaml.safe_load(path.read_text()) if path.exists() else None
    if loaded:
        deep_update(defaults, loaded)
    apply_overrides(defaults, overrides)
    return build_config(defaults)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else SNN_DIR / path


class TeacherStudentDataset(Dataset):
    def __init__(self, states: np.ndarray | torch.Tensor, actions: np.ndarray | torch.Tensor):
        self.states = torch.as_tensor(states, dtype=torch.float32)
        self.actions = torch.as_tensor(actions, dtype=torch.float32)

    @classmethod
    def from_npz(cls, npz_file: str | Path) -> "TeacherStudentDataset":
        data = np.load(npz_file)
        return cls(states=data["states"], actions=data["actions"])

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.states[idx], self.actions[idx]


class Network(torch.nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_dim = int(config.input_dim)
        self.output_dim = int(config.output_dim)
        self.hidden_dims = tuple(int(dim) for dim in config.hidden_dims)
        self.time_steps = int(config.time_steps)
        self.input_strategy = config.input_strategy
        if self.input_strategy not in {"identity", "signed_split"}:
            raise ValueError(f"Unsupported input_strategy: {self.input_strategy}")
        self.encoded_input_dim = self.input_dim * 2 if self.input_strategy == "signed_split" else self.input_dim

        neuron_params = {
            "threshold": config.neuron.threshold,
            "current_decay": config.neuron.current_decay,
            "voltage_decay": config.neuron.voltage_decay,
            "tau_grad": config.neuron.tau_grad,
            "scale_grad": config.neuron.scale_grad,
        }
        neuron_params_norm = dict(neuron_params)

        blocks = [
            bootstrap.block.cuba.Input(
                neuron_params,
                weight=config.input_weight,
                bias=config.input_bias,
                delay_shift=config.delay_shift,
            ),
            bootstrap.block.cuba.Flatten(),
        ]
        prev_dim = self.encoded_input_dim
        for hidden_dim in self.hidden_dims:
            blocks.append(
                bootstrap.block.cuba.Dense(
                    neuron_params_norm,
                    prev_dim,
                    hidden_dim,
                    weight_norm=config.weight_norm,
                    weight_scale=config.weight_scale,
                    delay_shift=config.delay_shift,
                )
            )
            prev_dim = hidden_dim
        blocks.append(
            bootstrap.block.cuba.Affine(
                neuron_params,
                prev_dim,
                self.output_dim,
                weight_norm=config.weight_norm,
                weight_scale=config.weight_scale,
                dynamics=False,
            )
        )
        self.blocks = torch.nn.ModuleList(blocks)

    def initialize_from_ann_checkpoint(self, checkpoint_path: str | Path, device: torch.device | None = None) -> "Network":
        checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        weight_keys = sorted(
            [key for key in state_dict if key.startswith("net.") and key.endswith(".weight")],
            key=lambda key: int(key.split(".")[1]),
        )
        bootstrap_layers = [block for block in self.blocks if hasattr(block, "synapse")]
        if len(weight_keys) != len(bootstrap_layers):
            raise ValueError(
                f"ANN has {len(weight_keys)} linear layers, bootstrap network has {len(bootstrap_layers)} weighted layers."
            )

        for layer_index, (weight_key, bootstrap_layer) in enumerate(zip(weight_keys, bootstrap_layers)):
            weight = state_dict[weight_key].detach().to(
                bootstrap_layer.synapse.weight.device,
                dtype=bootstrap_layer.synapse.weight.dtype,
            )
            if layer_index == 0 and self.input_strategy == "signed_split":
                weight = torch.cat([weight, -weight], dim=1)
            weight = weight.reshape(weight.shape[0], weight.shape[1], 1, 1, 1)
            if weight.shape != bootstrap_layer.synapse.weight.shape:
                raise ValueError(f"{weight_key} has mapped shape {tuple(weight.shape)}, expected {tuple(bootstrap_layer.synapse.weight.shape)}.")
            bootstrap_layer.synapse.weight.data.copy_(weight)
        return self

    def encode_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} student observation features, received {x.shape[1]}.")
        if self.input_strategy == "identity":
            return x
        return torch.cat([torch.relu(x), torch.relu(-x)], dim=1)

    def forward(self, x: torch.Tensor, mode: bootstrap.routine.LayerMode) -> torch.Tensor:
        x = self.encode_input(x)
        x = x.reshape(x.shape[0], self.encoded_input_dim, 1, 1, 1)

        if mode.base_mode != bootstrap.Mode.ANN and self.time_steps > 1:
            x = x.repeat(1, 1, 1, 1, self.time_steps)

        for block, block_mode in zip(self.blocks, mode):
            x = block(x, mode=block_mode)

        while x.ndim > 3 and x.shape[-2] == 1:
            x = x.squeeze(-2)
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        return x

    def export_hdf5(self, filename: str | Path) -> Path:
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(filename, "w") as handle:
            simulation = handle.create_group("simulation")
            simulation["Ts"] = 1
            simulation["tSample"] = self.time_steps
            layer = handle.create_group("layer")
            for index, block in enumerate(self.blocks):
                block.export_hdf5(layer.create_group(f"{index}"))
        return filename


def resolve_device(config: RuntimeConfig) -> torch.device:
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.device)


def subset_dataset(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, range(max_samples))


def make_loaders(config: Config) -> tuple[Dataset, Dataset, Dataset, DataLoader, DataLoader, DataLoader]:
    dataset = TeacherStudentDataset.from_npz(resolve_path(config.paths.dataset))
    generator = torch.Generator().manual_seed(config.training.seed)
    val_size = max(1, int(len(dataset) * config.training.val_split))
    train_size = len(dataset) - val_size
    training_set, testing_set = random_split(dataset, [train_size, val_size], generator=generator)

    training_set = subset_dataset(training_set, config.training.max_train_samples)
    testing_set = subset_dataset(testing_set, config.training.max_val_samples)

    val_eval_size = min(config.training.val_eval_samples, len(testing_set))
    val_holdout_size = len(testing_set) - val_eval_size
    if val_holdout_size > 0:
        val_eval_set, _ = random_split(
            testing_set,
            [val_eval_size, val_holdout_size],
            generator=torch.Generator().manual_seed(config.training.val_subset_seed),
        )
    else:
        val_eval_set = testing_set

    device = resolve_device(config.runtime)
    pin_memory = config.training.pin_memory if config.training.pin_memory is not None else device.type == "cuda"
    train_loader = DataLoader(
        dataset=training_set,
        batch_size=config.training.train_batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        dataset=val_eval_set,
        batch_size=config.training.val_batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=pin_memory,
    )
    full_val_loader = DataLoader(
        dataset=testing_set,
        batch_size=config.training.val_batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=pin_memory,
    )
    return dataset, training_set, testing_set, train_loader, val_loader, full_val_loader


def pure_snn_rate(model: Network, states: torch.Tensor) -> torch.Tensor:
    snn_mode = bootstrap.routine.LayerMode(0, bootstrap.Mode.SNN)
    output = model.forward(states, snn_mode)
    return torch.mean(output, dim=-1).reshape((states.shape[0], -1))


def evaluate_snn_mse(model: Network, loader: DataLoader, device: torch.device, pin_memory: bool) -> float:
    model.eval()
    loss_sum = 0.0
    samples = 0
    with torch.no_grad():
        for states, actions in loader:
            states = states.to(device, non_blocking=pin_memory)
            actions = actions.to(device, non_blocking=pin_memory)
            rate = pure_snn_rate(model, states)
            loss = F.mse_loss(rate, actions)
            loss_sum += loss.item() * states.shape[0]
            samples += states.shape[0]
    return loss_sum / max(1, samples)


def save_training_plot(history: dict[str, list[Any]], plot_path: Path) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, loss_ax = plt.subplots(figsize=(10, 4))
    loss_ax.plot(history["train_snn_loss"], label="train SNN MSE")
    loss_ax.plot(history["val_snn_loss"], label="validation subset SNN MSE")
    full_val_epochs = [index for index, value in enumerate(history["full_val_snn_loss"]) if value is not None]
    full_val_losses = [value for value in history["full_val_snn_loss"] if value is not None]
    loss_ax.plot(full_val_epochs, full_val_losses, marker="o", linestyle="none", label="full validation SNN MSE")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("SNN MSE loss")
    loss_ax.grid(True, alpha=0.3)

    lr_ax = loss_ax.twinx()
    lr_ax.plot(history["learning_rate"], color="tab:gray", linestyle="--", alpha=0.7, label="learning rate")
    lr_ax.set_ylabel("Learning rate")
    lr_ax.set_yscale("log")

    lines, labels = loss_ax.get_legend_handles_labels()
    lr_lines, lr_labels = lr_ax.get_legend_handles_labels()
    loss_ax.legend(lines + lr_lines, labels + lr_labels, loc="best")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def train(config: Config) -> dict[str, Any]:
    device = resolve_device(config.runtime)
    output_dir = resolve_path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / config.paths.checkpoint_name
    export_path = output_dir / config.paths.export_name
    plot_path = output_dir / config.paths.plot_name
    pin_memory = config.training.pin_memory if config.training.pin_memory is not None else device.type == "cuda"

    dataset, training_set, testing_set, train_loader, val_loader, full_val_loader = make_loaders(config)

    model = Network(config.model).to(device)
    model.initialize_from_ann_checkpoint(resolve_path(config.paths.ann_checkpoint), device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    lr_scheduler = None
    if config.lr_scheduler.enabled:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.lr_scheduler.factor,
            patience=config.lr_scheduler.patience,
            threshold=config.lr_scheduler.threshold,
            min_lr=config.lr_scheduler.min_lr,
        )

    sample_state, sample_action = dataset[0]
    print(f"device: {device}")
    print(f"dataset: {resolve_path(config.paths.dataset)}")
    print(f"ann checkpoint: {resolve_path(config.paths.ann_checkpoint)}")
    print(f"output dir: {output_dir}")
    print(f"state shape: {tuple(sample_state.shape)} action shape: {tuple(sample_action.shape)}")
    print(f"train samples: {len(training_set)} validation samples: {len(testing_set)}")
    print(f"per-epoch validation subset: {len(val_loader.dataset)} full validation interval: {config.training.full_val_interval}")
    print(
        f"batch sizes: train={config.training.train_batch_size} validation={config.training.val_batch_size} "
        f"learning_rate={config.training.learning_rate}"
    )
    if lr_scheduler is not None:
        print(
            f"lr plateau scheduler: factor={config.lr_scheduler.factor} patience={config.lr_scheduler.patience} "
            f"threshold={config.lr_scheduler.threshold} min_lr={config.lr_scheduler.min_lr}"
        )

    best_snn_val_loss = float("inf")
    history: dict[str, list[Any]] = {
        "train_snn_loss": [],
        "val_snn_loss": [],
        "full_val_snn_loss": [],
        "learning_rate": [],
        "output_activity": [],
    }

    for epoch in range(config.training.epochs):
        model.train()
        train_snn_loss_sum = 0.0
        train_output_abs_sum = 0.0
        train_output_numel = 0
        train_samples = 0

        for states, actions in train_loader:
            states = states.to(device, non_blocking=pin_memory)
            actions = actions.to(device, non_blocking=pin_memory)

            rate = pure_snn_rate(model, states)
            loss = F.mse_loss(rate, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_snn_loss_sum += loss.detach().item() * states.shape[0]
            train_output_abs_sum += rate.detach().abs().sum().item()
            train_output_numel += rate.numel()
            train_samples += states.shape[0]

        train_snn_loss = train_snn_loss_sum / max(1, train_samples)
        output_activity = train_output_abs_sum / max(1, train_output_numel)
        history["train_snn_loss"].append(train_snn_loss)
        history["output_activity"].append(output_activity)

        val_snn_loss = evaluate_snn_mse(model, val_loader, device, pin_memory)
        history["val_snn_loss"].append(val_snn_loss)

        run_full_val = epoch == 0 or (epoch + 1) % config.training.full_val_interval == 0 or epoch == config.training.epochs - 1
        full_val_snn_loss = None
        checkpoint_note = ""
        if run_full_val:
            full_val_snn_loss = evaluate_snn_mse(model, full_val_loader, device, pin_memory)
            if full_val_snn_loss < best_snn_val_loss:
                best_snn_val_loss = full_val_snn_loss
                torch.save(model.state_dict(), checkpoint_path)
                checkpoint_note = " saved_best"

        scheduler_metric = full_val_snn_loss if full_val_snn_loss is not None else val_snn_loss
        if lr_scheduler is not None:
            lr_scheduler.step(scheduler_metric)
        current_lr = optimizer.param_groups[0]["lr"]
        history["learning_rate"].append(current_lr)
        history["full_val_snn_loss"].append(full_val_snn_loss)
        full_val_text = "n/a" if full_val_snn_loss is None else f"{full_val_snn_loss:.9f}"

        print(
            f"[Epoch {epoch + 1:03d}/{config.training.epochs}] "
            f"train_snn_mse={train_snn_loss:.9f} val_subset_snn_mse={val_snn_loss:.9f} "
            f"full_val_snn_mse={full_val_text} best_snn_val_mse={best_snn_val_loss:.9f} "
            f"lr={current_lr:.2e} output_abs_mean={output_activity:.6f}{checkpoint_note}"
        )

    if config.runtime.save_plot:
        save_training_plot(history, plot_path)
        print(f"Saved training plot to {plot_path}")

    if config.runtime.export_hdf5:
        if checkpoint_path.exists():
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.export_hdf5(export_path)
        print(f"Exported network to {export_path}")

    return {
        "best_snn_val_loss": best_snn_val_loss,
        "checkpoint_path": checkpoint_path,
        "export_path": export_path,
        "plot_path": plot_path,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train URMA bootstrap student as a pure SNN.")
    parser.add_argument("--config", type=Path, default=SNN_DIR / "train_v2_config_default.yaml", help="YAML training config path.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, e.g. --set training.epochs=5 --set model.time_steps=3",
    )
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    train(config)


if __name__ == "__main__":
    main()
