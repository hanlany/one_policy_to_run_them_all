#!/usr/bin/env python3
"""Train the URMA bootstrap student as a pure SNN.

This is the script version of train_v2.ipynb. Hyperparameters live in a YAML
config so architecture, neuron dynamics, optimization, validation cadence, and
output paths can be changed without editing code.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
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
from lava.lib.dl.slayer.utils import (  # noqa: E402
    DecomposedWeightQuantizer,
    SignMode,
)

SUPPORTED_INIT_POLICIES = {"exact", "partial", "none"}
SUPPORTED_BOOTSTRAP_TRAINING_MODES = {"pure_snn", "scheduler"}
SUPPORTED_WEIGHT_QUANTIZATION_MODES = {"legacy_8bit", "decomposed"}
SUPPORTED_WEIGHT_QUANTIZATION_SCOPES = {"all", "first"}


@dataclass
class PathConfig:
    dataset: str = "teacher_student_dagger_dataset.npz"
    ann_checkpoint: str = "student_model_latest.pth"
    output_dir: str = "Trained"
    checkpoint_name: str = "network.pt"
    export_name: str = "network.net"
    plot_name: str = "snn_training_curves.png"
    history_name: str = "snn_training_history.json"


@dataclass
class NeuronConfig:
    threshold: float = 0.5
    current_decay: float = 0.3
    voltage_decay: float = 0.02
    tau_grad: float = 1.0
    scale_grad: float = 1.0


@dataclass
class WeightQuantizationConfig:
    mode: str = "legacy_8bit"
    target_bits: int = 24
    chunk_bits: int = 8
    sign_mode: str = "mixed"
    scope: str = "all"



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
    weight_quantization: WeightQuantizationConfig = field(default_factory=WeightQuantizationConfig)
    init_policy: str = "exact"
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
class BootstrapTrainingConfig:
    mode: str = "pure_snn"
    num_sample_iter: int = 10
    sample_period: int = 10
    crossover_epochs: list[int] = field(default_factory=list)


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
    bootstrap_training: BootstrapTrainingConfig = field(default_factory=BootstrapTrainingConfig)
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
    weight_quantization = WeightQuantizationConfig(
        **data.get("model", {}).get("weight_quantization", {})
    )
    model_data = dict(data.get("model", {}))
    model_data.pop("neuron", None)
    model_data.pop("weight_quantization", None)
    return Config(
        paths=PathConfig(**data.get("paths", {})),
        model=ModelConfig(**model_data, neuron=neuron, weight_quantization=weight_quantization),
        training=TrainingConfig(**data.get("training", {})),
        lr_scheduler=LRSchedulerConfig(**data.get("lr_scheduler", {})),
        bootstrap_training=BootstrapTrainingConfig(**data.get("bootstrap_training", {})),
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
    config = build_config(defaults)
    validate_config(config)
    return config

def validate_config(config: Config) -> None:
    quantization = config.model.weight_quantization
    if quantization.mode not in SUPPORTED_WEIGHT_QUANTIZATION_MODES:
        raise ValueError(
            f"Unsupported weight quantization mode {quantization.mode!r}; "
            f"expected one of {sorted(SUPPORTED_WEIGHT_QUANTIZATION_MODES)}"
        )
    if quantization.scope not in SUPPORTED_WEIGHT_QUANTIZATION_SCOPES:
        raise ValueError(
            f"Unsupported weight quantization scope {quantization.scope!r}; "
            f"expected one of {sorted(SUPPORTED_WEIGHT_QUANTIZATION_SCOPES)}"
        )
    try:
        SignMode(quantization.sign_mode)
    except ValueError as error:
        raise ValueError(
            f"Unsupported weight quantization sign mode "
            f"{quantization.sign_mode!r}"
        ) from error
    if quantization.target_bits <= 0 or quantization.chunk_bits <= 0:
        raise ValueError("Weight quantization bit widths must be positive")
    if quantization.target_bits % quantization.chunk_bits:
        raise ValueError(
            "weight_quantization.target_bits must be divisible by "
            "weight_quantization.chunk_bits"
        )
    if quantization.mode == "decomposed" and config.runtime.export_hdf5:
        raise ValueError(
            "runtime.export_hdf5=true is unsupported with decomposed "
            "weights. Disable HDF5 export or select legacy_8bit mode."
        )




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
        self.neuron_settings = {
            "threshold": float(config.neuron.threshold),
            "current_decay": float(config.neuron.current_decay),
            "voltage_decay": float(config.neuron.voltage_decay),
            "tau_grad": float(config.neuron.tau_grad),
            "scale_grad": float(config.neuron.scale_grad),
        }
        self.input_weight = float(config.input_weight)
        self.input_bias = float(config.input_bias)
        self.weight_quantization = dataclass_to_dict(
            config.weight_quantization
        )
        self.quantized_synapses: list[tuple[int, Any]] = []
        weighted_blocks = [
            block for block in self.blocks if hasattr(block, "synapse")
        ]
        if config.weight_quantization.mode == "decomposed":
            selected = (
                weighted_blocks
                if config.weight_quantization.scope == "all"
                else weighted_blocks[:1]
            )
            for layer_index, block in enumerate(weighted_blocks):
                if block not in selected:
                    continue
                quantizer = DecomposedWeightQuantizer(
                    target_bits=config.weight_quantization.target_bits,
                    chunk_bits=config.weight_quantization.chunk_bits,
                    sign_mode=config.weight_quantization.sign_mode,
                    scale=64,
                )
                block.synapse.pre_hook_fx = quantizer
                self.quantized_synapses.append((layer_index, block.synapse))



    def quantization_diagnostics(self) -> list[dict[str, Any]]:
        diagnostics = []
        weighted_blocks = [
            block for block in self.blocks if hasattr(block, "synapse")
        ]
        for layer_index, block in enumerate(weighted_blocks):
            synapse = block.synapse
            raw = synapse.weight.detach()
            hook = synapse.pre_hook_fx
            with torch.no_grad():
                quantized = hook(raw) if hook is not None else raw
            hook_diagnostics = dict(
                getattr(hook, "last_diagnostics", {})
            )
            diagnostics.append({
                "layer_index": layer_index,
                "layer_type": type(block).__name__,
                "selected": bool(
                    getattr(hook, "is_decomposed_weight_quantizer", False)
                ),
                "raw_min": float(raw.min().item()),
                "raw_max": float(raw.max().item()),
                "quantized_min": float(quantized.min().item()),
                "quantized_max": float(quantized.max().item()),
                "raw_finite": bool(torch.isfinite(raw).all().item()),
                "quantized_finite": bool(
                    torch.isfinite(quantized).all().item()
                ),
                "reconstruction_error_count": int(
                    hook_diagnostics.get("reconstruction_error_count", 0)
                ),
                "max_reconstruction_error": int(
                    hook_diagnostics.get("max_reconstruction_error", 0)
                ),
                "saturation_count": int(
                    hook_diagnostics.get("saturation_count", 0)
                ),
                "max_quantization_error": float(
                    hook_diagnostics.get("max_quantization_error", 0.0)
                ),
            })
        return diagnostics

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dims": list(self.hidden_dims),
            "timesteps": self.time_steps,
            "readout": "mean",
            "neuron_threshold": self.neuron_settings["threshold"],
            "current_decay": self.neuron_settings["current_decay"],
            "voltage_decay": self.neuron_settings["voltage_decay"],
            "neuron_settings": dict(self.neuron_settings),
            "input_strategy": self.input_strategy,
            "input_weight": self.input_weight,
            "input_bias": self.input_bias,
            "weight_quantization": dict(self.weight_quantization),
            "architecture": {
                "input_dim": self.input_dim,
                "encoded_input_dim": self.encoded_input_dim,
                "hidden_dims": list(self.hidden_dims),
                "output_dim": self.output_dim,
            },
            "quantization_diagnostics": self.quantization_diagnostics(),
            "backend": "bootstrap",
        }


    def initialize_from_ann_checkpoint(
        self,
        checkpoint_path: str | Path,
        device: torch.device | None = None,
        policy: str = "exact",
    ) -> dict[str, Any]:
        if policy not in SUPPORTED_INIT_POLICIES:
            raise ValueError(f"Unsupported init_policy {policy!r}. Supported policies: {sorted(SUPPORTED_INIT_POLICIES)}")
        if policy == "none":
            return {
                "policy": policy,
                "checkpoint_path": str(checkpoint_path),
                "copied_layers": [],
                "skipped_layers": [],
            }

        checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        ann_weight_keys = sorted(
            [key for key in state_dict if key.startswith("net.") and key.endswith(".weight")],
            key=lambda key: int(key.split(".")[1]),
        )
        bootstrap_weight_keys = sorted(
            [key for key in state_dict if key.endswith("synapse.weight")],
            key=lambda key: [int(part) if part.isdigit() else part for part in key.split(".")],
        )
        source_kind = "ann" if ann_weight_keys else "bootstrap"
        weight_keys = ann_weight_keys if ann_weight_keys else bootstrap_weight_keys
        bootstrap_layers = [block for block in self.blocks if hasattr(block, "synapse")]
        if not weight_keys:
            raise ValueError(f"No supported weight tensors found in checkpoint: {checkpoint_path}")
        if policy == "exact" and len(weight_keys) != len(bootstrap_layers):
            raise ValueError(
                f"Checkpoint has {len(weight_keys)} weighted layers, bootstrap network has {len(bootstrap_layers)} weighted layers."
            )

        source_hidden_keys = weight_keys[:-1]
        bootstrap_hidden_layers = bootstrap_layers[:-1]
        layer_pairs: list[tuple[str, Any, int, str]] = [
            (weight_key, bootstrap_layer, layer_index, "hidden")
            for layer_index, (weight_key, bootstrap_layer) in enumerate(zip(source_hidden_keys, bootstrap_hidden_layers))
        ]
        layer_pairs.append((weight_keys[-1], bootstrap_layers[-1], len(bootstrap_layers) - 1, "output"))

        copied_layers: list[dict[str, Any]] = []
        for weight_key, bootstrap_layer, layer_index, layer_role in layer_pairs:
            weight = state_dict[weight_key].detach().to(
                bootstrap_layer.synapse.weight.device,
                dtype=bootstrap_layer.synapse.weight.dtype,
            )
            if source_kind == "ann" and layer_index == 0 and self.input_strategy == "signed_split":
                weight = torch.cat([weight, -weight], dim=1)
            if weight.ndim == 2:
                weight = weight.reshape(weight.shape[0], weight.shape[1], 1, 1, 1)
            target = bootstrap_layer.synapse.weight.data
            if policy == "exact" and weight.shape != target.shape:
                raise ValueError(f"{weight_key} has mapped shape {tuple(weight.shape)}, expected {tuple(target.shape)}.")
            copy_shape = tuple(min(int(source_dim), int(target_dim)) for source_dim, target_dim in zip(weight.shape, target.shape))
            tensor_slice = tuple(slice(0, dim) for dim in copy_shape)
            target[tensor_slice].copy_(weight[tensor_slice])
            copied_layers.append(
                {
                    "source_key": weight_key,
                    "role": layer_role,
                    "source_shape": list(weight.shape),
                    "target_shape": list(target.shape),
                    "copied_shape": list(copy_shape),
                    "exact": tuple(weight.shape) == tuple(target.shape),
                }
            )

        skipped_layers = []
        if len(bootstrap_hidden_layers) > len(source_hidden_keys):
            skipped_layers.extend(
                {
                    "role": "hidden",
                    "target_index": index,
                    "target_shape": list(layer.synapse.weight.shape),
                    "reason": "no_matching_source_hidden_layer",
                }
                for index, layer in enumerate(bootstrap_hidden_layers[len(source_hidden_keys) :], start=len(source_hidden_keys))
            )
        if len(source_hidden_keys) > len(bootstrap_hidden_layers):
            skipped_layers.extend(
                {
                    "role": "hidden",
                    "source_key": key,
                    "reason": "no_matching_bootstrap_hidden_layer",
                }
                for key in source_hidden_keys[len(bootstrap_hidden_layers) :]
            )

        return {
            "policy": policy,
            "source_kind": source_kind,
            "checkpoint_path": str(checkpoint_path),
            "source_weight_layers": len(weight_keys),
            "bootstrap_weight_layers": len(bootstrap_layers),
            "copied_layers": copied_layers,
            "skipped_layers": skipped_layers,
        }

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


def rate_from_mode(model: Network, states: torch.Tensor, mode: bootstrap.routine.LayerMode) -> torch.Tensor:
    output = model.forward(states, mode)
    return torch.mean(output, dim=-1).reshape((states.shape[0], -1))


def pure_snn_rate(model: Network, states: torch.Tensor) -> torch.Tensor:
    snn_mode = bootstrap.routine.LayerMode(0, bootstrap.Mode.SNN)
    return rate_from_mode(model, states, snn_mode)


def samplers_ready(model: Network) -> bool:
    ready = True
    for block in model.blocks:
        sampler = getattr(block, "f", None)
        if sampler is None:
            continue
        if getattr(sampler, "centers", None) is None:
            ready = False
            break
    return ready


def fit_available_samplers(model: Network) -> None:
    for block in model.blocks:
        sampler = getattr(block, "f", None)
        if sampler is None:
            continue
        if len(getattr(sampler, "z", [])) == 0:
            continue
        block.fit()


def warmup_samplers(model: Network, states: torch.Tensor) -> None:
    if samplers_ready(model):
        return
    with torch.no_grad():
        _ = rate_from_mode(model, states, bootstrap.routine.LayerMode(0, bootstrap.Mode.SAMPLE))
    fit_available_samplers(model)


def percentage_error(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    numerator = torch.linalg.norm(predictions - targets, dim=1)
    denominator = torch.linalg.norm(targets, dim=1).clamp_min(1e-8)
    return 100.0 * numerator / denominator


def summarize_percentage_error(errors: list[torch.Tensor]) -> tuple[float, float]:
    if not errors:
        return 0.0, 0.0
    all_errors = torch.cat(errors)
    return float(all_errors.mean().item()), float(all_errors.median().item())


def evaluate_snn_metrics(model: Network, loader: DataLoader, device: torch.device, pin_memory: bool) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    output_abs_sum = 0.0
    output_numel = 0
    samples = 0
    percentage_errors: list[torch.Tensor] = []
    with torch.no_grad():
        for states, actions in loader:
            states = states.to(device, non_blocking=pin_memory)
            actions = actions.to(device, non_blocking=pin_memory)
            rate = pure_snn_rate(model, states)
            if not torch.isfinite(rate).all():
                raise FloatingPointError("Non-finite validation output")
            loss = F.mse_loss(rate, actions)
            loss_sum += loss.item() * states.shape[0]
            output_abs_sum += rate.abs().sum().item()
            output_numel += rate.numel()
            samples += states.shape[0]
            percentage_errors.append(percentage_error(rate, actions).detach().cpu())
    mean_pct, median_pct = summarize_percentage_error(percentage_errors)
    return {
        "mse": loss_sum / max(1, samples),
        "mean_percentage_error": mean_pct,
        "median_percentage_error": median_pct,
        "output_abs_mean": output_abs_sum / max(1, output_numel),
    }


def evaluate_snn_mse(model: Network, loader: DataLoader, device: torch.device, pin_memory: bool) -> float:
    return evaluate_snn_metrics(model, loader, device, pin_memory)["mse"]


def save_training_plot(history: dict[str, list[Any]], plot_path: Path) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (loss_ax, pct_ax) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    loss_ax.plot(history["train_snn_loss"], label="train SNN MSE")
    loss_ax.plot(history["val_snn_loss"], label="validation subset SNN MSE")
    full_val_epochs = [index for index, value in enumerate(history["full_val_snn_loss"]) if value is not None]
    full_val_losses = [value for value in history["full_val_snn_loss"] if value is not None]
    loss_ax.plot(full_val_epochs, full_val_losses, marker="o", linestyle="none", label="full validation SNN MSE")
    loss_ax.set_ylabel("SNN MSE loss")
    loss_ax.grid(True, alpha=0.3)

    lr_ax = loss_ax.twinx()
    lr_ax.plot(history["learning_rate"], color="tab:gray", linestyle="--", alpha=0.7, label="learning rate")
    lr_ax.set_ylabel("Learning rate")
    lr_ax.set_yscale("log")

    lines, labels = loss_ax.get_legend_handles_labels()
    lr_lines, lr_labels = lr_ax.get_legend_handles_labels()
    loss_ax.legend(lines + lr_lines, labels + lr_labels, loc="best")

    pct_ax.plot(history["train_mean_percentage_error"], label="train mean % error")
    pct_ax.plot(history["train_median_percentage_error"], label="train median % error")
    pct_ax.plot(history["val_mean_percentage_error"], label="validation subset mean % error")
    pct_ax.plot(history["val_median_percentage_error"], label="validation subset median % error")
    full_val_mean_pct = [value for value in history["full_val_mean_percentage_error"] if value is not None]
    full_val_median_pct = [value for value in history["full_val_median_percentage_error"] if value is not None]
    pct_ax.plot(full_val_epochs, full_val_mean_pct, marker="o", linestyle="none", label="full validation mean % error")
    pct_ax.plot(full_val_epochs, full_val_median_pct, marker="x", linestyle="none", label="full validation median % error")
    pct_ax.set_xlabel("Epoch")
    pct_ax.set_ylabel("Percentage error")
    pct_ax.grid(True, alpha=0.3)
    pct_ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_training_history(
    history: dict[str, list[Any]],
    history_path: Path,
    config: Config,
    best_snn_val_loss: float,
    best_epoch: int | None,
    best_full_val_mean_pct: float | None,
    best_full_val_median_pct: float | None,
    init_report: dict[str, Any],
    checkpoint_path: Path,
    export_path: Path,
    plot_path: Path,
) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": dataclass_to_dict(config),
        "init_report": init_report,
        "best_snn_val_loss": best_snn_val_loss,
        "best_full_val_snn_mse": best_snn_val_loss,
        "best_epoch": best_epoch,
        "best_full_val_mean_percentage_error": best_full_val_mean_pct,
        "best_full_val_median_percentage_error": best_full_val_median_pct,
        "checkpoint_path": str(checkpoint_path),
        "export_path": str(export_path),
        "plot_path": str(plot_path),
        "history": history,
    }
    history_path.write_text(json.dumps(payload, indent=2))


def train(config: Config) -> dict[str, Any]:
    validate_config(config)
    device = resolve_device(config.runtime)
    output_dir = resolve_path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / config.paths.checkpoint_name
    export_path = output_dir / config.paths.export_name
    plot_path = output_dir / config.paths.plot_name
    history_path = output_dir / config.paths.history_name
    pin_memory = config.training.pin_memory if config.training.pin_memory is not None else device.type == "cuda"

    dataset, training_set, testing_set, train_loader, val_loader, full_val_loader = make_loaders(config)

    model = Network(config.model).to(device)
    init_report = model.initialize_from_ann_checkpoint(
        resolve_path(config.paths.ann_checkpoint),
        device=device,
        policy=config.model.init_policy,
    )
    initial_quantization_diagnostics = model.quantization_diagnostics()
    print("[quantization:init] " + json.dumps(initial_quantization_diagnostics))

    training_mode = config.bootstrap_training.mode
    if training_mode not in SUPPORTED_BOOTSTRAP_TRAINING_MODES:
        raise ValueError(
            f"Unsupported bootstrap_training.mode {training_mode!r}. "
            f"Supported modes: {sorted(SUPPORTED_BOOTSTRAP_TRAINING_MODES)}"
        )
    bootstrap_scheduler = None
    if training_mode == "scheduler":
        bootstrap_scheduler = bootstrap.routine.Scheduler(
            num_sample_iter=config.bootstrap_training.num_sample_iter,
            sample_period=config.bootstrap_training.sample_period,
            crossover_epochs=list(config.bootstrap_training.crossover_epochs) if config.bootstrap_training.crossover_epochs else None,
        )

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
    print(f"init policy: {config.model.init_policy} copied_layers={len(init_report.get('copied_layers', []))} skipped_layers={len(init_report.get('skipped_layers', []))}")
    print(f"bootstrap training mode: {training_mode}")
    if bootstrap_scheduler is not None:
        print(
            f"bootstrap scheduler: num_sample_iter={config.bootstrap_training.num_sample_iter} "
            f"sample_period={config.bootstrap_training.sample_period} "
            f"crossover_epochs={config.bootstrap_training.crossover_epochs}"
        )
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
    best_epoch = None
    best_full_val_mean_pct = None
    best_full_val_median_pct = None
    history: dict[str, list[Any]] = {
        "train_snn_loss": [],
        "val_snn_loss": [],
        "full_val_snn_loss": [],
        "train_mean_percentage_error": [],
        "train_median_percentage_error": [],
        "val_mean_percentage_error": [],
        "val_median_percentage_error": [],
        "full_val_mean_percentage_error": [],
        "full_val_median_percentage_error": [],
        "learning_rate": [],
        "output_activity": [],
        "val_output_activity": [],
        "full_val_output_activity": [],
        "gradient_norm": [],
        "quantization_diagnostics": [{"stage": "initial", "layers": initial_quantization_diagnostics}],
    }

    for epoch in range(config.training.epochs):
        model.train()
        train_snn_loss_sum = 0.0
        train_output_abs_sum = 0.0
        train_output_numel = 0
        train_samples = 0
        train_percentage_errors: list[torch.Tensor] = []
        train_gradient_norm_sum = 0.0
        train_gradient_batches = 0


        for batch_index, (states, actions) in enumerate(train_loader):
            states = states.to(device, non_blocking=pin_memory)
            actions = actions.to(device, non_blocking=pin_memory)

            if training_mode == "scheduler":
                assert bootstrap_scheduler is not None
                layer_mode = bootstrap_scheduler.mode(epoch, batch_index, train=True)
                base_mode = getattr(layer_mode, "base_mode", layer_mode)
                if base_mode in {bootstrap.Mode.ANN, bootstrap.Mode.FIT}:
                    warmup_samplers(model, states)
                rate = rate_from_mode(model, states, layer_mode)
            else:
                rate = pure_snn_rate(model, states)
            if not torch.isfinite(rate).all():
                raise FloatingPointError(
                    f"Non-finite model output at epoch {epoch + 1}, "
                    f"batch {batch_index}"
                )
            loss = F.mse_loss(rate, actions)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch + 1}, "
                    f"batch {batch_index}"
                )

            optimizer.zero_grad()
            loss.backward()
            gradient_sq_sum = 0.0
            for parameter in model.parameters():
                if parameter.grad is None:
                    continue
                if not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("Non-finite model gradient")
                gradient_sq_sum += float(
                    parameter.grad.square().sum().item()
                )
            train_gradient_norm_sum += gradient_sq_sum ** 0.5
            train_gradient_batches += 1
            optimizer.step()

            train_snn_loss_sum += loss.detach().item() * states.shape[0]
            train_output_abs_sum += rate.detach().abs().sum().item()
            train_output_numel += rate.numel()
            train_samples += states.shape[0]
            train_percentage_errors.append(percentage_error(rate.detach(), actions).detach().cpu())

        if training_mode == "scheduler":
            fit_available_samplers(model)

        train_snn_loss = train_snn_loss_sum / max(1, train_samples)
        train_mean_pct, train_median_pct = summarize_percentage_error(train_percentage_errors)
        output_activity = train_output_abs_sum / max(1, train_output_numel)
        history["train_snn_loss"].append(train_snn_loss)
        history["train_mean_percentage_error"].append(train_mean_pct)
        history["train_median_percentage_error"].append(train_median_pct)
        history["output_activity"].append(output_activity)
        history["gradient_norm"].append(
            train_gradient_norm_sum / max(1, train_gradient_batches)
        )

        val_metrics = evaluate_snn_metrics(model, val_loader, device, pin_memory)
        val_snn_loss = val_metrics["mse"]
        history["val_snn_loss"].append(val_snn_loss)
        history["val_mean_percentage_error"].append(val_metrics["mean_percentage_error"])
        history["val_median_percentage_error"].append(val_metrics["median_percentage_error"])
        history["val_output_activity"].append(val_metrics["output_abs_mean"])

        run_full_val = epoch == 0 or (epoch + 1) % config.training.full_val_interval == 0 or epoch == config.training.epochs - 1
        full_val_snn_loss = None
        full_val_mean_pct = None
        full_val_median_pct = None
        full_val_output_activity = None
        checkpoint_note = ""
        if run_full_val:
            full_val_metrics = evaluate_snn_metrics(model, full_val_loader, device, pin_memory)
            full_val_snn_loss = full_val_metrics["mse"]
            full_val_mean_pct = full_val_metrics["mean_percentage_error"]
            full_val_median_pct = full_val_metrics["median_percentage_error"]
            full_val_output_activity = full_val_metrics["output_abs_mean"]
            epoch_quantization_diagnostics = model.quantization_diagnostics()
            history["quantization_diagnostics"].append({
                "stage": "full_validation",
                "epoch": epoch + 1,
                "layers": epoch_quantization_diagnostics,
            })
            print(
                f"[quantization:epoch={epoch + 1}] "
                + json.dumps(epoch_quantization_diagnostics)
            )
            if any(
                not layer["raw_finite"]
                or not layer["quantized_finite"]
                or layer["saturation_count"] != 0
                for layer in epoch_quantization_diagnostics
            ):
                raise FloatingPointError(
                    "Quantization diagnostics reported non-finite values "
                    "or saturation"
                )
            if full_val_snn_loss < best_snn_val_loss:
                best_snn_val_loss = full_val_snn_loss
                best_epoch = epoch + 1
                best_full_val_mean_pct = full_val_mean_pct
                best_full_val_median_pct = full_val_median_pct
                checkpoint_payload = model.checkpoint_payload()
                checkpoint_payload.update({
                    "epoch": best_epoch,
                    "best_full_val_snn_mse": best_snn_val_loss,
                    "best_full_val_mean_percentage_error": best_full_val_mean_pct,
                    "best_full_val_median_percentage_error": best_full_val_median_pct,
                    "resolved_config": dataclass_to_dict(config),
                })
                torch.save(checkpoint_payload, checkpoint_path)
                checkpoint_note = " saved_best"

        scheduler_metric = full_val_snn_loss if full_val_snn_loss is not None else val_snn_loss
        if lr_scheduler is not None:
            lr_scheduler.step(scheduler_metric)
        current_lr = optimizer.param_groups[0]["lr"]
        history["learning_rate"].append(current_lr)
        history["full_val_snn_loss"].append(full_val_snn_loss)
        history["full_val_mean_percentage_error"].append(full_val_mean_pct)
        history["full_val_median_percentage_error"].append(full_val_median_pct)
        history["full_val_output_activity"].append(full_val_output_activity)
        full_val_text = "n/a" if full_val_snn_loss is None else f"{full_val_snn_loss:.9f}"
        full_val_mean_pct_text = "n/a" if full_val_mean_pct is None else f"{full_val_mean_pct:.4f}%"
        full_val_median_pct_text = "n/a" if full_val_median_pct is None else f"{full_val_median_pct:.4f}%"

        print(
            f"[Epoch {epoch + 1:03d}/{config.training.epochs}] "
            f"train_snn_mse={train_snn_loss:.9f} val_subset_snn_mse={val_snn_loss:.9f} "
            f"full_val_snn_mse={full_val_text} best_snn_val_mse={best_snn_val_loss:.9f} "
            f"train_mean_pct={train_mean_pct:.4f}% train_median_pct={train_median_pct:.4f}% "
            f"val_mean_pct={val_metrics['mean_percentage_error']:.4f}% "
            f"val_median_pct={val_metrics['median_percentage_error']:.4f}% "
            f"full_val_mean_pct={full_val_mean_pct_text} full_val_median_pct={full_val_median_pct_text} "
            f"lr={current_lr:.2e} output_abs_mean={output_activity:.6f}{checkpoint_note}"
        )

    if checkpoint_path.exists():
        best_checkpoint = torch.load(checkpoint_path, map_location=device)
        best_state_dict = (
            best_checkpoint["state_dict"]
            if isinstance(best_checkpoint, dict)
            and "state_dict" in best_checkpoint
            else best_checkpoint
        )
        model.load_state_dict(best_state_dict)
    final_quantization_diagnostics = model.quantization_diagnostics()
    history["quantization_diagnostics"].append({
        "stage": "final_best_checkpoint",
        "epoch": best_epoch,
        "layers": final_quantization_diagnostics,
    })
    print(
        "[quantization:final] "
        + json.dumps(final_quantization_diagnostics)
    )


    save_training_history(
        history,
        history_path,
        config,
        best_snn_val_loss,
        best_epoch,
        best_full_val_mean_pct,
        best_full_val_median_pct,
        init_report,
        checkpoint_path,
        export_path,
        plot_path,
    )
    print(f"Saved training history to {history_path}")

    if config.runtime.save_plot:
        save_training_plot(history, plot_path)
        print(f"Saved training plot to {plot_path}")

    if config.runtime.export_hdf5:
        model.export_hdf5(export_path)
        print(f"Exported network to {export_path}")

    return {
        "best_snn_val_loss": best_snn_val_loss,
        "best_full_val_snn_mse": best_snn_val_loss,
        "best_epoch": best_epoch,
        "best_full_val_mean_percentage_error": best_full_val_mean_pct,
        "best_full_val_median_percentage_error": best_full_val_median_pct,
        "init_report": init_report,
        "checkpoint_path": checkpoint_path,
        "export_path": export_path,
        "plot_path": plot_path,
        "history_path": history_path,
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
