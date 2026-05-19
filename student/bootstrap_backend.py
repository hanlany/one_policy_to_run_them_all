from dataclasses import dataclass, field
import warnings
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

try:
    from lava.lib.dl.bootstrap import block as lava_bootstrap_block
    from lava.lib.dl.bootstrap import routine as lava_bootstrap_routine
except ImportError:
    lava_bootstrap_block = None
    lava_bootstrap_routine = None

TensorLike = Union[np.ndarray, torch.Tensor]
DEFAULT_HIDDEN_DIMS = [1024, 1024, 1024, 1024, 1024]
SUPPORTED_SNN_READOUTS = ("mean", "last", "sum")
DEFAULT_BOOTSTRAP_NEURON_THRESHOLD = 1.0
DEFAULT_BOOTSTRAP_CURRENT_DECAY = 0.25
DEFAULT_BOOTSTRAP_VOLTAGE_DECAY = 0.03
DEFAULT_BOOTSTRAP_NUM_SAMPLE_ITER = 10
DEFAULT_BOOTSTRAP_SAMPLE_PERIOD = 10
_BOOTSTRAP_DEVICE_WARNING_EMITTED = False


def resolve_bootstrap_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    global _BOOTSTRAP_DEVICE_WARNING_EMITTED
    requested = torch.device(device or "cpu")
    if requested.type != "cuda":
        return requested

    python_header = Path("/usr/include/python3.10/Python.h")
    ninja_path = Path("/usr/local/bin/ninja")
    cuda_ready = torch.cuda.is_available() and python_header.exists() and ninja_path.exists()
    if cuda_ready:
        return requested

    if not _BOOTSTRAP_DEVICE_WARNING_EMITTED:
        warnings.warn(
            "Bootstrap requested CUDA, but this environment cannot build Lava's CUDA extension path. Falling back to CPU. "
            "Install Python development headers and ninja in the image to enable CUDA bootstrap.",
            RuntimeWarning,
            stacklevel=2,
        )
        _BOOTSTRAP_DEVICE_WARNING_EMITTED = True
    return torch.device("cpu")


@dataclass
class TrainingArtifacts:
    best_val_loss: float
    best_checkpoint_path: str
    latest_checkpoint_path: str
    last_train_loss: float
    last_val_loss: float
    iteration_checkpoint_path: Optional[str] = None


@dataclass
class BootstrapTrainingConfig:
    dataset_path: str = "teacher_student_dagger_dataset.npz"
    input_strategy: str = "signed_split"
    input_weight: float = 1.0
    input_bias: float = 0.0
    batch_size: int = 64
    learning_rate: float = 1e-4
    epochs: int = 60
    hidden_dims: Sequence[int] = field(default_factory=lambda: list(DEFAULT_HIDDEN_DIMS))
    val_split: float = 0.2
    num_workers: int = 0
    checkpoint_dir: str = str(Path(__file__).resolve().parent)
    best_checkpoint_name: str = "student_model_best.pth"
    latest_checkpoint_name: str = "student_model_latest.pth"
    iteration_checkpoint_template: str = "student_model_dagger_iter_{iteration}.pth"
    timesteps: int = 3
    readout: str = "mean"
    num_sample_iter: int = DEFAULT_BOOTSTRAP_NUM_SAMPLE_ITER
    sample_period: int = DEFAULT_BOOTSTRAP_SAMPLE_PERIOD
    crossover_epochs: Sequence[int] = field(default_factory=tuple)
    neuron_threshold: float = DEFAULT_BOOTSTRAP_NEURON_THRESHOLD
    current_decay: float = DEFAULT_BOOTSTRAP_CURRENT_DECAY
    voltage_decay: float = DEFAULT_BOOTSTRAP_VOLTAGE_DECAY
    weight_scale: float = 1.0
    weight_norm: bool = False
    initialize_from_ann: bool = False


@dataclass
class BootstrapEvaluationArtifacts:
    parity_metrics: dict[str, object]
    teacher_metrics: dict[str, float]
    beats_conversion_baseline: bool


class TeacherStudentDataset(Dataset):
    def __init__(self, states: TensorLike, actions: TensorLike):
        self.states = torch.as_tensor(states, dtype=torch.float32)
        self.actions = torch.as_tensor(actions, dtype=torch.float32)

    @classmethod
    def from_npz(cls, npz_file: Union[str, Path]):
        data = np.load(npz_file)
        return cls(states=data["states"], actions=data["actions"])

    @classmethod
    def from_arrays(cls, states: TensorLike, actions: TensorLike):
        return cls(states=states, actions=actions)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


def _batched_tensor_range(num_samples: int, batch_size: int):
    for start in range(0, num_samples, batch_size):
        yield start, min(start + batch_size, num_samples)


def _percentage_error(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    numerator = torch.linalg.norm(predictions - targets, dim=1)
    denominator = torch.linalg.norm(targets, dim=1).clamp_min(1e-8)
    return 100.0 * numerator / denominator


def load_teacher_dataset(
    dataset_path: Union[str, Path],
    max_samples: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.load(dataset_path)
    states = torch.as_tensor(data["states"], dtype=torch.float32)
    actions = torch.as_tensor(data["actions"], dtype=torch.float32)
    if max_samples is not None:
        states = states[:max_samples]
        actions = actions[:max_samples]
    return states, actions


class BootstrapCheckpointManager:
    def __init__(self, checkpoint_dir: Union[str, Path]):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def best_path(self, config: BootstrapTrainingConfig) -> Path:
        return self.checkpoint_dir / config.best_checkpoint_name

    def latest_path(self, config: BootstrapTrainingConfig) -> Path:
        return self.checkpoint_dir / config.latest_checkpoint_name

    def iteration_path(self, config: BootstrapTrainingConfig, iteration: int) -> Path:
        return self.checkpoint_dir / config.iteration_checkpoint_template.format(iteration=iteration)

    def save(self, model: "BootstrapStudentPolicy", path: Union[str, Path], extra_metadata: Optional[Dict[str, object]] = None):
        payload = {
            "state_dict": model.state_dict(),
            **model.get_architecture_config(),
        }
        if extra_metadata:
            payload.update(extra_metadata)
        torch.save(payload, path)


SUPPORTED_BOOTSTRAP_INPUT_STRATEGIES = ("identity", "signed_split")


class BootstrapStudentPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[Sequence[int]] = None,
        timesteps: int = 3,
        readout: str = "mean",
        neuron_threshold: float = DEFAULT_BOOTSTRAP_NEURON_THRESHOLD,
        current_decay: float = DEFAULT_BOOTSTRAP_CURRENT_DECAY,
        voltage_decay: float = DEFAULT_BOOTSTRAP_VOLTAGE_DECAY,
        weight_scale: float = 1.0,
        weight_norm: bool = False,
        input_strategy: str = "signed_split",
        input_weight: float = 1.0,
        input_bias: float = 0.0,
    ):
        super().__init__()
        if lava_bootstrap_block is None or lava_bootstrap_routine is None:
            raise ImportError("lava.lib.dl.bootstrap is required to build BootstrapStudentPolicy.")
        if readout not in SUPPORTED_SNN_READOUTS:
            raise ValueError(f"Unsupported bootstrap readout '{readout}'. Supported readouts: {SUPPORTED_SNN_READOUTS}")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dims = list(hidden_dims or DEFAULT_HIDDEN_DIMS)
        self.timesteps = int(timesteps)
        self.readout = readout
        self.neuron_threshold = float(neuron_threshold)
        self.current_decay = float(current_decay)
        self.voltage_decay = float(voltage_decay)
        self.weight_scale = float(weight_scale)
        self.weight_norm = bool(weight_norm)
        self.input_strategy = input_strategy
        self.input_weight = float(input_weight)
        self.input_bias = float(input_bias)
        if self.input_strategy not in SUPPORTED_BOOTSTRAP_INPUT_STRATEGIES:
            raise ValueError(f"Unsupported bootstrap input strategy '{self.input_strategy}'. Supported strategies: {SUPPORTED_BOOTSTRAP_INPUT_STRATEGIES}")
        self.encoded_input_dim = self.input_dim * 2 if self.input_strategy == "signed_split" else self.input_dim

        neuron_params = {
            "threshold": self.neuron_threshold,
            "current_decay": self.current_decay,
            "voltage_decay": self.voltage_decay,
            "shared_param": True,
            "persistent_state": False,
            "requires_grad": False,
        }

        blocks = [lava_bootstrap_block.cuba.Input(neuron_params=dict(neuron_params), weight=self.input_weight, bias=self.input_bias, delay_shift=False), lava_bootstrap_block.cuba.Flatten()]
        prev_dim = self.encoded_input_dim
        for hidden_dim in self.hidden_dims:
            blocks.append(
                lava_bootstrap_block.cuba.Dense(
                    neuron_params=dict(neuron_params),
                    in_neurons=prev_dim,
                    out_neurons=int(hidden_dim),
                    weight_scale=self.weight_scale,
                    weight_norm=self.weight_norm,
                    delay_shift=False,
                )
            )
            prev_dim = int(hidden_dim)
        blocks.append(
            lava_bootstrap_block.cuba.Affine(
                neuron_params=dict(neuron_params),
                in_neurons=prev_dim,
                out_neurons=self.output_dim,
                weight_scale=self.weight_scale,
                weight_norm=self.weight_norm,
                dynamics=False,
            )
        )
        self.blocks = nn.ModuleList(blocks)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_architecture_config(self) -> Dict[str, object]:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dims": list(self.hidden_dims),
            "timesteps": int(self.timesteps),
            "readout": self.readout,
            "neuron_threshold": float(self.neuron_threshold),
            "current_decay": float(self.current_decay),
            "voltage_decay": float(self.voltage_decay),
            "weight_scale": float(self.weight_scale),
            "weight_norm": bool(self.weight_norm),
            "input_strategy": self.input_strategy,
            "input_weight": float(self.input_weight),
            "input_bias": float(self.input_bias),
            "backend": "bootstrap",
        }

    def initialize_from_ann(self, student_model: nn.Module):
        linear_layers = [module for module in student_model.net if isinstance(module, nn.Linear)]
        bootstrap_layers = [
            block
            for block in self.blocks
            if isinstance(block, (lava_bootstrap_block.cuba.Dense, lava_bootstrap_block.cuba.Affine))
        ]
        if len(linear_layers) != len(bootstrap_layers):
            raise ValueError(
                f"Cannot initialize bootstrap model from ANN: {len(linear_layers)} ANN layers != {len(bootstrap_layers)} bootstrap layers."
            )
        for layer_index, (ann_layer, bootstrap_layer) in enumerate(zip(linear_layers, bootstrap_layers)):
            weight = ann_layer.weight.detach().to(bootstrap_layer.synapse.weight.device, dtype=bootstrap_layer.synapse.weight.dtype)
            if layer_index == 0 and self.input_strategy == "signed_split":
                weight = torch.cat([weight, -weight], dim=1)
            weight = weight.reshape(ann_layer.out_features, weight.shape[1], 1, 1, 1)
            bootstrap_layer.synapse.weight.data.copy_(weight)
        self.eval()
        return self

    def _transform_observation(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.input_strategy == "identity":
            return tensor
        positive = torch.relu(tensor)
        negative = torch.relu(-tensor)
        return torch.cat([positive, negative], dim=1)

    def encode_input(self, states: TensorLike) -> torch.Tensor:
        tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim == 2:
            tensor = self._transform_observation(tensor)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1).unsqueeze(-1)
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(-1)
        if tensor.ndim != 5:
            raise ValueError(f"Expected encoded bootstrap input to have 5 dimensions, received shape {tuple(tensor.shape)}")

        if tensor.shape[-1] == 1 and self.timesteps > 1:
            tensor = tensor.repeat(1, 1, 1, 1, self.timesteps)
        elif tensor.shape[-1] != self.timesteps:
            raise ValueError(
                f"Encoded bootstrap input timestep dimension {tensor.shape[-1]} does not match configured timesteps {self.timesteps}."
            )
        return tensor

    def _coerce_mode_iterator(self, mode) -> Iterable:
        if isinstance(mode, str):
            mode = lava_bootstrap_routine.Mode[mode.upper()]
        if isinstance(mode, lava_bootstrap_routine.Mode):
            def iterator():
                while True:
                    yield mode
            return iterator()
        return iter(mode)

    def raw_forward(self, states: TensorLike, mode=None) -> torch.Tensor:
        if mode is None:
            mode = lava_bootstrap_routine.Mode.SNN
        x = self.encode_input(states)
        layer_modes = self._coerce_mode_iterator(mode)
        for block in self.blocks:
            x = block(x, mode=next(layer_modes))
        while x.ndim > 3 and x.shape[-2] == 1:
            x = x.squeeze(-2)
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        return x

    def decode_output(self, raw_output: torch.Tensor, readout: Optional[str] = None) -> torch.Tensor:
        readout = readout or self.readout
        if readout not in SUPPORTED_SNN_READOUTS:
            raise ValueError(f"Unsupported bootstrap readout '{readout}'. Supported readouts: {SUPPORTED_SNN_READOUTS}")
        if raw_output.ndim != 3:
            raise ValueError(f"Expected raw bootstrap output to have shape (B, C, T), received {tuple(raw_output.shape)}")
        if readout == "mean":
            return raw_output.mean(dim=-1)
        if readout == "last":
            return raw_output[..., -1]
        return raw_output.sum(dim=-1)

    def forward(self, states: TensorLike, mode=None, readout: Optional[str] = None) -> torch.Tensor:
        if mode is None:
            mode = lava_bootstrap_routine.Mode.ANN
        raw_output = self.raw_forward(states, mode=mode)
        return self.decode_output(raw_output, readout=readout)

    def predict(self, states: TensorLike, mode=None, readout: Optional[str] = None) -> np.ndarray:
        if mode is None:
            mode = lava_bootstrap_routine.Mode.SNN
        with torch.no_grad():
            return self(states, mode=mode, readout=readout).detach().cpu().numpy()


class BootstrapStudentTrainer:
    def __init__(
        self,
        config: Optional[BootstrapTrainingConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
        checkpoint_manager: Optional[BootstrapCheckpointManager] = None,
    ):
        self.config = config or BootstrapTrainingConfig()
        self.device = resolve_bootstrap_device(device)
        self.checkpoint_manager = checkpoint_manager or BootstrapCheckpointManager(self.config.checkpoint_dir)
        self.criterion = nn.MSELoss()
        self.model: Optional[BootstrapStudentPolicy] = None

    def _samplers_ready(self) -> bool:
        if self.model is None:
            return False
        ready = True
        for block in self.model.blocks:
            sampler = getattr(block, "f", None)
            if sampler is None:
                continue
            if getattr(sampler, "centers", None) is None:
                ready = False
                break
        return ready

    def _fit_available_samplers(self):
        if self.model is None:
            return
        for block in self.model.blocks:
            sampler = getattr(block, "f", None)
            if sampler is None:
                continue
            if len(getattr(sampler, "z", [])) == 0:
                continue
            block.fit()

    def _warmup_samplers(self, states: torch.Tensor):
        if self.model is None or self._samplers_ready():
            return
        with torch.no_grad():
            _ = self.model(states, mode=lava_bootstrap_routine.Mode.SAMPLE, readout=self.config.readout)
        self._fit_available_samplers()

    def _resolve_dataset_path(self, dataset_path: Optional[Union[str, Path]] = None) -> Path:
        raw_path = Path(dataset_path or self.config.dataset_path)
        if raw_path.is_absolute():
            return raw_path
        return Path(__file__).resolve().parent / raw_path

    def _split_dataset(self, dataset: TeacherStudentDataset) -> Tuple[Dataset, Optional[Dataset]]:
        if len(dataset) < 2 or self.config.val_split <= 0.0:
            return dataset, None
        val_size = max(1, int(len(dataset) * self.config.val_split))
        train_size = len(dataset) - val_size
        if train_size <= 0:
            train_size = len(dataset) - 1
            val_size = 1
        return random_split(dataset, [train_size, val_size])

    def _create_loaders(self, dataset: TeacherStudentDataset) -> Tuple[DataLoader, Optional[DataLoader]]:
        train_dataset, val_dataset = self._split_dataset(dataset)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers)
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers)
        return train_loader, val_loader

    def _build_model(self, input_dim: int, output_dim: int) -> BootstrapStudentPolicy:
        return BootstrapStudentPolicy(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=self.config.hidden_dims,
            timesteps=self.config.timesteps,
            readout=self.config.readout,
            neuron_threshold=self.config.neuron_threshold,
            current_decay=self.config.current_decay,
            voltage_decay=self.config.voltage_decay,
            weight_scale=self.config.weight_scale,
            weight_norm=self.config.weight_norm,
            input_strategy=self.config.input_strategy,
            input_weight=self.config.input_weight,
            input_bias=self.config.input_bias,
        ).to(self.device)

    def train(self, dataset_path: Optional[Union[str, Path]] = None, initial_model: Optional[BootstrapStudentPolicy] = None) -> TrainingArtifacts:
        dataset = TeacherStudentDataset.from_npz(self._resolve_dataset_path(dataset_path))
        return self.train_dataset(dataset, initial_model=initial_model)

    def train_on_arrays(
        self,
        states: TensorLike,
        actions: TensorLike,
        iteration: Optional[int] = None,
        initial_model: Optional[BootstrapStudentPolicy] = None,
        initialize_from_ann: Optional[nn.Module] = None,
    ) -> TrainingArtifacts:
        dataset = TeacherStudentDataset.from_arrays(states=states, actions=actions)
        return self.train_dataset(dataset, iteration=iteration, initial_model=initial_model, initialize_from_ann=initialize_from_ann)

    def train_dataset(
        self,
        dataset: TeacherStudentDataset,
        iteration: Optional[int] = None,
        initial_model: Optional[BootstrapStudentPolicy] = None,
        initialize_from_ann: Optional[nn.Module] = None,
    ) -> TrainingArtifacts:
        if lava_bootstrap_routine is None:
            raise ImportError("lava.lib.dl.bootstrap is required to train BootstrapStudentPolicy.")

        train_loader, val_loader = self._create_loaders(dataset)
        input_dim = int(dataset.states.shape[1])
        output_dim = int(dataset.actions.shape[1])

        self.model = initial_model.to(self.device) if initial_model is not None else self._build_model(input_dim, output_dim)
        if initialize_from_ann is not None:
            self.model.initialize_from_ann(initialize_from_ann)
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = lava_bootstrap_routine.Scheduler(
            num_sample_iter=self.config.num_sample_iter,
            sample_period=self.config.sample_period,
            crossover_epochs=list(self.config.crossover_epochs) if self.config.crossover_epochs else None,
        )

        best_val_loss = float("inf")
        last_train_loss = float("inf")
        last_val_loss = float("inf")
        best_checkpoint_path = self.checkpoint_manager.best_path(self.config)
        latest_checkpoint_path = self.checkpoint_manager.latest_path(self.config)

        for epoch in range(self.config.epochs):
            self.model.train()
            train_loss = 0.0
            for batch_index, (states, actions) in enumerate(train_loader):
                states = states.to(self.device)
                actions = actions.to(self.device)
                optimizer.zero_grad()
                layer_mode = scheduler.mode(epoch, batch_index, train=True)
                base_mode = getattr(layer_mode, "base_mode", layer_mode)
                if base_mode in {lava_bootstrap_routine.Mode.ANN, lava_bootstrap_routine.Mode.FIT}:
                    self._warmup_samplers(states)
                predicted_actions = self.model(states, mode=layer_mode, readout=self.config.readout)
                loss = self.criterion(predicted_actions, actions)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self._fit_available_samplers()
            last_train_loss = train_loss / max(1, len(train_loader))

            self.model.eval()
            val_loss = 0.0
            if val_loader is None:
                val_loss = last_train_loss
            else:
                with torch.no_grad():
                    for states, actions in val_loader:
                        states = states.to(self.device)
                        actions = actions.to(self.device)
                        predicted_actions = self.model(states, mode=lava_bootstrap_routine.Mode.SNN, readout=self.config.readout)
                        loss = self.criterion(predicted_actions, actions)
                        val_loss += loss.item()
                val_loss = val_loss / max(1, len(val_loader))
            last_val_loss = val_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.checkpoint_manager.save(
                    self.model,
                    best_checkpoint_path,
                    extra_metadata={"best_val_loss": best_val_loss, "epoch": epoch + 1},
                )

        self.checkpoint_manager.save(
            self.model,
            latest_checkpoint_path,
            extra_metadata={
                "best_val_loss": best_val_loss,
                "last_train_loss": last_train_loss,
                "last_val_loss": last_val_loss,
            },
        )

        iteration_checkpoint_path = None
        if iteration is not None:
            iteration_checkpoint_path = self.checkpoint_manager.iteration_path(self.config, iteration)
            self.checkpoint_manager.save(
                self.model,
                iteration_checkpoint_path,
                extra_metadata={"best_val_loss": best_val_loss, "iteration": iteration},
            )

        self.model.eval()
        return TrainingArtifacts(
            best_val_loss=best_val_loss,
            best_checkpoint_path=str(best_checkpoint_path),
            latest_checkpoint_path=str(latest_checkpoint_path),
            last_train_loss=last_train_loss,
            last_val_loss=last_val_loss,
            iteration_checkpoint_path=str(iteration_checkpoint_path) if iteration_checkpoint_path is not None else None,
        )


def infer_bootstrap_architecture_from_state_dict(state_dict) -> tuple[int, int, list[int]]:
    weight_keys = sorted(
        [key for key in state_dict.keys() if key.endswith("synapse.weight")],
        key=lambda key: [int(part) if part.isdigit() else part for part in key.split(".")],
    )
    if not weight_keys:
        raise ValueError("Could not infer bootstrap architecture from checkpoint state dict.")
    input_dim = int(state_dict[weight_keys[0]].shape[1])
    output_dim = int(state_dict[weight_keys[-1]].shape[0])
    hidden_dims = [int(state_dict[key].shape[0]) for key in weight_keys[:-1]]
    return input_dim, output_dim, hidden_dims


def load_bootstrap_policy_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    input_dim: Optional[int] = None,
    output_dim: Optional[int] = None,
    hidden_dims: Optional[Sequence[int]] = None,
) -> tuple[BootstrapStudentPolicy, dict[str, object]]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

    input_dim = checkpoint.get("input_dim", input_dim) if isinstance(checkpoint, dict) else input_dim
    output_dim = checkpoint.get("output_dim", output_dim) if isinstance(checkpoint, dict) else output_dim
    hidden_dims = checkpoint.get("hidden_dims", hidden_dims) if isinstance(checkpoint, dict) else hidden_dims
    timesteps = checkpoint.get("timesteps", 3) if isinstance(checkpoint, dict) else 3
    readout = checkpoint.get("readout", "mean") if isinstance(checkpoint, dict) else "mean"
    neuron_threshold = checkpoint.get("neuron_threshold", DEFAULT_BOOTSTRAP_NEURON_THRESHOLD) if isinstance(checkpoint, dict) else DEFAULT_BOOTSTRAP_NEURON_THRESHOLD
    current_decay = checkpoint.get("current_decay", DEFAULT_BOOTSTRAP_CURRENT_DECAY) if isinstance(checkpoint, dict) else DEFAULT_BOOTSTRAP_CURRENT_DECAY
    voltage_decay = checkpoint.get("voltage_decay", DEFAULT_BOOTSTRAP_VOLTAGE_DECAY) if isinstance(checkpoint, dict) else DEFAULT_BOOTSTRAP_VOLTAGE_DECAY
    weight_scale = checkpoint.get("weight_scale", 1.0) if isinstance(checkpoint, dict) else 1.0
    weight_norm = checkpoint.get("weight_norm", False) if isinstance(checkpoint, dict) else False
    input_strategy = checkpoint.get("input_strategy", "signed_split") if isinstance(checkpoint, dict) else "signed_split"
    input_weight = checkpoint.get("input_weight", 1.0) if isinstance(checkpoint, dict) else 1.0
    input_bias = checkpoint.get("input_bias", 0.0) if isinstance(checkpoint, dict) else 0.0

    if input_dim is None or output_dim is None or hidden_dims is None:
        input_dim, output_dim, hidden_dims = infer_bootstrap_architecture_from_state_dict(state_dict)

    model = BootstrapStudentPolicy(
        input_dim=int(input_dim),
        output_dim=int(output_dim),
        hidden_dims=list(hidden_dims),
        timesteps=int(timesteps),
        readout=str(readout),
        neuron_threshold=float(neuron_threshold),
        current_decay=float(current_decay),
        voltage_decay=float(voltage_decay),
        weight_scale=float(weight_scale),
        weight_norm=bool(weight_norm),
        input_strategy=str(input_strategy),
        input_weight=float(input_weight),
        input_bias=float(input_bias),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, {
        "input_dim": int(input_dim),
        "output_dim": int(output_dim),
        "hidden_dims": list(hidden_dims),
        "timesteps": int(timesteps),
        "readout": str(readout),
        "neuron_threshold": float(neuron_threshold),
        "current_decay": float(current_decay),
        "voltage_decay": float(voltage_decay),
        "weight_scale": float(weight_scale),
        "weight_norm": bool(weight_norm),
        "input_strategy": str(input_strategy),
        "input_weight": float(input_weight),
        "input_bias": float(input_bias),
        "backend": "bootstrap",
    }


def evaluate_ann_bootstrap_parity(
    student_model: nn.Module,
    bootstrap_model: BootstrapStudentPolicy,
    states: TensorLike,
    batch_size: int = 128,
    device: Optional[Union[str, torch.device]] = None,
) -> dict[str, object]:
    states_tensor = torch.as_tensor(states, dtype=torch.float32)
    device = torch.device(device or next(student_model.parameters()).device)
    student_model.eval()
    diagnostics = {
        readout: {"sq_error": 0.0, "abs_error_sum": 0.0, "output_abs_sum": 0.0, "numel": 0}
        for readout in SUPPORTED_SNN_READOUTS
    }
    ann_sq_sum = 0.0
    ann_abs_sum = 0.0

    for start, end in _batched_tensor_range(len(states_tensor), batch_size):
        batch_states = states_tensor[start:end]
        with torch.no_grad():
            ann_out = student_model(batch_states.to(device)).detach().cpu()
            raw_bootstrap_out = bootstrap_model.raw_forward(batch_states.numpy()).detach().cpu()

        ann_sq_sum += torch.square(ann_out).sum().item()
        ann_abs_sum += ann_out.abs().sum().item()
        for readout in SUPPORTED_SNN_READOUTS:
            decoded = bootstrap_model.decode_output(raw_bootstrap_out, readout=readout)
            diff = decoded - ann_out
            diagnostics[readout]["sq_error"] += torch.square(diff).sum().item()
            diagnostics[readout]["abs_error_sum"] += diff.abs().sum().item()
            diagnostics[readout]["output_abs_sum"] += decoded.abs().sum().item()
            diagnostics[readout]["numel"] += diff.numel()

    readout_metrics = {}
    ann_l2_norm = ann_sq_sum ** 0.5
    for readout, values in diagnostics.items():
        readout_metrics[readout] = {
            "relative_l2_error": (values["sq_error"] ** 0.5) / (ann_l2_norm + 1e-8),
            "mae": values["abs_error_sum"] / max(1, values["numel"]),
            "output_scale_ratio": values["output_abs_sum"] / (ann_abs_sum + 1e-8),
        }

    selected_metrics = readout_metrics[bootstrap_model.readout]
    return {
        "selected_readout": bootstrap_model.readout,
        "samples": int(len(states_tensor)),
        "relative_l2_error": float(selected_metrics["relative_l2_error"]),
        "mae": float(selected_metrics["mae"]),
        "output_scale_ratio": float(selected_metrics["output_scale_ratio"]),
        "readout_diagnostics": {
            name: {metric: float(value) for metric, value in metrics.items()}
            for name, metrics in readout_metrics.items()
        },
    }


def evaluate_bootstrap_against_teacher(
    student_model: nn.Module,
    bootstrap_model: BootstrapStudentPolicy,
    states: TensorLike,
    teacher_actions: TensorLike,
    batch_size: int = 128,
    device: Optional[Union[str, torch.device]] = None,
) -> dict[str, float]:
    states_tensor = torch.as_tensor(states, dtype=torch.float32)
    teacher_tensor = torch.as_tensor(teacher_actions, dtype=torch.float32)
    device = torch.device(device or next(student_model.parameters()).device)
    student_model.eval()

    student_errors = []
    bootstrap_errors = []
    for start, end in _batched_tensor_range(len(states_tensor), batch_size):
        batch_states = states_tensor[start:end]
        batch_targets = teacher_tensor[start:end]
        with torch.no_grad():
            student_actions = student_model(batch_states.to(device)).detach().cpu()
        bootstrap_actions = torch.from_numpy(bootstrap_model.predict(batch_states.numpy(), readout=bootstrap_model.readout)).float()
        student_errors.append(_percentage_error(student_actions, batch_targets))
        bootstrap_errors.append(_percentage_error(bootstrap_actions, batch_targets))

    student_errors = torch.cat(student_errors)
    bootstrap_errors = torch.cat(bootstrap_errors)
    return {
        "student_mean_percentage_error": float(student_errors.mean().item()),
        "student_median_percentage_error": float(student_errors.median().item()),
        "snn_mean_percentage_error": float(bootstrap_errors.mean().item()),
        "snn_median_percentage_error": float(bootstrap_errors.median().item()),
    }


def collect_bootstrap_activity_stats(
    bootstrap_model: BootstrapStudentPolicy,
    states: TensorLike,
    device: Optional[Union[str, torch.device]] = None,
) -> dict[str, object]:
    device = torch.device(device or bootstrap_model.device)
    batch = torch.as_tensor(states, dtype=torch.float32, device=device)
    if batch.ndim == 1:
        batch = batch.unsqueeze(0)
    x = bootstrap_model.encode_input(batch)
    block_stats = []
    with torch.no_grad():
        for index, block in enumerate(bootstrap_model.blocks):
            x = block(x, mode=lava_bootstrap_routine.Mode.SNN)
            block_stats.append(
                {
                    "block_index": int(index),
                    "block_name": type(block).__name__,
                    "shape": list(x.shape),
                    "abs_mean": float(x.abs().mean().item()),
                    "nonzero_fraction": float((x != 0).float().mean().item()),
                    "min": float(x.min().item()),
                    "max": float(x.max().item()),
                }
            )
    final_nonzero_fraction = block_stats[-1]["nonzero_fraction"] if block_stats else 0.0
    return {
        "input_strategy": bootstrap_model.input_strategy,
        "encoded_input_dim": int(bootstrap_model.encoded_input_dim),
        "block_stats": block_stats,
        "network_silent": bool(final_nonzero_fraction == 0.0),
    }


class BootstrapPolicyPipeline:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[Sequence[int]] = None,
        training_config: Optional[BootstrapTrainingConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = resolve_bootstrap_device(device)
        self.training_config = training_config or BootstrapTrainingConfig(hidden_dims=list(hidden_dims or DEFAULT_HIDDEN_DIMS))
        if hidden_dims is not None:
            self.training_config.hidden_dims = list(hidden_dims)
        self.checkpoint_manager = BootstrapCheckpointManager(self.training_config.checkpoint_dir)
        self.bootstrap_trainer = BootstrapStudentTrainer(
            config=self.training_config,
            device=self.device,
            checkpoint_manager=self.checkpoint_manager,
        )
        self.bootstrap_model: Optional[BootstrapStudentPolicy] = None
        self.snn_policy: Optional[BootstrapStudentPolicy] = None

    def build_student_policy(self) -> BootstrapStudentPolicy:
        self.bootstrap_model = BootstrapStudentPolicy(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=self.training_config.hidden_dims,
            timesteps=self.training_config.timesteps,
            readout=self.training_config.readout,
            neuron_threshold=self.training_config.neuron_threshold,
            current_decay=self.training_config.current_decay,
            voltage_decay=self.training_config.voltage_decay,
            weight_scale=self.training_config.weight_scale,
            weight_norm=self.training_config.weight_norm,
            input_strategy=self.training_config.input_strategy,
            input_weight=self.training_config.input_weight,
            input_bias=self.training_config.input_bias,
        ).to(self.device)
        return self.bootstrap_model

    def load_student_policy(self, checkpoint_path: Union[str, Path]) -> BootstrapStudentPolicy:
        self.bootstrap_model, _ = load_bootstrap_policy_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=self.device,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=self.training_config.hidden_dims,
        )
        return self.bootstrap_model

    def train_offline(self, dataset_path: Optional[Union[str, Path]] = None, initialize_from_ann: Optional[nn.Module] = None) -> TrainingArtifacts:
        dataset = TeacherStudentDataset.from_npz(self.bootstrap_trainer._resolve_dataset_path(dataset_path))
        artifacts = self.bootstrap_trainer.train_dataset(dataset, initial_model=self.bootstrap_model, initialize_from_ann=initialize_from_ann)
        self.bootstrap_model = self.bootstrap_trainer.model
        return artifacts

    def train_on_aggregated_data(
        self,
        states: TensorLike,
        actions: TensorLike,
        iteration: Optional[int] = None,
        initialize_from_ann: Optional[nn.Module] = None,
    ) -> TrainingArtifacts:
        artifacts = self.bootstrap_trainer.train_on_arrays(
            states=states,
            actions=actions,
            iteration=iteration,
            initial_model=self.bootstrap_model,
            initialize_from_ann=initialize_from_ann,
        )
        self.bootstrap_model = self.bootstrap_trainer.model
        return artifacts

    def build_snn_policy(
        self,
        student_checkpoint_path: Optional[Union[str, Path]] = None,
        student_model: Optional[BootstrapStudentPolicy] = None,
    ) -> BootstrapStudentPolicy:
        if student_model is None:
            if student_checkpoint_path is not None:
                student_model = self.load_student_policy(student_checkpoint_path)
            elif self.bootstrap_model is not None:
                student_model = self.bootstrap_model
            else:
                raise ValueError("A bootstrap model or checkpoint path is required to build the bootstrap SNN policy.")
        self.bootstrap_model = student_model
        self.snn_policy = student_model
        return self.snn_policy

    def predict_student(self, states: TensorLike) -> np.ndarray:
        if self.bootstrap_model is None:
            raise RuntimeError("Bootstrap student policy has not been initialized.")
        return self.bootstrap_model.predict(states, mode=lava_bootstrap_routine.Mode.ANN, readout=self.training_config.readout)

    def predict_snn(self, states: TensorLike, readout: Optional[str] = None) -> np.ndarray:
        if self.snn_policy is None:
            raise RuntimeError("Bootstrap SNN policy has not been initialized.")
        return self.snn_policy.predict(states, mode=lava_bootstrap_routine.Mode.SNN, readout=readout or self.training_config.readout)

    def evaluate_parity(self, student_model: nn.Module, states: TensorLike, batch_size: int = 128) -> dict[str, object]:
        if self.snn_policy is None:
            raise RuntimeError("Bootstrap SNN policy must be initialized before parity evaluation.")
        return evaluate_ann_bootstrap_parity(student_model=student_model, bootstrap_model=self.snn_policy, states=states, batch_size=batch_size, device=self.device)

    def evaluate_against_teacher(
        self,
        student_model: nn.Module,
        states: TensorLike,
        teacher_actions: TensorLike,
        batch_size: int = 128,
    ) -> dict[str, float]:
        if self.snn_policy is None:
            raise RuntimeError("Bootstrap SNN policy must be initialized before teacher evaluation.")
        return evaluate_bootstrap_against_teacher(
            student_model=student_model,
            bootstrap_model=self.snn_policy,
            states=states,
            teacher_actions=teacher_actions,
            batch_size=batch_size,
            device=self.device,
        )
