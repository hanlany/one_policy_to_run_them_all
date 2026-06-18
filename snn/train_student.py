from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    from lava.lib.dl import conversion as lava_conversion
except ImportError:
    lava_conversion = None

try:
    from one_policy_to_run_them_all.student.bootstrap_backend import (
        BootstrapEvaluationArtifacts,
        BootstrapPolicyPipeline,
        BootstrapTrainingConfig,
        BootstrapStudentPolicy,
    )
except ImportError:
    try:
        from bootstrap_backend import (
            BootstrapEvaluationArtifacts,
            BootstrapPolicyPipeline,
            BootstrapTrainingConfig,
            BootstrapStudentPolicy,
        )
    except ImportError:
        BootstrapEvaluationArtifacts = None
        BootstrapPolicyPipeline = None
        BootstrapTrainingConfig = None
        BootstrapStudentPolicy = None

TensorLike = Union[np.ndarray, torch.Tensor]
DEFAULT_HIDDEN_DIMS = [1024, 1024, 1024, 1024, 1024]
SUPPORTED_SNN_READOUTS = ("mean", "last", "sum")
SUPPORTED_SNN_OUTPUT_ACTIVATIONS = ("sigma", "sdrelu", "delta")
SUPPORTED_STUDENT_BACKENDS = ("ann", "bootstrap")
DEFAULT_SNN_THRESHOLD = 0.2
DEFAULT_SNN_TIMESTEPS = 3
CONVERSION_TARGET_HIDDEN_DIMS = [
    [1024, 512],
    [512, 512],
    [512, 256],
    [1024, 1024, 512],
]
FOCUSED_SWEEP_THRESHOLDS = (0.2, 0.25, 0.3, 0.4, 0.5)
FOCUSED_SWEEP_TIMESTEPS = (1, 2, 3, 4, 5, 6)
ACTIVATION_SCALE_RATIO_THRESHOLD = 20.0
BASELINE_PARITY_REFERENCE = {
    "relative_l2_error": 3.71,
    "output_scale_ratio": 2.45,
    "student_mean_percentage_error": 20.6434,
}
STUDENT_ERROR_REGRESSION_TOLERANCE = 0.10
SNN_TEACHER_ERROR_RATIO_LIMIT = 2.0
SNN_MEAN_PERCENTAGE_ERROR_LIMIT = 30.0


@dataclass
class StudentTrainingConfig:
    dataset_path: str = "teacher_student_dagger_dataset.npz"
    backend: str = "ann"
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50
    hidden_dims: Sequence[int] = field(default_factory=lambda: list(DEFAULT_HIDDEN_DIMS))
    val_split: float = 0.2
    num_workers: int = 0
    checkpoint_dir: str = str(Path(__file__).resolve().parent)
    best_checkpoint_name: str = "student_model_best.pth"
    latest_checkpoint_name: str = "student_model_latest.pth"
    iteration_checkpoint_template: str = "student_model_dagger_iter_{iteration}.pth"


@dataclass
class TrainingArtifacts:
    best_val_loss: float
    best_checkpoint_path: str
    latest_checkpoint_path: str
    last_train_loss: float
    last_val_loss: float
    iteration_checkpoint_path: Optional[str] = None


@dataclass
class SNNConversionConfig:
    threshold: float = DEFAULT_SNN_THRESHOLD
    timesteps: int = DEFAULT_SNN_TIMESTEPS
    device: Optional[str] = None
    export_dir: Optional[str] = None
    readout: str = "mean"
    calibration_samples: Optional[int] = None
    output_activation: str = "sigma"


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


class StudentPolicy(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Optional[Sequence[int]] = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = list(hidden_dims or [256, 256])

        layers = []
        prev_dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_architecture_config(self) -> Dict[str, object]:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dims": list(self.hidden_dims),
        }


class StudentCheckpointManager:
    def __init__(self, checkpoint_dir: Union[str, Path]):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def best_path(self, config: StudentTrainingConfig) -> Path:
        return self.checkpoint_dir / config.best_checkpoint_name

    def latest_path(self, config: StudentTrainingConfig) -> Path:
        return self.checkpoint_dir / config.latest_checkpoint_name

    def iteration_path(self, config: StudentTrainingConfig, iteration: int) -> Path:
        return self.checkpoint_dir / config.iteration_checkpoint_template.format(iteration=iteration)

    def save(self, model: StudentPolicy, path: Union[str, Path], extra_metadata: Optional[Dict[str, object]] = None):
        payload = {
            "state_dict": model.state_dict(),
            **model.get_architecture_config(),
        }
        if extra_metadata:
            payload.update(extra_metadata)
        torch.save(payload, path)

    def load(
        self,
        checkpoint_path: Union[str, Path],
        device: Union[str, torch.device] = "cpu",
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        hidden_dims: Optional[Sequence[int]] = None,
    ) -> StudentPolicy:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint
        architecture = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dims": list(hidden_dims) if hidden_dims is not None else None,
        }

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            architecture["input_dim"] = checkpoint.get("input_dim", architecture["input_dim"])
            architecture["output_dim"] = checkpoint.get("output_dim", architecture["output_dim"])
            architecture["hidden_dims"] = checkpoint.get("hidden_dims", architecture["hidden_dims"])

        if architecture["input_dim"] is None or architecture["output_dim"] is None or architecture["hidden_dims"] is None:
            raise ValueError(
                "Student checkpoint is missing architecture metadata. Provide input_dim, output_dim, and hidden_dims to load a legacy checkpoint."
            )

        model = StudentPolicy(
            input_dim=int(architecture["input_dim"]),
            output_dim=int(architecture["output_dim"]),
            hidden_dims=list(architecture["hidden_dims"]),
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        return model


def infer_student_architecture_from_state_dict(state_dict) -> tuple[int, int, list[int]]:
    weight_keys = sorted(
        [key for key in state_dict.keys() if key.startswith("net.") and key.endswith(".weight")],
        key=lambda key: int(key.split(".")[1]),
    )
    if not weight_keys:
        raise ValueError("Could not infer architecture from checkpoint state dict.")

    input_dim = int(state_dict[weight_keys[0]].shape[1])
    output_dim = int(state_dict[weight_keys[-1]].shape[0])
    hidden_dims = [int(state_dict[key].shape[0]) for key in weight_keys[:-1]]
    return input_dim, output_dim, hidden_dims


def load_student_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> tuple[StudentPolicy, dict[str, object]]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    input_dim = output_dim = None
    hidden_dims = None
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        input_dim = checkpoint.get("input_dim")
        output_dim = checkpoint.get("output_dim")
        hidden_dims = checkpoint.get("hidden_dims")
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if input_dim is None or output_dim is None or hidden_dims is None:
        input_dim, output_dim, hidden_dims = infer_student_architecture_from_state_dict(state_dict)

    checkpoint_manager = StudentCheckpointManager(checkpoint_path.parent)
    model = checkpoint_manager.load(
        checkpoint_path=checkpoint_path,
        device=device,
        input_dim=int(input_dim),
        output_dim=int(output_dim),
        hidden_dims=list(hidden_dims),
    )
    return model, {
        "input_dim": int(input_dim),
        "output_dim": int(output_dim),
        "hidden_dims": list(hidden_dims),
    }


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


def _batched_tensor_range(num_samples: int, batch_size: int):
    for start in range(0, num_samples, batch_size):
        yield start, min(start + batch_size, num_samples)


def _percentage_error(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    numerator = torch.linalg.norm(predictions - targets, dim=1)
    denominator = torch.linalg.norm(targets, dim=1).clamp_min(1e-8)
    return 100.0 * numerator / denominator


def format_hidden_dims(hidden_dims: Sequence[int]) -> str:
    return "x".join(str(hidden_dim) for hidden_dim in hidden_dims)


def compute_snn_quality_threshold(student_mean_percentage_error: float) -> float:
    return min(
        float(SNN_MEAN_PERCENTAGE_ERROR_LIMIT),
        float(student_mean_percentage_error) * float(SNN_TEACHER_ERROR_RATIO_LIMIT),
    )


def _ensure_supported_backend(backend: str):
    if backend not in SUPPORTED_STUDENT_BACKENDS:
        raise ValueError(f"Unsupported student backend '{backend}'. Supported backends: {SUPPORTED_STUDENT_BACKENDS}")


def _get_selected_readout(model) -> str:
    conversion_config = getattr(model, "conversion_config", None)
    if conversion_config is not None and getattr(conversion_config, "readout", None) is not None:
        return conversion_config.readout
    return getattr(model, "readout", "mean")


def collect_student_activation_stats(
    student_model: StudentPolicy,
    states: TensorLike,
    batch_size: int = 128,
    device: Optional[Union[str, torch.device]] = None,
) -> dict[str, object]:
    states_tensor = torch.as_tensor(states, dtype=torch.float32)
    device = torch.device(device or next(student_model.parameters()).device)
    student_model.eval()

    modules = list(student_model.net)
    module_accumulators = [
        {
            "module_index": module_index,
            "module_name": module.__class__.__name__,
            "abs_sum": 0.0,
            "numel": 0,
            "abs_max": 0.0,
        }
        for module_index, module in enumerate(modules)
    ]

    input_abs_sum = 0.0
    input_numel = 0
    input_abs_max = 0.0
    output_abs_sum = 0.0
    output_numel = 0
    output_abs_max = 0.0

    for start, end in _batched_tensor_range(len(states_tensor), batch_size):
        batch_states = states_tensor[start:end].to(device)
        batch_input_abs = batch_states.abs()
        input_abs_sum += batch_input_abs.sum().item()
        input_numel += batch_states.numel()
        input_abs_max = max(input_abs_max, batch_input_abs.max().item())

        with torch.no_grad():
            x = batch_states
            for module_index, module in enumerate(modules):
                x = module(x)
                abs_x = x.abs()
                accumulator = module_accumulators[module_index]
                accumulator["abs_sum"] += abs_x.sum().item()
                accumulator["numel"] += x.numel()
                accumulator["abs_max"] = max(accumulator["abs_max"], abs_x.max().item())

        output_abs_sum += abs_x.sum().item()
        output_numel += x.numel()
        output_abs_max = max(output_abs_max, abs_x.max().item())

    input_abs_mean = input_abs_sum / max(1, input_numel)
    output_abs_mean = output_abs_sum / max(1, output_numel)
    reference_scale = max(input_abs_max, output_abs_max, 1e-8)

    module_stats = []
    linear_stats = []
    for accumulator in module_accumulators:
        module_stat = {
            "module_index": int(accumulator["module_index"]),
            "module_name": accumulator["module_name"],
            "abs_mean": accumulator["abs_sum"] / max(1, accumulator["numel"]),
            "abs_max": accumulator["abs_max"],
        }
        module_stats.append(module_stat)
        if module_stat["module_name"] == "Linear":
            linear_stats.append(module_stat)

    hidden_linear_stats = []
    for module_stat in linear_stats[:-1]:
        hidden_linear_stats.append(
            {
                **module_stat,
                "abs_max_ratio_to_io": module_stat["abs_max"] / reference_scale,
            }
        )

    over_threshold_count = sum(
        stat["abs_max_ratio_to_io"] > ACTIVATION_SCALE_RATIO_THRESHOLD for stat in hidden_linear_stats
    )
    hidden_linear_scale_rejected = bool(hidden_linear_stats) and over_threshold_count >= max(1, (len(hidden_linear_stats) + 1) // 2)

    return {
        "input_abs_mean": float(input_abs_mean),
        "input_abs_max": float(input_abs_max),
        "output_abs_mean": float(output_abs_mean),
        "output_abs_max": float(output_abs_max),
        "reference_scale": float(reference_scale),
        "hidden_linear_ratio_threshold": float(ACTIVATION_SCALE_RATIO_THRESHOLD),
        "hidden_linear_over_threshold_count": int(over_threshold_count),
        "hidden_linear_scale_rejected": bool(hidden_linear_scale_rejected),
        "hidden_linear_stats": hidden_linear_stats,
        "module_stats": module_stats,
    }


class StudentTrainer:
    def __init__(
        self,
        config: Optional[StudentTrainingConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
        checkpoint_manager: Optional[StudentCheckpointManager] = None,
    ):
        self.config = config or StudentTrainingConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_manager = checkpoint_manager or StudentCheckpointManager(self.config.checkpoint_dir)
        self.criterion = nn.MSELoss()
        self.model: Optional[StudentPolicy] = None

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
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
            )
        return train_loader, val_loader

    def _build_model(self, input_dim: int, output_dim: int) -> StudentPolicy:
        return StudentPolicy(input_dim=input_dim, output_dim=output_dim, hidden_dims=self.config.hidden_dims).to(self.device)

    def train(self, dataset_path: Optional[Union[str, Path]] = None, initial_model: Optional[StudentPolicy] = None) -> TrainingArtifacts:
        dataset = TeacherStudentDataset.from_npz(self._resolve_dataset_path(dataset_path))
        return self.train_dataset(dataset, initial_model=initial_model)

    def train_on_arrays(
        self,
        states: TensorLike,
        actions: TensorLike,
        iteration: Optional[int] = None,
        initial_model: Optional[StudentPolicy] = None,
    ) -> TrainingArtifacts:
        dataset = TeacherStudentDataset.from_arrays(states=states, actions=actions)
        return self.train_dataset(dataset, iteration=iteration, initial_model=initial_model)

    def train_dataset(
        self,
        dataset: TeacherStudentDataset,
        iteration: Optional[int] = None,
        initial_model: Optional[StudentPolicy] = None,
    ) -> TrainingArtifacts:
        train_loader, val_loader = self._create_loaders(dataset)
        input_dim = int(dataset.states.shape[1])
        output_dim = int(dataset.actions.shape[1])

        self.model = initial_model.to(self.device) if initial_model is not None else self._build_model(input_dim, output_dim)
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        best_val_loss = float("inf")
        last_train_loss = float("inf")
        last_val_loss = float("inf")
        best_checkpoint_path = self.checkpoint_manager.best_path(self.config)
        latest_checkpoint_path = self.checkpoint_manager.latest_path(self.config)

        for epoch in range(self.config.epochs):
            self.model.train()
            train_loss = 0.0
            for states, actions in train_loader:
                states = states.to(self.device)
                actions = actions.to(self.device)
                optimizer.zero_grad()
                predicted_actions = self.model(states)
                loss = self.criterion(predicted_actions, actions)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
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
                        predicted_actions = self.model(states)
                        loss = self.criterion(predicted_actions, actions)
                        val_loss += loss.item()
                val_loss = val_loss / max(1, len(val_loader))
            last_val_loss = val_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.checkpoint_manager.save(
                    self.model,
                    best_checkpoint_path,
                    extra_metadata={
                        "best_val_loss": best_val_loss,
                        "epoch": epoch + 1,
                    },
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
                extra_metadata={
                    "best_val_loss": best_val_loss,
                    "iteration": iteration,
                },
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


class SNNPolicy(nn.Module):
    def __init__(self, conversion_config: Optional[SNNConversionConfig] = None):
        super().__init__()
        self.conversion_config = conversion_config or SNNConversionConfig()
        self.device = torch.device(self.conversion_config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.converted_network = None
        self.ann_architecture: Optional[Dict[str, object]] = None
        if self.conversion_config.readout not in SUPPORTED_SNN_READOUTS:
            raise ValueError(
                f"Unsupported SNN readout '{self.conversion_config.readout}'. Supported readouts: {SUPPORTED_SNN_READOUTS}"
            )
        if self.conversion_config.output_activation not in SUPPORTED_SNN_OUTPUT_ACTIVATIONS:
            raise ValueError(
                f"Unsupported SNN output activation '{self.conversion_config.output_activation}'. Supported output activations: {SUPPORTED_SNN_OUTPUT_ACTIVATIONS}"
            )

    @classmethod
    def from_student(cls, student_model: StudentPolicy, conversion_config: Optional[SNNConversionConfig] = None):
        snn_policy = cls(conversion_config=conversion_config)
        snn_policy.convert(student_model)
        return snn_policy

    def build_output_activation(self):
        output_activation = self.conversion_config.output_activation
        if output_activation == "sigma":
            return lava_conversion.activation.Sigma()
        if output_activation == "sdrelu":
            return lava_conversion.activation.SigmaDeltaReLU(
                threshold=self.conversion_config.threshold
            )
        if output_activation == "delta":
            return lava_conversion.activation.Delta(threshold=0.0)
        raise ValueError(
            f"Unsupported SNN output activation '{output_activation}'. Supported output activations: {SUPPORTED_SNN_OUTPUT_ACTIVATIONS}"
        )

    def build_converted_network(self, student_model: StudentPolicy):
        linear_layers = [module for module in student_model.net if isinstance(module, nn.Linear)]
        if not linear_layers:
            raise ValueError("StudentPolicy does not contain any linear layers to convert.")

        network = lava_conversion.network.Network()
        network.add_block(lava_conversion.block.Flatten())
        map_dense_block = lava_conversion.translator.map_dense_block

        for dense_layer in linear_layers[:-1]:
            network.add_block(
                map_dense_block(
                    dense=dense_layer,
                    bn=None,
                    activation=lava_conversion.activation.SigmaDeltaReLU(
                        threshold=self.conversion_config.threshold
                    ),
                )
            )

        network.add_block(
            map_dense_block(
                dense=linear_layers[-1],
                bn=None,
                activation=self.build_output_activation(),
            )
        )
        network.output_idx = [len(network.blocks) - 1]
        network.full_precision()
        return network

    def convert(self, student_model: StudentPolicy):
        if lava_conversion is None:
            raise ImportError("lava.lib.dl.conversion is required to build SNNPolicy.")

        original_device = next(student_model.parameters()).device
        conversion_model = StudentPolicy(**student_model.get_architecture_config())
        conversion_model.load_state_dict(student_model.state_dict())
        conversion_model = conversion_model.to("cpu")
        conversion_model.eval()

        network = self.build_converted_network(conversion_model)
        self.converted_network = network.to(self.device)
        self.converted_network.eval()
        self.ann_architecture = student_model.get_architecture_config()
        student_model.to(original_device)
        student_model.eval()
        return self

    def encode_input(self, states: TensorLike) -> torch.Tensor:
        tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1).unsqueeze(-1)
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(-1)
        if tensor.ndim != 5:
            raise ValueError(f"Expected encoded SNN input to have 5 dimensions, received shape {tuple(tensor.shape)}")

        if tensor.shape[-1] == 1 and self.conversion_config.timesteps > 1:
            tensor = tensor.repeat(1, 1, 1, 1, self.conversion_config.timesteps)
        elif tensor.shape[-1] != self.conversion_config.timesteps:
            raise ValueError(
                f"Encoded input timestep dimension {tensor.shape[-1]} does not match configured timesteps {self.conversion_config.timesteps}."
            )
        return tensor

    def raw_forward(self, states: TensorLike) -> torch.Tensor:
        if self.converted_network is None:
            raise RuntimeError("SNNPolicy has not been converted from a StudentPolicy yet.")

        snn_input = self.encode_input(states)
        raw_output = self.converted_network(snn_input)
        while raw_output.ndim > 3 and raw_output.shape[-2] == 1:
            raw_output = raw_output.squeeze(-2)
        if raw_output.ndim == 2:
            raw_output = raw_output.unsqueeze(-1)
        return raw_output

    def decode_output(self, raw_output: torch.Tensor, readout: Optional[str] = None) -> torch.Tensor:
        readout = readout or self.conversion_config.readout
        if readout not in SUPPORTED_SNN_READOUTS:
            raise ValueError(f"Unsupported SNN readout '{readout}'. Supported readouts: {SUPPORTED_SNN_READOUTS}")
        if raw_output.ndim != 3:
            raise ValueError(f"Expected raw SNN output to have shape (B, C, T), received {tuple(raw_output.shape)}")

        if readout == "mean":
            return raw_output.mean(dim=-1)
        if readout == "last":
            return raw_output[..., -1]
        return raw_output.sum(dim=-1)

    def forward(self, states: TensorLike, readout: Optional[str] = None) -> torch.Tensor:
        raw_output = self.raw_forward(states)
        return self.decode_output(raw_output, readout=readout)

    def predict(self, states: TensorLike, readout: Optional[str] = None) -> np.ndarray:
        with torch.no_grad():
            return self(states, readout=readout).detach().cpu().numpy()

    def export(self, export_path: Union[str, Path]) -> str:
        if self.converted_network is None:
            raise RuntimeError("SNNPolicy has not been converted from a StudentPolicy yet.")
        if self.ann_architecture is None:
            raise RuntimeError("SNNPolicy is missing ANN architecture metadata required for export.")

        if any(getattr(block, "shape", None) is None for block in self.converted_network.blocks):
            dummy_input = torch.zeros((1, int(self.ann_architecture["input_dim"])), dtype=torch.float32)
            with torch.no_grad():
                _ = self.raw_forward(dummy_input.numpy())

        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        self.converted_network.export_hdf5(str(export_path))
        return str(export_path)


def evaluate_ann_snn_parity(
    student_model: StudentPolicy,
    snn_model: SNNPolicy,
    states: TensorLike,
    batch_size: int = 128,
    device: Optional[Union[str, torch.device]] = None,
    calibration_samples: Optional[int] = None,
) -> dict[str, object]:
    states_tensor = torch.as_tensor(states, dtype=torch.float32)
    if calibration_samples is not None:
        states_tensor = states_tensor[:calibration_samples]

    device = torch.device(device or next(student_model.parameters()).device)
    student_model.eval()

    diagnostics = {
        readout: {
            "sq_error": 0.0,
            "abs_error_sum": 0.0,
            "output_abs_sum": 0.0,
            "numel": 0,
        }
        for readout in SUPPORTED_SNN_READOUTS
    }
    ann_sq_sum = 0.0
    ann_abs_sum = 0.0

    for start, end in _batched_tensor_range(len(states_tensor), batch_size):
        batch_states = states_tensor[start:end]
        with torch.no_grad():
            ann_out = student_model(batch_states.to(device)).detach().cpu()
            raw_snn_out = snn_model.raw_forward(batch_states.numpy()).detach().cpu()

        ann_sq_sum += torch.square(ann_out).sum().item()
        ann_abs_sum += ann_out.abs().sum().item()

        for readout in SUPPORTED_SNN_READOUTS:
            decoded = snn_model.decode_output(raw_snn_out, readout=readout)
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

    selected_readout = _get_selected_readout(snn_model)
    selected_metrics = readout_metrics[selected_readout]
    return {
        "selected_readout": selected_readout,
        "samples": int(len(states_tensor)),
        "relative_l2_error": float(selected_metrics["relative_l2_error"]),
        "mae": float(selected_metrics["mae"]),
        "output_scale_ratio": float(selected_metrics["output_scale_ratio"]),
        "readout_diagnostics": {
            name: {metric: float(value) for metric, value in metrics.items()}
            for name, metrics in readout_metrics.items()
        },
    }


def evaluate_models_against_teacher(
    student_model: StudentPolicy,
    snn_model: SNNPolicy,
    states: TensorLike,
    teacher_actions: TensorLike,
    batch_size: int = 128,
    device: Optional[Union[str, torch.device]] = None,
    readout: Optional[str] = None,
) -> dict[str, float]:
    states_tensor = torch.as_tensor(states, dtype=torch.float32)
    teacher_tensor = torch.as_tensor(teacher_actions, dtype=torch.float32)
    device = torch.device(device or next(student_model.parameters()).device)
    student_model.eval()

    student_errors = []
    snn_errors = []
    for start, end in _batched_tensor_range(len(states_tensor), batch_size):
        batch_states = states_tensor[start:end]
        batch_targets = teacher_tensor[start:end]
        with torch.no_grad():
            student_actions = student_model(batch_states.to(device)).detach().cpu()
        snn_actions = torch.from_numpy(snn_model.predict(batch_states.numpy(), readout=readout)).float()
        student_errors.append(_percentage_error(student_actions, batch_targets))
        snn_errors.append(_percentage_error(snn_actions, batch_targets))

    student_errors = torch.cat(student_errors)
    snn_errors = torch.cat(snn_errors)
    return {
        "student_mean_percentage_error": float(student_errors.mean().item()),
        "student_median_percentage_error": float(student_errors.median().item()),
        "snn_mean_percentage_error": float(snn_errors.mean().item()),
        "snn_median_percentage_error": float(snn_errors.median().item()),
    }


class PolicyDistillationPipeline:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        student_hidden_dims: Optional[Sequence[int]] = None,
        training_config: Optional[StudentTrainingConfig] = None,
        snn_config: Optional[SNNConversionConfig] = None,
        bootstrap_config: Optional[BootstrapTrainingConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.training_config = training_config or StudentTrainingConfig(hidden_dims=list(student_hidden_dims or DEFAULT_HIDDEN_DIMS))
        _ensure_supported_backend(self.training_config.backend)
        if student_hidden_dims is not None:
            self.training_config.hidden_dims = list(student_hidden_dims)
        self.backend = self.training_config.backend
        self.snn_config = snn_config or SNNConversionConfig(device=str(self.device))
        if self.snn_config.device is None:
            self.snn_config.device = str(self.device)

        self.checkpoint_manager = StudentCheckpointManager(self.training_config.checkpoint_dir)
        self.student_trainer = None
        self.bootstrap_pipeline = None
        self.student_model = None
        self.snn_policy = None

        if self.backend == "bootstrap":
            if BootstrapPolicyPipeline is None or BootstrapTrainingConfig is None:
                raise ImportError("bootstrap_backend is required for the bootstrap student backend.")
            self.bootstrap_config = bootstrap_config or BootstrapTrainingConfig(
                dataset_path=self.training_config.dataset_path,
                batch_size=self.training_config.batch_size,
                learning_rate=self.training_config.learning_rate,
                epochs=self.training_config.epochs,
                hidden_dims=list(self.training_config.hidden_dims),
                val_split=self.training_config.val_split,
                num_workers=self.training_config.num_workers,
                checkpoint_dir=self.training_config.checkpoint_dir,
                best_checkpoint_name=self.training_config.best_checkpoint_name,
                latest_checkpoint_name=self.training_config.latest_checkpoint_name,
                iteration_checkpoint_template=self.training_config.iteration_checkpoint_template,
            )
            self.bootstrap_pipeline = BootstrapPolicyPipeline(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                hidden_dims=self.training_config.hidden_dims,
                training_config=self.bootstrap_config,
                device=self.device,
            )
        else:
            self.bootstrap_config = bootstrap_config
            self.student_trainer = StudentTrainer(
                config=self.training_config,
                device=self.device,
                checkpoint_manager=self.checkpoint_manager,
            )

    def build_student_policy(self):
        if self.backend == "bootstrap":
            self.student_model = self.bootstrap_pipeline.build_student_policy()
        else:
            self.student_model = StudentPolicy(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                hidden_dims=self.training_config.hidden_dims,
            ).to(self.device)
        return self.student_model

    def load_student_policy(self, checkpoint_path: Union[str, Path]):
        if self.backend == "bootstrap":
            self.student_model = self.bootstrap_pipeline.load_student_policy(checkpoint_path)
        else:
            self.student_model = self.checkpoint_manager.load(
                checkpoint_path=checkpoint_path,
                device=self.device,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                hidden_dims=self.training_config.hidden_dims,
            )
        return self.student_model

    def train_offline(self, dataset_path: Optional[Union[str, Path]] = None) -> TrainingArtifacts:
        if self.backend == "bootstrap":
            artifacts = self.bootstrap_pipeline.train_offline(dataset_path=dataset_path)
            self.student_model = self.bootstrap_pipeline.bootstrap_model
            return artifacts
        artifacts = self.student_trainer.train(dataset_path=dataset_path, initial_model=self.student_model)
        self.student_model = self.student_trainer.model
        return artifacts

    def train_on_aggregated_data(
        self,
        states: TensorLike,
        actions: TensorLike,
        iteration: Optional[int] = None,
    ) -> TrainingArtifacts:
        if self.backend == "bootstrap":
            artifacts = self.bootstrap_pipeline.train_on_aggregated_data(states=states, actions=actions, iteration=iteration)
            self.student_model = self.bootstrap_pipeline.bootstrap_model
            return artifacts
        artifacts = self.student_trainer.train_on_arrays(
            states=states,
            actions=actions,
            iteration=iteration,
            initial_model=self.student_model,
        )
        self.student_model = self.student_trainer.model
        return artifacts

    def build_snn_policy(
        self,
        student_checkpoint_path: Optional[Union[str, Path]] = None,
        student_model: Optional[Union[StudentPolicy, BootstrapStudentPolicy]] = None,
    ):
        if self.backend == "bootstrap":
            self.snn_policy = self.bootstrap_pipeline.build_snn_policy(
                student_checkpoint_path=student_checkpoint_path,
                student_model=student_model,
            )
            self.student_model = self.bootstrap_pipeline.bootstrap_model
            return self.snn_policy

        if student_model is None:
            if student_checkpoint_path is not None:
                student_model = self.load_student_policy(student_checkpoint_path)
            elif self.student_model is not None:
                student_model = self.student_model
            else:
                raise ValueError("A student model or checkpoint path is required to build the SNN policy.")

        self.snn_policy = SNNPolicy.from_student(student_model=student_model, conversion_config=self.snn_config)
        return self.snn_policy

    def evaluate_parity(self, states: TensorLike, batch_size: int = 128) -> dict[str, object]:
        if self.student_model is None or self.snn_policy is None:
            raise RuntimeError("Student and SNN policies must both be initialized before parity evaluation.")
        return evaluate_ann_snn_parity(
            student_model=self.student_model,
            snn_model=self.snn_policy,
            states=states,
            batch_size=batch_size,
            device=self.device,
            calibration_samples=self.snn_config.calibration_samples if self.backend != "bootstrap" else None,
        )

    def evaluate_against_teacher(
        self,
        states: TensorLike,
        teacher_actions: TensorLike,
        batch_size: int = 128,
    ) -> dict[str, float]:
        if self.student_model is None or self.snn_policy is None:
            raise RuntimeError("Student and SNN policies must both be initialized before teacher evaluation.")
        return evaluate_models_against_teacher(
            student_model=self.student_model,
            snn_model=self.snn_policy,
            states=states,
            teacher_actions=teacher_actions,
            batch_size=batch_size,
            device=self.device,
            readout=self.snn_config.readout if self.backend != "bootstrap" else self.bootstrap_config.readout,
        )

    def predict_student(self, states: TensorLike) -> np.ndarray:
        if self.student_model is None:
            raise RuntimeError("Student policy has not been initialized.")
        if self.backend == "bootstrap":
            return self.bootstrap_pipeline.predict_student(states)

        self.student_model.eval()
        state_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.student_model(state_tensor).detach().cpu().numpy()

    def predict_snn(self, states: TensorLike, readout: Optional[str] = None) -> np.ndarray:
        if self.snn_policy is None:
            raise RuntimeError("SNN policy has not been initialized.")
        if self.backend == "bootstrap":
            return self.bootstrap_pipeline.predict_snn(states, readout=readout)
        return self.snn_policy.predict(states, readout=readout)

def search_conversion_target_students(
    dataset_path: Union[str, Path],
    output_dir: Union[str, Path],
    candidate_architectures: Optional[Sequence[Sequence[int]]] = None,
    threshold_values: Sequence[float] = (0.05, 0.1, 0.2),
    timestep_values: Sequence[int] = (5, 10, 20),
    train_batch_size: int = 64,
    evaluation_batch_size: int = 128,
    learning_rate: float = 1e-4,
    epochs: int = 60,
    max_samples: int = 2048,
    train_max_samples: Optional[int] = None,
    calibration_samples: Optional[int] = None,
    readout: str = "mean",
    device: Optional[Union[str, torch.device]] = None,
    val_split: float = 0.2,
    include_baseline: bool = True,
    baseline_hidden_dims: Optional[Sequence[int]] = None,
    baseline_reference: Optional[dict[str, float]] = None,
    student_error_regression_tolerance: float = STUDENT_ERROR_REGRESSION_TOLERANCE,
    snn_teacher_error_ratio_limit: float = SNN_TEACHER_ERROR_RATIO_LIMIT,
    snn_mean_percentage_error_limit: float = SNN_MEAN_PERCENTAGE_ERROR_LIMIT,
) -> dict[str, object]:
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if readout not in SUPPORTED_SNN_READOUTS:
        raise ValueError(f"Unsupported SNN readout '{readout}'. Supported readouts: {SUPPORTED_SNN_READOUTS}")

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    evaluation_states, evaluation_actions = load_teacher_dataset(dataset_path, max_samples=max_samples)

    train_states = train_actions = None
    if train_max_samples is not None:
        train_states, train_actions = load_teacher_dataset(dataset_path, max_samples=train_max_samples)

    baseline_hidden_dims = list(baseline_hidden_dims or DEFAULT_HIDDEN_DIMS)
    candidate_architectures = [list(hidden_dims) for hidden_dims in (candidate_architectures or CONVERSION_TARGET_HIDDEN_DIMS)]
    if include_baseline and baseline_hidden_dims not in candidate_architectures:
        candidate_architectures.append(baseline_hidden_dims)

    baseline_reference = dict(BASELINE_PARITY_REFERENCE if baseline_reference is None else baseline_reference)
    baseline_scale_distance = abs(float(baseline_reference["output_scale_ratio"]) - 1.0)
    student_error_threshold = float(baseline_reference["student_mean_percentage_error"]) * (1.0 + student_error_regression_tolerance)

    conversion_steps_per_architecture = len(threshold_values) * len(timestep_values)
    total_progress_steps = len(candidate_architectures) * (1 + conversion_steps_per_architecture)
    progress_bar = None
    if tqdm is not None:
        progress_bar = tqdm(total=total_progress_steps, desc="conversion-target search", unit="step")

    try:
        results = []
        for hidden_dims in candidate_architectures:
            architecture_name = format_hidden_dims(hidden_dims)
            if progress_bar is not None:
                progress_bar.set_description(f"train {architecture_name}")
            checkpoint_dir = output_dir / architecture_name
            training_config = StudentTrainingConfig(
                dataset_path=str(dataset_path),
                batch_size=train_batch_size,
                learning_rate=learning_rate,
                epochs=epochs,
                hidden_dims=list(hidden_dims),
                val_split=val_split,
                checkpoint_dir=str(checkpoint_dir),
            )
            trainer = StudentTrainer(config=training_config, device=resolved_device)

            if train_states is not None and train_actions is not None:
                artifacts = trainer.train_on_arrays(states=train_states, actions=train_actions)
            else:
                artifacts = trainer.train(dataset_path=dataset_path)
            if progress_bar is not None:
                progress_bar.update(1)

            if trainer.model is None:
                raise RuntimeError(f"Student model for architecture {architecture_name} was not initialized after training.")

            student_model = trainer.model
            activation_stats = collect_student_activation_stats(
                student_model=student_model,
                states=evaluation_states,
                batch_size=evaluation_batch_size,
                device=resolved_device,
            )

            conversion_results = []
            for threshold in threshold_values:
                for timesteps in timestep_values:
                    if progress_bar is not None:
                        progress_bar.set_description(
                            f"eval {architecture_name} t={threshold:.2f} T={timesteps}"
                        )
                    snn_model = SNNPolicy.from_student(
                        student_model=student_model,
                        conversion_config=SNNConversionConfig(
                            threshold=float(threshold),
                            timesteps=int(timesteps),
                            device=str(resolved_device),
                            readout=readout,
                            calibration_samples=calibration_samples,
                        ),
                    )
                    parity_metrics = evaluate_ann_snn_parity(
                        student_model=student_model,
                        snn_model=snn_model,
                        states=evaluation_states,
                        batch_size=evaluation_batch_size,
                        device=resolved_device,
                        calibration_samples=calibration_samples,
                    )
                    teacher_metrics = evaluate_models_against_teacher(
                        student_model=student_model,
                        snn_model=snn_model,
                        states=evaluation_states,
                        teacher_actions=evaluation_actions,
                        batch_size=evaluation_batch_size,
                        device=resolved_device,
                        readout=readout,
                    )
                    conversion_results.append(
                        {
                            "threshold": float(threshold),
                            "timesteps": int(timesteps),
                            "parity_relative_l2_error": float(parity_metrics["relative_l2_error"]),
                            "parity_mae": float(parity_metrics["mae"]),
                            "parity_output_scale_ratio": float(parity_metrics["output_scale_ratio"]),
                            "readout_diagnostics": parity_metrics["readout_diagnostics"],
                            "student_mean_percentage_error": float(teacher_metrics["student_mean_percentage_error"]),
                            "student_median_percentage_error": float(teacher_metrics["student_median_percentage_error"]),
                            "snn_mean_percentage_error": float(teacher_metrics["snn_mean_percentage_error"]),
                            "snn_median_percentage_error": float(teacher_metrics["snn_median_percentage_error"]),
                        }
                    )
                    if progress_bar is not None:
                        progress_bar.update(1)

            conversion_results.sort(
                key=lambda result: (
                    result["parity_relative_l2_error"],
                    abs(result["parity_output_scale_ratio"] - 1.0),
                    result["student_mean_percentage_error"],
                )
            )
            best_conversion = conversion_results[0]
            snn_quality_threshold = min(
                float(snn_mean_percentage_error_limit),
                float(best_conversion["student_mean_percentage_error"]) * float(snn_teacher_error_ratio_limit),
            )
            passes_snn_quality_gate = best_conversion["snn_mean_percentage_error"] <= snn_quality_threshold
            best_conversion["snn_quality_threshold"] = float(snn_quality_threshold)
            best_conversion["passes_snn_quality_gate"] = bool(passes_snn_quality_gate)
            passes_baseline_thresholds = (
                not activation_stats["hidden_linear_scale_rejected"]
                and best_conversion["parity_relative_l2_error"] < float(baseline_reference["relative_l2_error"])
                and abs(best_conversion["parity_output_scale_ratio"] - 1.0) < baseline_scale_distance
                and best_conversion["student_mean_percentage_error"] <= student_error_threshold
                and passes_snn_quality_gate
            )

            results.append(
                {
                    "hidden_dims": list(hidden_dims),
                    "architecture_name": architecture_name,
                    "checkpoint_dir": str(checkpoint_dir),
                    "best_checkpoint_path": artifacts.best_checkpoint_path,
                    "latest_checkpoint_path": artifacts.latest_checkpoint_path,
                    "training_best_val_loss": float(artifacts.best_val_loss),
                    "training_last_train_loss": float(artifacts.last_train_loss),
                    "training_last_val_loss": float(artifacts.last_val_loss),
                    "activation_stats": activation_stats,
                    "activation_scale_rejected": bool(activation_stats["hidden_linear_scale_rejected"]),
                    "passes_snn_quality_gate": bool(passes_snn_quality_gate),
                    "passes_baseline_thresholds": bool(passes_baseline_thresholds),
                    "best_conversion": best_conversion,
                    "all_conversion_results": conversion_results,
                }
            )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    results.sort(
        key=lambda result: (
            result["activation_scale_rejected"],
            not result["passes_baseline_thresholds"],
            result["best_conversion"]["parity_relative_l2_error"],
            abs(result["best_conversion"]["parity_output_scale_ratio"] - 1.0),
            result["best_conversion"]["student_mean_percentage_error"],
        )
    )

    best_result = results[0]
    best_accepted_result = next((result for result in results if result["passes_baseline_thresholds"]), None)
    baseline_result = next((result for result in results if result["hidden_dims"] == baseline_hidden_dims), None)

    return {
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "device": str(resolved_device),
        "readout": readout,
        "evaluation_samples": int(len(evaluation_states)),
        "train_samples": int(len(train_states)) if train_states is not None else None,
        "baseline_reference": {
            "relative_l2_error": float(baseline_reference["relative_l2_error"]),
            "output_scale_ratio": float(baseline_reference["output_scale_ratio"]),
            "student_mean_percentage_error": float(baseline_reference["student_mean_percentage_error"]),
            "student_error_threshold": float(student_error_threshold),
            "snn_teacher_error_ratio_limit": float(snn_teacher_error_ratio_limit),
            "snn_mean_percentage_error_limit": float(snn_mean_percentage_error_limit),
        },
        "results": results,
        "best_result": best_result,
        "best_accepted_result": best_accepted_result,
        "baseline_result": baseline_result,
    }


def train():
    config = StudentTrainingConfig()
    trainer = StudentTrainer(config=config)
    dataset_path = trainer._resolve_dataset_path(config.dataset_path)
    if not dataset_path.exists():
        print(f"Error: {dataset_path} not found.")
        return

    print(f"Using device: {trainer.device}")
    print(f"Loading dataset from {dataset_path}...")
    dataset = TeacherStudentDataset.from_npz(dataset_path)
    print(f"Dataset loaded. Total samples: {len(dataset)}")
    print(f"Input dim: {dataset.states.shape[1]}, Output dim: {dataset.actions.shape[1]}")
    print("Starting training...")

    artifacts = trainer.train_dataset(dataset)

    print(f"Training complete. Best Val Loss: {artifacts.best_val_loss:.6f}")
    print(f"Best model saved to '{artifacts.best_checkpoint_path}'")


if __name__ == "__main__":
    train()
