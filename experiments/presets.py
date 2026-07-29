from copy import deepcopy
from pathlib import Path

import yaml

from one_policy_to_run_them_all.paths import project_path


CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "presets.yaml"
ROBOT_TYPE_KEYS = {"environment.train_robot_types", "environment.eval_robot_types"}
REQUIRED_PRESET_KEYS = ("algorithm.name", "environment.name", "runner.mode", "environment.mode")
SUPPORTED_STUDENT_BACKENDS = {"ann", "bootstrap"}
SUPPORTED_ROLLOUT_POLICY_STAGES = {"teacher", "student", "snn"}
SUPPORTED_READOUTS = {"mean", "last", "sum"}
SUPPORTED_WEIGHT_QUANTIZATION_MODES = {"legacy_8bit", "decomposed"}
SUPPORTED_WEIGHT_QUANTIZATION_SCOPES = {"all", "first"}



def _resolve_value(value):
    if isinstance(value, str):
        return value.replace("{project_root}", str(project_path()))
    if isinstance(value, list):
        return [_resolve_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_value(item) for key, item in value.items()}
    return value


def _normalize_preset_values(preset):
    normalized = {}
    for key, value in preset.items():
        value = _resolve_value(value)
        if key in ROBOT_TYPE_KEYS and isinstance(value, list):
            value = tuple(value)
        normalized[key] = value
    return normalized


def _load_config(config_path=CONFIG_PATH):
    with open(config_path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Preset config {config_path} must contain a mapping.")
    return config


def validate_preset(name, preset):
    missing = [key for key in REQUIRED_PRESET_KEYS if key not in preset]
    if missing:
        raise ValueError(f"Preset '{name}' is missing required keys: {missing}")

    student_backend = preset.get("algorithm.student_backend", "ann")
    if student_backend not in SUPPORTED_STUDENT_BACKENDS:
        raise ValueError(
            f"Preset '{name}' has unsupported algorithm.student_backend '{student_backend}'. "
            f"Available: {sorted(SUPPORTED_STUDENT_BACKENDS)}"
        )

    rollout_policy_stage = preset.get("algorithm.rollout_policy_stage")
    if rollout_policy_stage is not None and rollout_policy_stage not in SUPPORTED_ROLLOUT_POLICY_STAGES:
        raise ValueError(
            f"Preset '{name}' has unsupported algorithm.rollout_policy_stage '{rollout_policy_stage}'. "
            f"Available: {sorted(SUPPORTED_ROLLOUT_POLICY_STAGES)}"
        )

    readout = preset.get("algorithm.bootstrap_readout")
    if readout is not None and readout not in SUPPORTED_READOUTS:
        raise ValueError(
            f"Preset '{name}' has unsupported algorithm.bootstrap_readout '{readout}'. "
            f"Available: {sorted(SUPPORTED_READOUTS)}"
        )

    quantization_mode = preset.get(
        "algorithm.bootstrap_weight_quantization_mode"
    )
    if (
        quantization_mode is not None
        and quantization_mode not in SUPPORTED_WEIGHT_QUANTIZATION_MODES
    ):
        raise ValueError(
            f"Preset '{name}' has unsupported weight quantization mode "
            f"{quantization_mode!r}."
        )
    quantization_scope = preset.get(
        "algorithm.bootstrap_weight_quantization_scope"
    )
    if (
        quantization_scope is not None
        and quantization_scope not in SUPPORTED_WEIGHT_QUANTIZATION_SCOPES
    ):
        raise ValueError(
            f"Preset '{name}' has unsupported weight quantization scope "
            f"{quantization_scope!r}."
        )
    sign_mode = preset.get(
        "algorithm.bootstrap_weight_quantization_sign_mode"
    )
    if sign_mode is not None and sign_mode not in {
        "mixed", "excitatory", "inhibitory"
    }:
        raise ValueError(
            f"Preset '{name}' has unsupported weight quantization sign "
            f"mode {sign_mode!r}."
        )


def load_presets(config_path=CONFIG_PATH):
    config = _load_config(config_path)
    base_experiment = _normalize_preset_values(config.get("base_experiment", {}))
    raw_presets = config.get("presets", {})
    if not raw_presets:
        raise ValueError(f"Preset config {config_path} must define at least one preset.")

    presets = {}
    for name, overrides in raw_presets.items():
        preset = deepcopy(base_experiment)
        preset.update(_normalize_preset_values(overrides or {}))
        validate_preset(name, preset)
        presets[name] = preset
    return presets


def load_record_robots(config_path=CONFIG_PATH):
    config = _load_config(config_path)
    robots = tuple(config.get("record_robots", ()))
    if not robots:
        raise ValueError(f"Preset config {config_path} must define record_robots.")
    return robots


RECORD_ROBOTS = load_record_robots()
PRESETS = load_presets()


def get_record_robot(robot_index):
    try:
        return RECORD_ROBOTS[int(robot_index)]
    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"Invalid record robot index {robot_index}. Use an index from 0 to {len(RECORD_ROBOTS) - 1}."
        ) from exc


# Compatibility constants retained for callers that imported them from this module.
BASE_EXPERIMENT = _normalize_preset_values(_load_config().get("base_experiment", {}))
STUDENT_MODEL_PATH = str(project_path("student", "student_model_best.pth"))
D300K_STUDENT_MODEL_PATH = str(
    project_path(
        "experiments",
        "teacher-student",
        "conversion_target_search_d300k",
        "1024x1024x1024x1024x1024",
        "student_model_best.pth",
    )
)
ONLINE_DAGGER_SNN_DIR = str(project_path("experiments", "teacher-student", "online_dagger_snn"))
BOOTSTRAP_MODEL_PATH = str(project_path("experiments", "teacher-student", "bootstrap_parity", "student_model_best.pth"))
ONLINE_DAGGER_BOOTSTRAP_DIR = str(project_path("experiments", "teacher-student", "online_dagger_bootstrap"))
DEFAULT_RECORD_ROBOT = "unitree_h1"
DEFAULT_RECORD_OVERRIDES = {
    "environment.nr_envs": 1,
    "environment.train_robot_types": (DEFAULT_RECORD_ROBOT,),
    "environment.multi_render": False,
    "algorithm.record": True,
    "algorithm.record_robot_index": 0,
}


def build_preset(**overrides):
    preset = deepcopy(BASE_EXPERIMENT)
    preset.update(overrides)
    return preset
