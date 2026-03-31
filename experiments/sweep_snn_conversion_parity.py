import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
STUDENT_DIR = ROOT / "student"
DEFAULT_DATASET = ROOT / "experiments" / "teacher-student" / "traindata" / "teacher_dataset_100k.npz"
if str(STUDENT_DIR) not in sys.path:
    sys.path.append(str(STUDENT_DIR))

from train_student import (
    FOCUSED_SWEEP_THRESHOLDS,
    FOCUSED_SWEEP_TIMESTEPS,
    SNNConversionConfig,
    SNN_MEAN_PERCENTAGE_ERROR_LIMIT,
    SNN_TEACHER_ERROR_RATIO_LIMIT,
    SUPPORTED_SNN_OUTPUT_ACTIVATIONS,
    SUPPORTED_SNN_READOUTS,
    SNNPolicy,
    evaluate_ann_snn_parity,
    evaluate_models_against_teacher,
    load_student_model_from_checkpoint,
    load_teacher_dataset,
)


def parse_numeric_list(raw_value: str, cast_type):
    values = [cast_type(item.strip()) for item in raw_value.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one value in the sweep list.")
    return values


def parse_choice_list(raw_value: str, valid_choices: tuple[str, ...], value_name: str):
    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"Expected at least one {value_name} in the list.")
    invalid = [value for value in values if value not in valid_choices]
    if invalid:
        raise ValueError(
            f"Unsupported {value_name}(s) {invalid}. Supported {value_name}s: {valid_choices}"
        )
    return values


def resolve_checkpoint_path(checkpoint_arg: str, search_results_arg: str | None, selection: str) -> tuple[Path, str]:
    if search_results_arg is None:
        return Path(checkpoint_arg).resolve(), "checkpoint"

    search_results_path = Path(search_results_arg).resolve()
    if not search_results_path.exists():
        raise FileNotFoundError(f"Search results file not found: {search_results_path}")

    data = json.loads(search_results_path.read_text())
    candidate = None
    candidate_key = None
    if selection == "best_accepted_result":
        candidate = data.get("best_accepted_result")
        candidate_key = "best_accepted_result"
        if candidate is None:
            raise ValueError(
                f"Search results at {search_results_path} do not contain a best_accepted_result. "
                "Use --search-selection best_result or auto instead."
            )
    elif selection == "best_result":
        candidate = data.get("best_result")
        candidate_key = "best_result"
    else:
        candidate = data.get("best_accepted_result")
        candidate_key = "best_accepted_result"
        if candidate is None:
            candidate = data.get("best_result")
            candidate_key = "best_result"

    if candidate is None:
        raise ValueError(f"Search results at {search_results_path} do not contain a usable result entry.")

    checkpoint_path = candidate.get("best_checkpoint_path")
    if not checkpoint_path:
        raise ValueError(
            f"Selected result '{candidate_key}' in {search_results_path} is missing best_checkpoint_path."
        )
    return Path(checkpoint_path).resolve(), f"{candidate_key} from {search_results_path}"


def main():
    parser = argparse.ArgumentParser(description="Sweep SNN conversion settings for a fixed student and rank them by ANN-vs-SNN parity.")
    parser.add_argument(
        "--checkpoint",
        default=str(STUDENT_DIR / "student_model_best.pth"),
        help="Path to the student checkpoint to load when --search-results is not provided.",
    )
    parser.add_argument(
        "--search-results",
        default=None,
        help="Optional search_results.json file. If provided, the sweep loads the selected checkpoint from it.",
    )
    parser.add_argument(
        "--search-selection",
        choices=("auto", "best_result", "best_accepted_result"),
        default="auto",
        help="Which result entry to load from --search-results. 'auto' prefers best_accepted_result, then best_result.",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to the teacher dataset (.npz with states/actions).",
    )
    parser.add_argument(
        "--thresholds",
        default=",".join(str(value) for value in FOCUSED_SWEEP_THRESHOLDS),
        help="Comma-separated threshold sweep values.",
    )
    parser.add_argument(
        "--timesteps",
        default=",".join(str(value) for value in FOCUSED_SWEEP_TIMESTEPS),
        help="Comma-separated timestep sweep values.",
    )
    parser.add_argument(
        "--output-activations",
        default=",".join(SUPPORTED_SNN_OUTPUT_ACTIVATIONS),
        help="Comma-separated output activations to test.",
    )
    parser.add_argument(
        "--readout",
        choices=SUPPORTED_SNN_READOUTS,
        default="mean",
        help="Primary readout used to rank sweep results.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for evaluation.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2048,
        help="Fixed number of real states to evaluate for each sweep point.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=None,
        help="Optional cap on samples used for parity evaluation. Defaults to all loaded samples.",
    )
    parser.add_argument(
        "--snn-teacher-error-ratio-limit",
        type=float,
        default=SNN_TEACHER_ERROR_RATIO_LIMIT,
        help="Maximum allowed SNN mean teacher error as a multiple of the student mean teacher error.",
    )
    parser.add_argument(
        "--snn-mean-percentage-error-limit",
        type=float,
        default=SNN_MEAN_PERCENTAGE_ERROR_LIMIT,
        help="Absolute cap on allowed SNN mean teacher error.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for ANN inference during the sweep.",
    )
    args = parser.parse_args()

    checkpoint_path, checkpoint_source = resolve_checkpoint_path(
        checkpoint_arg=args.checkpoint,
        search_results_arg=args.search_results,
        selection=args.search_selection,
    )
    dataset_path = Path(args.dataset).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    thresholds = parse_numeric_list(args.thresholds, float)
    timestep_grid = parse_numeric_list(args.timesteps, int)
    output_activations = parse_choice_list(
        args.output_activations,
        SUPPORTED_SNN_OUTPUT_ACTIVATIONS,
        "output activation",
    )

    device = torch.device(args.device)
    student_model, architecture = load_student_model_from_checkpoint(checkpoint_path, device=device)
    states, teacher_actions = load_teacher_dataset(dataset_path, max_samples=args.max_samples)

    results = []
    for output_activation in output_activations:
        for threshold in thresholds:
            for timestep_count in timestep_grid:
                snn_model = SNNPolicy.from_student(
                    student_model=student_model,
                    conversion_config=SNNConversionConfig(
                        threshold=threshold,
                        timesteps=timestep_count,
                        device=str(device),
                        readout=args.readout,
                        calibration_samples=args.calibration_samples,
                        output_activation=output_activation,
                    ),
                )
                parity_metrics = evaluate_ann_snn_parity(
                    student_model=student_model,
                    snn_model=snn_model,
                    states=states,
                    batch_size=args.batch_size,
                    device=device,
                    calibration_samples=args.calibration_samples,
                )
                teacher_metrics = evaluate_models_against_teacher(
                    student_model=student_model,
                    snn_model=snn_model,
                    states=states,
                    teacher_actions=teacher_actions,
                    batch_size=args.batch_size,
                    device=device,
                    readout=args.readout,
                )
                snn_quality_threshold = min(
                    float(args.snn_mean_percentage_error_limit),
                    float(teacher_metrics["student_mean_percentage_error"]) * float(args.snn_teacher_error_ratio_limit),
                )
                passes_snn_quality_gate = float(teacher_metrics["snn_mean_percentage_error"]) <= snn_quality_threshold
                result = {
                    "output_activation": output_activation,
                    "threshold": float(threshold),
                    "timesteps": int(timestep_count),
                    "parity_relative_l2_error": float(parity_metrics["relative_l2_error"]),
                    "parity_mae": float(parity_metrics["mae"]),
                    "parity_output_scale_ratio": float(parity_metrics["output_scale_ratio"]),
                    "readout_diagnostics": parity_metrics["readout_diagnostics"],
                    "student_mean_percentage_error": float(teacher_metrics["student_mean_percentage_error"]),
                    "student_median_percentage_error": float(teacher_metrics["student_median_percentage_error"]),
                    "snn_mean_percentage_error": float(teacher_metrics["snn_mean_percentage_error"]),
                    "snn_median_percentage_error": float(teacher_metrics["snn_median_percentage_error"]),
                    "snn_quality_threshold": float(snn_quality_threshold),
                    "passes_snn_quality_gate": bool(passes_snn_quality_gate),
                }
                results.append(result)
                print(
                    f"output_activation={output_activation} threshold={threshold:.4f} timesteps={timestep_count} "
                    f"rel_l2={result['parity_relative_l2_error']:.6f} "
                    f"mae={result['parity_mae']:.6f} scale_ratio={result['parity_output_scale_ratio']:.6f} "
                    f"student_mean_pct={result['student_mean_percentage_error']:.4f}% "
                    f"snn_mean_pct={result['snn_mean_percentage_error']:.4f}% "
                    f"quality_gate={result['passes_snn_quality_gate']}"
                )

    results.sort(
        key=lambda item: (
            not item["passes_snn_quality_gate"],
            item["parity_relative_l2_error"],
            abs(item["parity_output_scale_ratio"] - 1.0),
            item["student_mean_percentage_error"],
        )
    )
    best = results[0]
    best_quality_gate_result = next((item for item in results if item["passes_snn_quality_gate"]), None)

    print("best_configuration:")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  checkpoint_source: {checkpoint_source}")
    print(f"  dataset: {dataset_path}")
    print(f"  input_dim: {architecture['input_dim']}")
    print(f"  output_dim: {architecture['output_dim']}")
    print(f"  hidden_dims: {architecture['hidden_dims']}")
    print(f"  samples: {len(states)}")
    print(f"  readout: {args.readout}")
    print(f"  output_activation: {best['output_activation']}")
    print(f"  snn_teacher_error_ratio_limit: {args.snn_teacher_error_ratio_limit}")
    print(f"  snn_mean_percentage_error_limit: {args.snn_mean_percentage_error_limit}")
    print(f"  threshold: {best['threshold']}")
    print(f"  timesteps: {best['timesteps']}")
    print(f"  parity_relative_l2_error: {best['parity_relative_l2_error']:.6f}")
    print(f"  parity_mae: {best['parity_mae']:.6f}")
    print(f"  parity_output_scale_ratio: {best['parity_output_scale_ratio']:.6f}")
    print(f"  student_mean_percentage_error: {best['student_mean_percentage_error']:.4f}%")
    print(f"  snn_mean_percentage_error: {best['snn_mean_percentage_error']:.4f}%")
    print(f"  snn_quality_threshold: {best['snn_quality_threshold']:.4f}%")
    print(f"  passes_snn_quality_gate: {best['passes_snn_quality_gate']}")

    for readout, metrics in best["readout_diagnostics"].items():
        print(
            f"  readout={readout} rel_l2={metrics['relative_l2_error']:.6f} "
            f"mae={metrics['mae']:.6f} scale_ratio={metrics['output_scale_ratio']:.6f}"
        )

    if best_quality_gate_result is None:
        print("best_quality_gate_configuration: none")
    else:
        print("best_quality_gate_configuration:")
        print(f"  output_activation: {best_quality_gate_result['output_activation']}")
        print(f"  threshold: {best_quality_gate_result['threshold']}")
        print(f"  timesteps: {best_quality_gate_result['timesteps']}")
        print(f"  parity_relative_l2_error: {best_quality_gate_result['parity_relative_l2_error']:.6f}")
        print(f"  parity_output_scale_ratio: {best_quality_gate_result['parity_output_scale_ratio']:.6f}")
        print(f"  student_mean_percentage_error: {best_quality_gate_result['student_mean_percentage_error']:.4f}%")
        print(f"  snn_mean_percentage_error: {best_quality_gate_result['snn_mean_percentage_error']:.4f}%")
        print(f"  snn_quality_threshold: {best_quality_gate_result['snn_quality_threshold']:.4f}%")


if __name__ == "__main__":
    main()
