import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
STUDENT_DIR = ROOT / "student"
DEFAULT_DATASET = ROOT / "student" / "teacher_student_dagger_dataset_d300k.npz"
DEFAULT_CHECKPOINT = (
    ROOT
    / "experiments"
    / "teacher-student"
    / "conversion_target_search_d300k"
    / "1024x1024x1024x1024x1024"
    / "student_model_best.pth"
)
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "teacher-student" / "bootstrap_sweep"

if str(STUDENT_DIR) not in sys.path:
    sys.path.append(str(STUDENT_DIR))

from bootstrap_backend import (
    BootstrapTrainingConfig,
    BootstrapStudentTrainer,
    SUPPORTED_SNN_READOUTS,
    collect_bootstrap_activity_stats,
    evaluate_ann_bootstrap_parity,
    evaluate_bootstrap_against_teacher,
    load_teacher_dataset,
    resolve_bootstrap_device,
)
from train_student import (
    SNNConversionConfig,
    SNNPolicy,
    evaluate_ann_snn_parity,
    evaluate_models_against_teacher,
    load_student_model_from_checkpoint,
)

SUPPORTED_BOOTSTRAP_INPUT_STRATEGIES = ("identity", "signed_split")


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


def format_trial_name(config: dict[str, object]) -> str:
    return (
        f"strategy-{config['input_strategy']}"
        f"_iw-{config['input_weight']}"
        f"_ib-{config['input_bias']}"
        f"_thr-{config['neuron_threshold']}"
        f"_id-{config['current_decay']}"
        f"_vd-{config['voltage_decay']}"
        f"_ts-{config['timesteps']}"
        f"_ni-{config['num_sample_iter']}"
        f"_sp-{config['sample_period']}"
    ).replace("/", "_")


def main():
    parser = argparse.ArgumentParser(description="Sweep Lava bootstrap training configurations and compare them against the current conversion baseline.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="ANN student checkpoint to use as the bootstrap initialization and parity reference.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Teacher dataset (.npz with states/actions).")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to store bootstrap checkpoints and optional sweep results JSON.")
    parser.add_argument("--output-json", default="", help="Optional path to write the full sweep report as JSON.")
    parser.add_argument("--max-samples", type=int, default=2048, help="Evaluation subset size.")
    parser.add_argument("--train-max-samples", type=int, default=8192, help="Training subset size. Use 0 for full dataset.")
    parser.add_argument("--batch-size", type=int, default=128, help="Evaluation batch size.")
    parser.add_argument("--bootstrap-batch-size", type=int, default=64, help="Bootstrap training batch size.")
    parser.add_argument("--epochs", type=int, default=20, help="Bootstrap training epochs per trial.")
    parser.add_argument("--device", default="cuda", help="Requested device. Bootstrap resolves this safely.")
    parser.add_argument("--timesteps", default="3", help="Comma-separated bootstrap timestep values.")
    parser.add_argument("--readout", choices=SUPPORTED_SNN_READOUTS, default="mean", help="Primary bootstrap readout.")
    parser.add_argument("--input-strategies", default=",".join(SUPPORTED_BOOTSTRAP_INPUT_STRATEGIES), help="Comma-separated bootstrap input strategies.")
    parser.add_argument("--input-weights", default="1.0,2.0", help="Comma-separated bootstrap input weight values.")
    parser.add_argument("--input-biases", default="0.0", help="Comma-separated bootstrap input bias values.")
    parser.add_argument("--neuron-thresholds", default="0.5,1.0,1.5", help="Comma-separated bootstrap neuron thresholds.")
    parser.add_argument("--current-decays", default="0.1,0.25,0.5", help="Comma-separated bootstrap current decays.")
    parser.add_argument("--voltage-decays", default="0.01,0.03,0.1", help="Comma-separated bootstrap voltage decays.")
    parser.add_argument("--num-sample-iters", default="10", help="Comma-separated bootstrap num_sample_iter values.")
    parser.add_argument("--sample-periods", default="10", help="Comma-separated bootstrap sample_period values.")
    parser.add_argument("--bootstrap-weight-scale", type=float, default=1.0)
    parser.add_argument("--bootstrap-weight-norm", action="store_true")
    parser.add_argument("--no-bootstrap-init-from-ann", action="store_true")
    parser.add_argument("--conversion-threshold", type=float, default=0.2)
    parser.add_argument("--conversion-timesteps", type=int, default=3)
    parser.add_argument("--conversion-readout", default="mean")
    parser.add_argument("--conversion-output-activation", default="sigma")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    device = resolve_bootstrap_device(args.device)
    student_model, architecture = load_student_model_from_checkpoint(checkpoint_path, device=device)
    eval_states, eval_actions = load_teacher_dataset(dataset_path, max_samples=args.max_samples)
    train_max_samples = args.train_max_samples if args.train_max_samples > 0 else None
    train_states, train_actions = load_teacher_dataset(dataset_path, max_samples=train_max_samples)

    conversion_snn = SNNPolicy.from_student(
        student_model=student_model,
        conversion_config=SNNConversionConfig(
            threshold=args.conversion_threshold,
            timesteps=args.conversion_timesteps,
            device=str(device),
            readout=args.conversion_readout,
            output_activation=args.conversion_output_activation,
        ),
    )
    conversion_parity = evaluate_ann_snn_parity(
        student_model=student_model,
        snn_model=conversion_snn,
        states=eval_states,
        batch_size=args.batch_size,
        device=device,
    )
    conversion_teacher = evaluate_models_against_teacher(
        student_model=student_model,
        snn_model=conversion_snn,
        states=eval_states,
        teacher_actions=eval_actions,
        batch_size=args.batch_size,
        device=device,
        readout=args.conversion_readout,
    )

    input_strategies = parse_choice_list(args.input_strategies, SUPPORTED_BOOTSTRAP_INPUT_STRATEGIES, "input strategy")
    input_weights = parse_numeric_list(args.input_weights, float)
    input_biases = parse_numeric_list(args.input_biases, float)
    neuron_thresholds = parse_numeric_list(args.neuron_thresholds, float)
    current_decays = parse_numeric_list(args.current_decays, float)
    voltage_decays = parse_numeric_list(args.voltage_decays, float)
    timesteps = parse_numeric_list(args.timesteps, int)
    num_sample_iters = parse_numeric_list(args.num_sample_iters, int)
    sample_periods = parse_numeric_list(args.sample_periods, int)

    results = []
    total = (
        len(input_strategies)
        * len(input_weights)
        * len(input_biases)
        * len(neuron_thresholds)
        * len(current_decays)
        * len(voltage_decays)
        * len(timesteps)
        * len(num_sample_iters)
        * len(sample_periods)
    )
    trial_index = 0

    for input_strategy in input_strategies:
        for input_weight in input_weights:
            for input_bias in input_biases:
                for neuron_threshold in neuron_thresholds:
                    for current_decay in current_decays:
                        for voltage_decay in voltage_decays:
                            for timestep in timesteps:
                                for num_sample_iter in num_sample_iters:
                                    for sample_period in sample_periods:
                                        trial_index += 1
                                        trial_config = {
                                            "input_strategy": input_strategy,
                                            "input_weight": float(input_weight),
                                            "input_bias": float(input_bias),
                                            "neuron_threshold": float(neuron_threshold),
                                            "current_decay": float(current_decay),
                                            "voltage_decay": float(voltage_decay),
                                            "timesteps": int(timestep),
                                            "num_sample_iter": int(num_sample_iter),
                                            "sample_period": int(sample_period),
                                        }
                                        trial_dir = output_dir / format_trial_name(trial_config)
                                        bootstrap_config = BootstrapTrainingConfig(
                                            dataset_path=str(dataset_path),
                                            batch_size=args.bootstrap_batch_size,
                                            learning_rate=1e-4,
                                            epochs=args.epochs,
                                            hidden_dims=architecture["hidden_dims"],
                                            checkpoint_dir=str(trial_dir),
                                            timesteps=int(timestep),
                                            readout=args.readout,
                                            num_sample_iter=int(num_sample_iter),
                                            sample_period=int(sample_period),
                                            neuron_threshold=float(neuron_threshold),
                                            current_decay=float(current_decay),
                                            voltage_decay=float(voltage_decay),
                                            weight_scale=float(args.bootstrap_weight_scale),
                                            weight_norm=bool(args.bootstrap_weight_norm),
                                            input_strategy=input_strategy,
                                            input_weight=float(input_weight),
                                            input_bias=float(input_bias),
                                            initialize_from_ann=not args.no_bootstrap_init_from_ann,
                                        )
                                        trainer = BootstrapStudentTrainer(config=bootstrap_config, device=device)
                                        artifacts = trainer.train_on_arrays(
                                            states=train_states,
                                            actions=train_actions,
                                            initialize_from_ann=None if args.no_bootstrap_init_from_ann else student_model,
                                        )
                                        bootstrap_model = trainer.model
                                        bootstrap_parity = evaluate_ann_bootstrap_parity(
                                            student_model=student_model,
                                            bootstrap_model=bootstrap_model,
                                            states=eval_states,
                                            batch_size=args.batch_size,
                                            device=device,
                                        )
                                        bootstrap_teacher = evaluate_bootstrap_against_teacher(
                                            student_model=student_model,
                                            bootstrap_model=bootstrap_model,
                                            states=eval_states,
                                            teacher_actions=eval_actions,
                                            batch_size=args.batch_size,
                                            device=device,
                                        )
                                        bootstrap_activity = collect_bootstrap_activity_stats(
                                            bootstrap_model=bootstrap_model,
                                            states=eval_states[: min(len(eval_states), args.batch_size)],
                                            device=device,
                                        )
                                        beats_conversion_baseline = (
                                            bootstrap_parity["relative_l2_error"] < conversion_parity["relative_l2_error"]
                                            and bootstrap_teacher["snn_mean_percentage_error"] < conversion_teacher["snn_mean_percentage_error"]
                                        )
                                        result = {
                                            **trial_config,
                                            "trial_dir": str(trial_dir),
                                            "best_checkpoint_path": artifacts.best_checkpoint_path,
                                            "best_val_loss": float(artifacts.best_val_loss),
                                            "last_train_loss": float(artifacts.last_train_loss),
                                            "last_val_loss": float(artifacts.last_val_loss),
                                            "parity_relative_l2_error": float(bootstrap_parity["relative_l2_error"]),
                                            "parity_mae": float(bootstrap_parity["mae"]),
                                            "parity_output_scale_ratio": float(bootstrap_parity["output_scale_ratio"]),
                                            "readout_diagnostics": bootstrap_parity["readout_diagnostics"],
                                            "student_mean_percentage_error": float(bootstrap_teacher["student_mean_percentage_error"]),
                                            "student_median_percentage_error": float(bootstrap_teacher["student_median_percentage_error"]),
                                            "snn_mean_percentage_error": float(bootstrap_teacher["snn_mean_percentage_error"]),
                                            "snn_median_percentage_error": float(bootstrap_teacher["snn_median_percentage_error"]),
                                            "activity": bootstrap_activity,
                                            "network_silent": bool(bootstrap_activity["network_silent"]),
                                            "beats_conversion_baseline": bool(beats_conversion_baseline),
                                        }
                                        results.append(result)
                                        print(
                                            f"[{trial_index}/{total}] "
                                            f"strategy={input_strategy} iw={input_weight:.3f} ib={input_bias:.3f} "
                                            f"thr={neuron_threshold:.3f} id={current_decay:.3f} vd={voltage_decay:.3f} "
                                            f"T={timestep} ns={num_sample_iter} sp={sample_period} "
                                            f"rel_l2={result['parity_relative_l2_error']:.6f} "
                                            f"scale={result['parity_output_scale_ratio']:.6f} "
                                            f"snn_mean_pct={result['snn_mean_percentage_error']:.4f}% "
                                            f"silent={result['network_silent']} "
                                            f"beats_conversion={result['beats_conversion_baseline']}"
                                        )

    results.sort(
        key=lambda item: (
            item["network_silent"],
            not item["beats_conversion_baseline"],
            item["snn_mean_percentage_error"],
            item["parity_relative_l2_error"],
            abs(item["parity_output_scale_ratio"] - 1.0),
        )
    )

    best_result = results[0] if results else None
    best_beating_result = next((item for item in results if item["beats_conversion_baseline"]), None)

    report = {
        "student_checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        "device": str(device),
        "samples": int(len(eval_states)),
        "student_architecture": architecture,
        "conversion_baseline": {
            "threshold": float(args.conversion_threshold),
            "timesteps": int(args.conversion_timesteps),
            "readout": args.conversion_readout,
            "output_activation": args.conversion_output_activation,
            "parity": conversion_parity,
            "teacher": conversion_teacher,
        },
        "sweep": {
            "epochs": int(args.epochs),
            "readout": args.readout,
            "input_strategies": input_strategies,
            "input_weights": input_weights,
            "input_biases": input_biases,
            "neuron_thresholds": neuron_thresholds,
            "current_decays": current_decays,
            "voltage_decays": voltage_decays,
            "timesteps": timesteps,
            "num_sample_iters": num_sample_iters,
            "sample_periods": sample_periods,
            "weight_scale": float(args.bootstrap_weight_scale),
            "weight_norm": bool(args.bootstrap_weight_norm),
            "initialized_from_ann": not args.no_bootstrap_init_from_ann,
            "trial_count": len(results),
        },
        "best_result": best_result,
        "best_beating_result": best_beating_result,
        "results": results,
    }

    print("best_bootstrap_configuration:")
    if best_result is None:
        print("  none")
    else:
        print(f"  strategy: {best_result['input_strategy']}")
        print(f"  input_weight: {best_result['input_weight']}")
        print(f"  input_bias: {best_result['input_bias']}")
        print(f"  neuron_threshold: {best_result['neuron_threshold']}")
        print(f"  current_decay: {best_result['current_decay']}")
        print(f"  voltage_decay: {best_result['voltage_decay']}")
        print(f"  timesteps: {best_result['timesteps']}")
        print(f"  num_sample_iter: {best_result['num_sample_iter']}")
        print(f"  sample_period: {best_result['sample_period']}")
        print(f"  parity_relative_l2_error: {best_result['parity_relative_l2_error']:.6f}")
        print(f"  parity_output_scale_ratio: {best_result['parity_output_scale_ratio']:.6f}")
        print(f"  snn_mean_percentage_error: {best_result['snn_mean_percentage_error']:.4f}%")
        print(f"  network_silent: {best_result['network_silent']}")
        print(f"  beats_conversion_baseline: {best_result['beats_conversion_baseline']}")

    if best_beating_result is None:
        print("best_bootstrap_beating_conversion: none")
    else:
        print("best_bootstrap_beating_conversion:")
        print(f"  strategy: {best_beating_result['input_strategy']}")
        print(f"  input_weight: {best_beating_result['input_weight']}")
        print(f"  neuron_threshold: {best_beating_result['neuron_threshold']}")
        print(f"  current_decay: {best_beating_result['current_decay']}")
        print(f"  voltage_decay: {best_beating_result['voltage_decay']}")
        print(f"  timesteps: {best_beating_result['timesteps']}")
        print(f"  snn_mean_percentage_error: {best_beating_result['snn_mean_percentage_error']:.4f}%")
        print(f"  parity_relative_l2_error: {best_beating_result['parity_relative_l2_error']:.6f}")

    if args.output_json:
        output_json_path = Path(args.output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
