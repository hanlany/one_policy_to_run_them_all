#!/usr/bin/env python3
"""Run the SNN v1 point-1/point-2 sweep and long best-parameter training.

This orchestrates `snn/train_v2.py` without modifying it:

1. Short sweep over time steps, input drive, and neuron dynamics.
2. Rank successful trials by best full-validation SNN MSE.
3. Launch one long training run using the best short-sweep parameters.

Every training process receives a labeled output directory and labeled model
filenames. The runner writes JSON summaries so the sweep can be monitored and
resumed after interruption.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SNN_DIR = REPO_ROOT / "snn"
DEFAULT_EXPERIMENT_ROOT = SNN_DIR / "experiments" / "snn_v1_points_1_2"
DEFAULT_CONFIG = SNN_DIR / "train_v2_config_long.yaml"
DEFAULT_TRAIN_SCRIPT = SNN_DIR / "train_v2.py"

METRIC_LINE_RE = re.compile(
    r"\[Epoch\s+(?P<epoch>\d+)/(?P<epochs>\d+)\]\s+"
    r"train_snn_mse=(?P<train>[^\s]+)\s+"
    r"val_subset_snn_mse=(?P<val_subset>[^\s]+)\s+"
    r"full_val_snn_mse=(?P<full_val>[^\s]+)\s+"
    r"best_snn_val_mse=(?P<best>[^\s]+).*?"
    r"lr=(?P<lr>[^\s]+)\s+"
    r"output_abs_mean=(?P<output_abs_mean>[^\s]+)"
)

DIRECT_MSE_KEYS = (
    "best_full_val_snn_mse",
    "best_snn_val_mse",
    "best_snn_val_loss",
    "best_val_mse",
    "best_val_loss",
)
HISTORY_MSE_KEYS = (
    "full_val_snn_loss",
    "full_val_snn_mse",
    "val_snn_loss",
    "val_snn_mse",
)
RESULT_JSON_NAMES = (
    "snn_training_history.json",
    "results.json",
    "training_results.json",
    "training_summary.json",
    "metrics.json",
    "summary.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_csv(raw: str, cast: type) -> list[Any]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError(f"Expected at least one comma-separated value, got {raw!r}")
    return [cast(value) for value in values]


def format_value(value: int | float | str) -> str:
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return text.replace("-", "m").replace(".", "p").replace("/", "_")


def trial_label(prefix: str, params: dict[str, int | float]) -> str:
    return (
        f"{prefix}"
        f"_ts{format_value(params['time_steps'])}"
        f"_iw{format_value(params['input_weight'])}"
        f"_thr{format_value(params['threshold'])}"
        f"_cd{format_value(params['current_decay'])}"
        f"_vd{format_value(params['voltage_decay'])}"
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def parse_float(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
    elif isinstance(raw, str) and raw.lower() != "n/a":
        try:
            value = float(raw)
        except ValueError:
            return None
    else:
        return None
    if value == float("inf") or value == float("-inf"):
        return None
    if value != value:
        return None
    return value


def find_best_mse_in_data(data: Any) -> tuple[float | None, str | None]:
    if isinstance(data, dict):
        for key in DIRECT_MSE_KEYS:
            value = parse_float(data.get(key))
            if value is not None:
                return value, key

        history = data.get("history")
        if isinstance(history, dict):
            for key in HISTORY_MSE_KEYS:
                values = [parse_float(item) for item in history.get(key, [])]
                values = [item for item in values if item is not None]
                if values:
                    return min(values), f"history.{key}"

        for key in HISTORY_MSE_KEYS:
            values = [parse_float(item) for item in data.get(key, [])] if isinstance(data.get(key), list) else []
            values = [item for item in values if item is not None]
            if values:
                return min(values), key

        for key, value in data.items():
            nested_value, nested_key = find_best_mse_in_data(value)
            if nested_value is not None and nested_key is not None:
                return nested_value, f"{key}.{nested_key}"

    if isinstance(data, list):
        best_value = None
        best_key = None
        for index, item in enumerate(data):
            nested_value, nested_key = find_best_mse_in_data(item)
            if nested_value is not None and (best_value is None or nested_value < best_value):
                best_value = nested_value
                best_key = f"{index}.{nested_key}"
        return best_value, best_key

    return None, None


def discover_result_metric(output_dir: Path) -> dict[str, Any] | None:
    candidates = [output_dir / filename for filename in RESULT_JSON_NAMES]
    candidates.extend(sorted(output_dir.glob("*history*.json")))
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        data = load_json(path)
        if data is None:
            continue
        mse, key = find_best_mse_in_data(data)
        if mse is not None:
            return {
                "best_full_val_snn_mse": mse,
                "metric_source": str(path),
                "metric_key": key,
            }
    return None


def parse_metric_line(line: str) -> dict[str, Any] | None:
    match = METRIC_LINE_RE.search(line)
    if match is None:
        return None
    data = match.groupdict()
    return {
        "epoch": int(data["epoch"]),
        "epochs": int(data["epochs"]),
        "train_snn_mse": parse_float(data["train"]),
        "val_subset_snn_mse": parse_float(data["val_subset"]),
        "full_val_snn_mse": parse_float(data["full_val"]),
        "best_snn_val_mse": parse_float(data["best"]),
        "learning_rate": parse_float(data["lr"]),
        "output_abs_mean": parse_float(data["output_abs_mean"]),
    }


def build_command(
    *,
    python: str,
    train_script: Path,
    config: Path,
    overrides: list[str],
) -> list[str]:
    command = [python, str(train_script), "--config", str(config)]
    for override in overrides:
        command.extend(["--set", override])
    return command


def common_overrides(
    *,
    params: dict[str, int | float],
    output_dir: Path,
    label: str,
    epochs: int,
    full_val_interval: int,
    export_hdf5: bool,
    save_plot: bool,
    device: str | None,
) -> list[str]:
    overrides = [
        f"paths.output_dir={output_dir}",
        f"paths.checkpoint_name={label}_network.pt",
        f"paths.export_name={label}_network.net",
        f"paths.plot_name={label}_training_curves.png",
        f"paths.history_name={label}_training_history.json",
        f"model.time_steps={params['time_steps']}",
        f"model.input_weight={params['input_weight']}",
        f"model.neuron.threshold={params['threshold']}",
        f"model.neuron.current_decay={params['current_decay']}",
        f"model.neuron.voltage_decay={params['voltage_decay']}",
        f"training.epochs={epochs}",
        f"training.full_val_interval={full_val_interval}",
        f"runtime.export_hdf5={str(export_hdf5).lower()}",
        f"runtime.save_plot={str(save_plot).lower()}",
    ]
    if device:
        overrides.append(f"runtime.device={device}")
    return overrides


def append_optional_overrides(overrides: list[str], optional: dict[str, Any]) -> list[str]:
    for key, value in optional.items():
        if value is not None:
            overrides.append(f"{key}={value}")
    return overrides


def run_training(
    *,
    phase: str,
    label: str,
    params: dict[str, int | float],
    output_dir: Path,
    log_path: Path,
    command: list[str],
    resume: bool,
) -> dict[str, Any]:
    result_path = output_dir / "trial_result.json"
    existing = load_json(result_path) if resume and result_path.exists() else None
    if existing and existing.get("exit_code") == 0 and existing.get("best_full_val_snn_mse") is not None:
        print(f"[resume] {phase} {label}: using existing result {result_path}", flush=True)
        existing["resumed"] = True
        return existing

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = utc_now()

    metadata = {
        "phase": phase,
        "label": label,
        "params": params,
        "output_dir": output_dir,
        "log_path": log_path,
        "command": command,
        "command_pretty": " ".join(shlex.quote(part) for part in command),
        "started_at": start_time,
    }
    write_json(output_dir / "command.json", metadata)

    print(f"[start] {phase} {label}", flush=True)
    print(f"[log] {log_path}", flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MPLBACKEND", "Agg")

    epoch_metrics: list[dict[str, Any]] = []
    best_full_val_mse = None
    best_epoch = None
    last_full_val_mse = None
    process_start = time.monotonic()

    with log_path.open("w", buffering=1) as log_file:
        log_file.write(f"# started_at: {start_time}\n")
        log_file.write(f"# cwd: {REPO_ROOT}\n")
        log_file.write("# command: " + " ".join(shlex.quote(part) for part in command) + "\n\n")
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            print(line, end="", flush=True)
            metrics = parse_metric_line(line)
            if metrics is None:
                continue
            epoch_metrics.append(metrics)
            full_val = metrics["full_val_snn_mse"]
            if full_val is not None:
                last_full_val_mse = full_val
                if best_full_val_mse is None or full_val < best_full_val_mse:
                    best_full_val_mse = full_val
                    best_epoch = metrics["epoch"]
        exit_code = process.wait()

    result_metric = discover_result_metric(output_dir)
    metric_source = "stdout"
    metric_key = "full_val_snn_mse"
    if result_metric is not None:
        best_full_val_mse = result_metric["best_full_val_snn_mse"]
        metric_source = result_metric["metric_source"]
        metric_key = result_metric["metric_key"]

    finished_at = utc_now()
    result = {
        **metadata,
        "finished_at": finished_at,
        "duration_seconds": round(time.monotonic() - process_start, 3),
        "exit_code": exit_code,
        "best_full_val_snn_mse": best_full_val_mse,
        "best_epoch": best_epoch,
        "last_full_val_snn_mse": last_full_val_mse,
        "metric_source": metric_source,
        "metric_key": metric_key,
        "checkpoint_path": output_dir / f"{label}_network.pt",
        "export_path": output_dir / f"{label}_network.net",
        "plot_path": output_dir / f"{label}_training_curves.png",
        "epoch_metrics": epoch_metrics,
    }
    write_json(result_path, result)

    if exit_code != 0:
        print(f"[failed] {phase} {label}: exit_code={exit_code}", flush=True)
    elif best_full_val_mse is None:
        print(f"[warning] {phase} {label}: no full-validation MSE found", flush=True)
    else:
        print(
            f"[done] {phase} {label}: best_full_val_snn_mse={best_full_val_mse:.9f}",
            flush=True,
        )
    return result


def build_short_trials(args: argparse.Namespace) -> list[dict[str, int | float]]:
    trials = [
        {
            "time_steps": int(time_steps),
            "input_weight": float(input_weight),
            "threshold": float(threshold),
            "current_decay": float(current_decay),
            "voltage_decay": float(voltage_decay),
        }
        for time_steps, input_weight, threshold, current_decay, voltage_decay in itertools.product(
            args.time_steps,
            args.input_weights,
            args.thresholds,
            args.current_decays,
            args.voltage_decays,
        )
    ]
    if args.start_index > 1:
        trials = trials[args.start_index - 1 :]
    if args.max_trials is not None:
        trials = trials[: args.max_trials]
    return trials


def rank_successful_short_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [
            result
            for result in results
            if result.get("exit_code") == 0 and result.get("best_full_val_snn_mse") is not None
        ],
        key=lambda result: float(result["best_full_val_snn_mse"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Durable SNN short sweep for improvement_snn_v1 points 1 and 2, followed by a long best-params run."
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable used to invoke snn/train_v2.py.")
    parser.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SCRIPT, help="Path to snn/train_v2.py.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Base train_v2 YAML config.")
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT, help="Parent directory for run IDs.")
    parser.add_argument("--run-root", type=Path, default=None, help="Concrete run directory. Defaults to experiment-root/run-id.")
    parser.add_argument("--run-id", default=None, help="Run ID used when --run-root is omitted.")
    parser.add_argument("--device", default=None, help="Optional runtime.device override, e.g. cuda, cuda:0, or cpu.")

    parser.add_argument("--short-epochs", type=int, default=40, help="Epochs for each short sweep trial.")
    parser.add_argument("--long-epochs", type=int, default=300, help="Epochs for the long best-params run.")
    parser.add_argument("--short-full-val-interval", type=int, default=10, help="Full-validation cadence for short trials.")
    parser.add_argument("--long-full-val-interval", type=int, default=10, help="Full-validation cadence for long training.")
    parser.add_argument("--short-export-hdf5", action="store_true", help="Export HDF5 files for short sweep trials.")
    parser.add_argument("--short-save-plot", action="store_true", help="Save training plots for short sweep trials.")
    parser.add_argument("--skip-long", action="store_true", help="Only run the short sweep and ranking.")
    parser.add_argument("--no-resume", action="store_true", help="Re-run trials even if trial_result.json exists.")
    parser.add_argument("--dry-run", action="store_true", help="Write the trial plan but do not start training.")

    parser.add_argument("--time-steps", type=lambda raw: parse_csv(raw, int), default=parse_csv("5,10,20", int))
    parser.add_argument("--input-weights", type=lambda raw: parse_csv(raw, float), default=parse_csv("1.0,2.0,4.0", float))
    parser.add_argument("--thresholds", type=lambda raw: parse_csv(raw, float), default=parse_csv("0.2,0.5,1.0", float))
    parser.add_argument("--current-decays", type=lambda raw: parse_csv(raw, float), default=parse_csv("0.1,0.3,0.5", float))
    parser.add_argument("--voltage-decays", type=lambda raw: parse_csv(raw, float), default=parse_csv("0.01,0.02,0.05", float))

    parser.add_argument("--start-index", type=int, default=1, help="1-based short-trial start index for manual partitioning.")
    parser.add_argument("--max-trials", type=int, default=None, help="Optional cap on short trials for smoke runs or partitioning.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional training sample cap passed to train_v2.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional validation sample cap passed to train_v2.")
    parser.add_argument("--val-eval-samples", type=int, default=None, help="Optional per-epoch validation subset size.")
    parser.add_argument(
        "--extra-short-set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra train_v2 --set override applied to each short trial.",
    )
    parser.add_argument(
        "--extra-long-set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra train_v2 --set override applied to the long run.",
    )
    args = parser.parse_args()

    if args.short_epochs < 1:
        parser.error("--short-epochs must be positive")
    if args.long_epochs < 1:
        parser.error("--long-epochs must be positive")
    if args.short_full_val_interval < 1 or args.long_full_val_interval < 1:
        parser.error("full-validation intervals must be positive")
    if args.start_index < 1:
        parser.error("--start-index is 1-based and must be at least 1")

    return args


def main() -> None:
    args = parse_args()
    run_id = args.run_id or default_run_id()
    run_root = args.run_root.resolve() if args.run_root else (args.experiment_root / run_id).resolve()
    logs_dir = run_root / "logs"
    short_root = run_root / "short_sweep"
    long_root = run_root / "long_best"
    resume = not args.no_resume
    optional_overrides = {
        "training.max_train_samples": args.max_train_samples,
        "training.max_val_samples": args.max_val_samples,
        "training.val_eval_samples": args.val_eval_samples,
    }

    short_trials = build_short_trials(args)
    plan = {
        "run_id": run_id,
        "run_root": run_root,
        "created_at": utc_now(),
        "config": args.config.resolve(),
        "train_script": args.train_script.resolve(),
        "short_epochs": args.short_epochs,
        "long_epochs": args.long_epochs,
        "short_full_val_interval": args.short_full_val_interval,
        "long_full_val_interval": args.long_full_val_interval,
        "resume": resume,
        "dry_run": args.dry_run,
        "grid": {
            "time_steps": args.time_steps,
            "input_weights": args.input_weights,
            "thresholds": args.thresholds,
            "current_decays": args.current_decays,
            "voltage_decays": args.voltage_decays,
        },
        "trial_count": len(short_trials),
        "trials": [
            {
                "index": index,
                "label": trial_label("short", params),
                "params": params,
            }
            for index, params in enumerate(short_trials, start=args.start_index)
        ],
    }
    write_json(run_root / "run_plan.json", plan)
    write_json(
        run_root / "status.json",
        {
            "updated_at": utc_now(),
            "status": "running",
            "run_root": run_root,
            "short_trials": len(short_trials),
            "short_epochs": args.short_epochs,
            "long_epochs": args.long_epochs,
        },
    )
    print(f"[plan] run_root={run_root}", flush=True)
    print(f"[plan] short_trials={len(short_trials)} short_epochs={args.short_epochs}", flush=True)
    print(f"[plan] long_epochs={args.long_epochs} skip_long={args.skip_long}", flush=True)

    if args.dry_run:
        print(f"[dry-run] wrote {run_root / 'run_plan.json'}", flush=True)
        return

    results: list[dict[str, Any]] = []
    for offset, params in enumerate(short_trials):
        index = args.start_index + offset
        label = trial_label("short", params)
        trial_dir = short_root / f"{index:03d}_{label}"
        log_path = logs_dir / f"{index:03d}_{label}.log"
        overrides = common_overrides(
            params=params,
            output_dir=trial_dir,
            label=label,
            epochs=args.short_epochs,
            full_val_interval=args.short_full_val_interval,
            export_hdf5=args.short_export_hdf5,
            save_plot=args.short_save_plot,
            device=args.device,
        )
        append_optional_overrides(overrides, optional_overrides)
        overrides.extend(args.extra_short_set)
        command = build_command(
            python=args.python,
            train_script=args.train_script.resolve(),
            config=args.config.resolve(),
            overrides=overrides,
        )
        result = run_training(
            phase="short_sweep",
            label=label,
            params=params,
            output_dir=trial_dir,
            log_path=log_path,
            command=command,
            resume=resume,
        )
        result["trial_index"] = index
        results.append(result)
        ranked_so_far = rank_successful_short_results(results)
        write_json(
            run_root / "short_sweep_results.json",
            {
                "updated_at": utc_now(),
                "run_root": run_root,
                "total_planned_trials": len(short_trials),
                "completed_trials": len(results),
                "successful_trials": len(ranked_so_far),
                "best_result": ranked_so_far[0] if ranked_so_far else None,
                "results": results,
            },
        )

    ranked = rank_successful_short_results(results)
    if not ranked:
        write_json(
            run_root / "status.json",
            {
                "updated_at": utc_now(),
                "status": "failed",
                "reason": "No short sweep trial completed with a full-validation MSE.",
                "run_root": run_root,
            },
        )
        raise SystemExit("No successful short sweep results with full-validation MSE.")

    best = ranked[0]
    best_params = best["params"]
    write_json(
        run_root / "best_params.json",
        {
            "selected_at": utc_now(),
            "selection_metric": "best_full_val_snn_mse",
            "best_full_val_snn_mse": best["best_full_val_snn_mse"],
            "best_trial_label": best["label"],
            "best_trial_dir": best["output_dir"],
            "best_trial_log": best["log_path"],
            "params": best_params,
            "ranked_results": ranked,
        },
    )
    print(
        f"[best] {best['label']} best_full_val_snn_mse={best['best_full_val_snn_mse']:.9f}",
        flush=True,
    )

    long_result = None
    if not args.skip_long:
        long_label = trial_label(f"long_e{args.long_epochs}", best_params)
        long_dir = long_root / long_label
        log_path = logs_dir / f"{long_label}.log"
        overrides = common_overrides(
            params=best_params,
            output_dir=long_dir,
            label=long_label,
            epochs=args.long_epochs,
            full_val_interval=args.long_full_val_interval,
            export_hdf5=True,
            save_plot=True,
            device=args.device,
        )
        append_optional_overrides(overrides, optional_overrides)
        overrides.extend(args.extra_long_set)
        command = build_command(
            python=args.python,
            train_script=args.train_script.resolve(),
            config=args.config.resolve(),
            overrides=overrides,
        )
        long_result = run_training(
            phase="long_best",
            label=long_label,
            params=best_params,
            output_dir=long_dir,
            log_path=log_path,
            command=command,
            resume=resume,
        )
        write_json(run_root / "long_best_result.json", long_result)

    write_json(
        run_root / "status.json",
        {
            "updated_at": utc_now(),
            "status": "complete",
            "run_root": run_root,
            "best_short_result": best,
            "long_result": long_result,
        },
    )
    print(f"[complete] run_root={run_root}", flush=True)


if __name__ == "__main__":
    main()
