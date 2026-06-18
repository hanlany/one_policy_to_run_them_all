import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from .presets import PRESETS, RECORD_ROBOTS, get_record_robot, validate_preset
except ImportError:
    from presets import PRESETS, RECORD_ROBOTS, get_record_robot, validate_preset

from one_policy_to_run_them_all.paths import artifact_path


def format_value(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def resolve_preset(preset_name, force_record=False, record_robot_index=None):
    preset = dict(PRESETS[preset_name])
    if force_record:
        preset["algorithm.record"] = True
        preset["environment.multi_render"] = False
        preset.setdefault("algorithm.record_robot_index", 0)
    if record_robot_index is not None:
        robot = get_record_robot(record_robot_index)
        preset["algorithm.record"] = True
        preset["algorithm.record_robot_index"] = 0
        preset["environment.nr_envs"] = 1
        preset["environment.multi_render"] = False
        preset["environment.train_robot_types"] = (robot,)
        preset["environment.eval_robot_types"] = RECORD_ROBOTS
    validate_preset(preset_name, preset)
    return preset


def build_command(preset_name, extra_args, force_record=False, record_robot_index=None):
    preset = resolve_preset(
        preset_name,
        force_record=force_record,
        record_robot_index=record_robot_index,
    )
    experiment_py = Path(__file__).resolve().parent / "experiment.py"
    command = [sys.executable, str(experiment_py)]
    for key, value in preset.items():
        command.append(f"--{key}={format_value(value)}")
    command.extend(extra_args)
    return command


def build_resolved_config(preset_name, extra_args, force_record=False, record_robot_index=None):
    preset = resolve_preset(
        preset_name,
        force_record=force_record,
        record_robot_index=record_robot_index,
    )
    return {
        "preset": preset_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "options": {key: format_value(value) for key, value in preset.items()},
        "extra_args": list(extra_args),
        "force_record": bool(force_record),
        "record_robot_index": record_robot_index,
    }


def write_resolved_config(resolved_config, output_path=None):
    if output_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = artifact_path("resolved_configs", f"{timestamp}_{resolved_config['preset']}.json")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as config_file:
        json.dump(resolved_config, config_file, indent=2, sort_keys=True)
        config_file.write("\n")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Run named one_policy_to_run_them_all experiment presets.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        required=True,
        help="Named experiment preset to run.",
    )
    parser.add_argument("--print-only", action="store_true", help="Print the resolved command without executing it.")
    parser.add_argument("--record", action="store_true", help="Enable a short video recording for the run.")
    parser.add_argument(
        "--record-robot",
        type=int,
        default=None,
        help="Robot index to record from the shared recording robot list.",
    )
    parser.add_argument(
        "--list-record-robots",
        action="store_true",
        help="Print the available recording robot indices and exit.",
    )
    parser.add_argument(
        "--resolved-config-out",
        default=None,
        help="Write the resolved preset config to this JSON path before executing the run.",
    )
    parser.add_argument(
        "--no-save-resolved-config",
        action="store_true",
        help="Skip writing the resolved config JSON for executed runs.",
    )
    args, extra_args = parser.parse_known_args()

    if args.list_record_robots:
        for index, robot in enumerate(RECORD_ROBOTS):
            print(f"{index}: {robot}")
        return 0

    command = build_command(
        args.preset,
        extra_args,
        force_record=args.record,
        record_robot_index=args.record_robot,
    )
    experiment_dir = Path(__file__).resolve().parent
    printable_command = " ".join(command)

    if args.print_only:
        print(printable_command)
        print(f"cwd={experiment_dir}")
        return 0

    if not args.no_save_resolved_config:
        resolved_config = build_resolved_config(
            args.preset,
            extra_args,
            force_record=args.record,
            record_robot_index=args.record_robot,
        )
        resolved_config_path = write_resolved_config(resolved_config, args.resolved_config_out)
        print(f"resolved_config={resolved_config_path}")

    print(f"Running preset '{args.preset}':")
    print(printable_command)
    print(f"cwd={experiment_dir}")
    return subprocess.call(command, cwd=experiment_dir)


if __name__ == "__main__":
    raise SystemExit(main())
