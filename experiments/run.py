import argparse
import subprocess
import sys
from pathlib import Path

try:
    from .presets import PRESETS, RECORD_ROBOTS, get_record_robot
except ImportError:
    from presets import PRESETS, RECORD_ROBOTS, get_record_robot


def format_value(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def build_command(preset_name, extra_args, force_record=False, record_robot_index=None):
    preset = dict(PRESETS[preset_name])
    if force_record:
        preset["algorithm.record"] = True
    if record_robot_index is not None:
        preset["algorithm.record_robot_index"] = int(record_robot_index)
        preset["environment.nr_envs"] = len(RECORD_ROBOTS)
        preset["environment.multi_render"] = True
        preset["environment.train_robot_types"] = RECORD_ROBOTS
    experiment_py = Path(__file__).resolve().parent / "experiment.py"
    command = [sys.executable, str(experiment_py)]
    for key, value in preset.items():
        command.append(f"--{key}={format_value(value)}")
    command.extend(extra_args)
    return command


def main():
    parser = argparse.ArgumentParser(description="Run named one_policy_to_run_them_all experiment presets.")
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()), required=True, help="Named experiment preset to run.")
    parser.add_argument("--print-only", action="store_true", help="Print the resolved command without executing it.")
    parser.add_argument("--record", action="store_true", help="Enable a short video recording for the run.")
    parser.add_argument("--record-robot", type=int, default=None, help="Robot index to record from the shared recording robot list.")
    parser.add_argument("--list-record-robots", action="store_true", help="Print the available recording robot indices and exit.")
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

    print(f"Running preset '{args.preset}':")
    print(printable_command)
    print(f"cwd={experiment_dir}")
    return subprocess.call(command, cwd=experiment_dir)


if __name__ == "__main__":
    raise SystemExit(main())
