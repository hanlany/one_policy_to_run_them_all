import argparse
import subprocess
import sys
from pathlib import Path

try:
    from .presets import PRESETS
except ImportError:
    from presets import PRESETS


def format_value(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def build_command(preset_name, extra_args):
    preset = PRESETS[preset_name]
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
    args, extra_args = parser.parse_known_args()

    command = build_command(args.preset, extra_args)
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
