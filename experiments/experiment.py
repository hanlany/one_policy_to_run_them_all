import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDENT_DIR = ROOT / "student"


def ensure_local_import_paths():
    for path in (ROOT, STUDENT_DIR):
        path_string = str(path)
        if path_string not in sys.path:
            sys.path.append(path_string)


ensure_local_import_paths()

from rl_x.runner.runner import Runner


if __name__ == "__main__":
    runner = Runner(implementation_package_names=["rl_x", "one_policy_to_run_them_all"])
    runner.run()
