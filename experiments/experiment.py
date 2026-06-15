import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDENT_DIR = ROOT / "student"

if str(STUDENT_DIR) not in sys.path:
    sys.path.append(str(STUDENT_DIR))

from rl_x.runner.runner import Runner


if __name__ == "__main__":
    runner = Runner(implementation_package_names=["rl_x", "one_policy_to_run_them_all"])
    runner.run()