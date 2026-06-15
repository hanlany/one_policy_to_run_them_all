# Tests

Run the lightweight test suite from the repository root:

```bash
cd /app/one_policy_to_run_them_all
python -m pytest tests -q
```

Run only the Phase 1 guardrail tests:

```bash
python -m pytest \
  tests/test_experiment_run.py \
  tests/test_packaging_smoke.py \
  tests/test_student_checkpoint.py \
  -q
```

Useful smoke checks after packaging or preset changes:

```bash
python -c "import one_policy_to_run_them_all; import one_policy_to_run_them_all.environments.multi_robot.robot_helper"
python experiments/run.py --preset test_default --print-only
python -m pip wheel . --no-deps --no-build-isolation -w /tmp/optra-wheel
```
