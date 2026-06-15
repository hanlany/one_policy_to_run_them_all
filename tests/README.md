# Tests

Run the lightweight test suite from the repository root:

```bash
cd /app/one_policy_to_run_them_all
python -m pytest tests -q
```

Run only the infrastructure and entrypoint guardrail tests:

```bash
python -m pytest \
  tests/test_experiment_run.py \
  tests/test_packaging_smoke.py \
  tests/test_shell_wrappers.py \
  tests/test_student_checkpoint.py \
  -q
```

Useful smoke checks after packaging, preset, or entrypoint changes:

```bash
python -c "import one_policy_to_run_them_all; import one_policy_to_run_them_all.environments.multi_robot.robot_helper"
python experiments/run.py --preset test_default --print-only
python -m experiments.run --preset test_default --print-only
bash experiments/test.sh --print-only
python -m pip wheel . --no-deps --no-build-isolation -w /tmp/optra-wheel
```
