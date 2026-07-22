# Offline Orchestrator Procedure

Use this note as a reusable reference for launching long-running sweeps or training jobs that should continue after VS Code or the Codex session closes. The concrete example in this repository is the SNN points 1+2 sweep, but the same pattern applies to other offline experiment runners.

Audit date: 2026-07-08

Scope: `/app/one_policy_to_run_them_all`, especially `dev/codex/launch_snn_points_1_2_sweep.sh` and `dev/codex/run_snn_points_1_2_sweep.py`.

## Core Pattern

An offline experiment should have two layers:

1. A small launcher shell script that creates a run directory, detaches the orchestrator, writes a PID file, and prints monitor paths.
2. A Python orchestrator that owns the experiment plan, trial labels, subprocess execution, result parsing, status JSON, resume behavior, and follow-up runs.

The launcher should use `setsid nohup`, not plain `nohup`, in this environment. Plain `nohup` returned a PID but the child exited immediately when the launcher shell ended. `setsid nohup` reparented the process to PID 1 and survived the shell/session boundary.

Current launcher pattern:

```bash
setsid nohup python3 dev/codex/run_snn_points_1_2_sweep.py \
  --run-root "${RUN_ROOT}" \
  --run-id "${RUN_ID}" \
  "$@" \
  > "${ORCH_LOG}" 2>&1 < /dev/null &

PID="$!"
printf '%s\n' "${PID}" > "${RUN_ROOT}/orchestrator.pid"
```

## Launch Procedure

Start from the repository root:

```bash
cd /app/one_policy_to_run_them_all
```

Run a dry run first. This writes the run plan and confirms the trial count without starting training:

```bash
python dev/codex/run_snn_points_1_2_sweep.py \
  --run-id codex_dryrun \
  --dry-run \
  --short-epochs 40 \
  --long-epochs 300 \
  --time-steps 5,10,20 \
  --input-weights 1.0,2.0,4.0 \
  --thresholds 0.2,0.5 \
  --current-decays 0.1,0.3 \
  --voltage-decays 0.02 \
  --max-train-samples 50000 \
  --max-val-samples 10000 \
  --val-eval-samples 10000
```

Launch the detached run:

```bash
bash dev/codex/launch_snn_points_1_2_sweep.sh \
  --device cuda \
  --short-epochs 40 \
  --long-epochs 300 \
  --short-full-val-interval 10 \
  --long-full-val-interval 10 \
  --short-save-plot \
  --time-steps 5,10,20 \
  --input-weights 1.0,2.0,4.0 \
  --thresholds 0.2,0.5 \
  --current-decays 0.1,0.3 \
  --voltage-decays 0.02 \
  --extra-short-set training.max_train_samples=50000 \
  --extra-short-set training.max_val_samples=10000 \
  --extra-short-set training.val_eval_samples=10000
```

The launcher prints:

- process PID,
- concrete run root,
- orchestrator log path,
- status JSON path.

The default run root shape is:

```text
snn/experiments/snn_v1_points_1_2/<UTC_RUN_ID>/
```

## Monitoring

Check that the orchestrator is alive and detached:

```bash
ps -p <PID> -o pid,ppid,sid,stat,etime,cmd
```

Expected detached shape:

```text
PPID = 1
SID  = <PID>
```

Follow the orchestrator log:

```bash
tail -f snn/experiments/snn_v1_points_1_2/<RUN_ID>/logs/orchestrator_<RUN_ID>.log
```

Read coarse status:

```bash
cat snn/experiments/snn_v1_points_1_2/<RUN_ID>/status.json
```

Read short-sweep progress:

```bash
cat snn/experiments/snn_v1_points_1_2/<RUN_ID>/short_sweep_results.json
```

After the short sweep finishes, inspect selected parameters:

```bash
cat snn/experiments/snn_v1_points_1_2/<RUN_ID>/best_params.json
```

After the long run finishes, inspect:

```bash
cat snn/experiments/snn_v1_points_1_2/<RUN_ID>/long_best_result.json
```

## Output Layout

Each run root should contain:

- `orchestrator.pid`: PID printed by the launcher.
- `run_plan.json`: complete grid, trial labels, config, and train script path.
- `status.json`: coarse state such as `running`, `failed`, or `complete`.
- `logs/orchestrator_<RUN_ID>.log`: top-level orchestrator output.
- `logs/<TRIAL_LABEL>.log`: per-trial training output.
- `short_sweep/<INDEX>_<TRIAL_LABEL>/`: per-trial model, plot, command, history, and result files.
- `short_sweep_results.json`: accumulating short-trial summary.
- `best_params.json`: ranked successful trials and selected best parameter set.
- `long_best/<LONG_LABEL>/`: long run artifacts for the selected parameters.
- `long_best_result.json`: long-run summary.

Use labels that encode the material hyperparameters, for example:

```text
001_short_ts5_iw1_thr0p2_cd0p1_vd0p02
long_e300_ts10_iw2_thr0p5_cd0p3_vd0p02
```

This keeps logs, checkpoints, plots, and JSON summaries readable without reopening `command.json`.

## Orchestrator Skills

Keep these implementation details when writing a new offline runner:

- Write a `run_plan.json` before starting compute.
- Write `status.json` immediately after planning so the user has a stable monitor target.
- Stream each child process line-by-line to both the per-trial log and the orchestrator stdout.
- Set `PYTHONUNBUFFERED=1` so logs update in real time.
- Set `MPLBACKEND=Agg` for headless plot generation.
- Write `command.json` for every trial with the exact command and parameter payload.
- Use deterministic, human-readable labels for output folders and filenames.
- Save a `trial_result.json` after each child exits, including exit code, duration, best metric, paths, and parsed epoch metrics.
- Resume completed trials by default if `trial_result.json` has `exit_code == 0` and a valid best metric.
- Provide `--no-resume` for forced reruns.
- Provide `--dry-run` for plan validation.
- Provide `--start-index` and `--max-trials` for manual partitioning or smoke tests.
- Keep short-run sample caps separate from the final long run so the sweep finishes in a reasonable time but final training can use full data.

## Preflight Checks

Before launching a large offline run:

```bash
python -m py_compile dev/codex/run_snn_points_1_2_sweep.py
bash -n dev/codex/launch_snn_points_1_2_sweep.sh
```

Run a tiny CPU smoke training job if the training script changed:

```bash
python snn/train_v2.py \
  --config snn/train_v2_config_default.yaml \
  --set paths.output_dir=experiments/snn_sweep_v1_smoke \
  --set training.epochs=1 \
  --set training.max_train_samples=256 \
  --set training.max_val_samples=128 \
  --set training.val_eval_samples=128 \
  --set training.train_batch_size=64 \
  --set training.val_batch_size=64 \
  --set training.full_val_interval=1 \
  --set runtime.device=cpu \
  --set runtime.export_hdf5=false \
  --set runtime.save_plot=true
```

If using CUDA, run a tiny CUDA smoke job before committing an overnight run:

```bash
python snn/train_v2.py \
  --config snn/train_v2_config_default.yaml \
  --set paths.output_dir=experiments/snn_sweep_v1_cuda_smoke \
  --set training.epochs=1 \
  --set training.max_train_samples=64 \
  --set training.max_val_samples=64 \
  --set training.val_eval_samples=64 \
  --set training.train_batch_size=32 \
  --set training.val_batch_size=32 \
  --set training.full_val_interval=1 \
  --set runtime.device=cuda \
  --set runtime.export_hdf5=false \
  --set runtime.save_plot=false
```

## Detach Verification

If a detached launch exits immediately, check:

```bash
cat <RUN_ROOT>/orchestrator.pid
ps -p <PID> -o pid,ppid,sid,stat,etime,cmd
ls -la <RUN_ROOT> <RUN_ROOT>/logs
tail -n 80 <RUN_ROOT>/logs/orchestrator_<RUN_ID>.log
```

If the log is empty and `ps` shows no process, test the detach mechanism:

```bash
setsid bash -c 'sleep 60' >/tmp/codex_detach_test.log 2>&1 < /dev/null & echo $!
ps -p <TEST_PID> -o pid,ppid,sid,stat,etime,cmd
```

If the test process has `PPID=1`, the session-detach mechanism works. Use `setsid nohup` in the launcher.

## Result Selection

Prefer structured metrics over parsing text logs. The current orchestrator checks:

- `snn_training_history.json`,
- other common result JSON names,
- `*history*.json` files,
- stdout metric lines as a fallback.

For `train_v2.py`, the useful training history keys are:

- `full_val_snn_loss`,
- `val_snn_loss`,
- `train_snn_loss`,
- `train_mean_percentage_error`,
- `train_median_percentage_error`,
- `val_mean_percentage_error`,
- `val_median_percentage_error`,
- `full_val_mean_percentage_error`,
- `full_val_median_percentage_error`.

Ranking should use the best full-validation MSE, not the best subset validation MSE, when available.

## Common Failure Modes

- Launcher prints a PID but no `status.json` appears: the child likely exited before the orchestrator wrote its plan. Check the orchestrator log and whether the launcher used `setsid nohup`.
- Orchestrator log is empty: the process may have been killed at session teardown or failed before Python emitted output.
- Short trials finish but no best params are selected: metric parsing failed or no trial produced a full-validation metric. Inspect per-trial `trial_result.json` and `snn_training_history.json`.
- CUDA is available but training fails: rerun the tiny CUDA smoke job. If it fails, launch the sweep with `--device cpu` or fix the CUDA/Lava path before a large run.
- Trial grid is too large: reduce the Cartesian product, add `--max-trials`, or use short-run sample caps.

## When It Is Safe To Close VS Code

It is safe to close VS Code after all of these are true:

- the launcher has printed a PID and run root,
- `ps -p <PID> -o pid,ppid,sid,stat,etime,cmd` shows the orchestrator,
- `PPID` is `1`,
- `status.json` exists and says `running`,
- the orchestrator log shows the first trial has started.

Example healthy status:

```text
PID      PPID  SID     STAT  CMD
266666   1     266666  Ss    python3 dev/codex/run_snn_points_1_2_sweep.py ...
```

At that point, monitoring can be resumed later from the files under the run root.
