# Independent Codex Agent Brief: Reducing SNN MSE

Use this file as a future handoff reference for improving the pure SNN training path in `one_policy_to_run_them_all/snn/train_v2.py`. The immediate goal is to reduce validation MSE between the SNN rate output and teacher/student action targets without changing robot rollout semantics blindly.

Audit date: 2026-06-19

Scope: `one_policy_to_run_them_all/snn/train_v2.py`, `snn/train_v2_config_default.yaml`, `snn/train_v2_config_long.yaml`, and related SNN checkpoints/datasets.

## Current Baseline

Observed on a 5k validation slice:

- Initial SNN after ANN weight initialization: about `0.280` MSE.
- Existing `snn/Trained/network.pt`: about `0.0405` MSE.
- Existing `snn/Trained_long/network.pt`: about `0.0214` MSE.
- Source ANN/student checkpoint metadata:
  - `best_val_loss`: about `0.000178`.
  - `last_train_loss`: about `0.000127`.
  - `last_val_loss`: about `0.000232`.

Dataset scale:

- Dataset shape: `states (400000, 668)`, `actions (400000, 24)`.
- Action mean absolute value: about `0.310`.
- Action standard deviation: about `0.544`.
- Zero-action MSE baseline: about `0.296`.
- Per-action-dimension zero baseline is uneven, roughly `0.025` to `0.716`.

Interpretation:

- The SNN training is working: it improves from near the zero-action baseline to about `0.0214`.
- The remaining gap to the ANN is large, so the main issue is likely SNN dynamics, output rate resolution, action scaling, or conversion mismatch rather than total training collapse.

## Highest-Leverage Experiments

Run these before broad refactors.

### 1. Increase SNN Time Steps

Current default and long configs use `model.time_steps: 3`. The training objective uses:

```python
rate = torch.mean(output, dim=-1)
loss = F.mse_loss(rate, actions)
```

With only 3 time steps, the rate readout is coarse. Try `5`, `10`, and possibly `20` if runtime allows.

Example:

```bash
python snn/train_v2.py --config snn/train_v2_config_long.yaml --set model.time_steps=10
```

Expected signal:

- If MSE drops cleanly, the main bottleneck is rate quantization/time resolution.
- If MSE does not drop, inspect activity and scaling next.

### 2. Sweep Input Drive And Neuron Dynamics Together

Current long config:

- `model.input_weight: 2.0`
- `model.neuron.threshold: 0.5`
- `model.neuron.current_decay: 0.3`
- `model.neuron.voltage_decay: 0.02`

Recommended sweep grid:

- `input_weight`: `1.0`, `2.0`, `4.0`
- `threshold`: `0.2`, `0.5`, `1.0`
- `current_decay`: `0.1`, `0.3`, `0.5`
- `voltage_decay`: `0.01`, `0.02`, `0.05`

Watch the printed `output_abs_mean`.

- If `output_abs_mean` is much lower than action mean abs `~0.31`, the SNN is under-active.
- If `output_abs_mean` is high but MSE remains high, the problem is likely timing/readout shape or per-dimension scaling.

### 3. Normalize Or Reweight Action Targets

Plain MSE is dominated by high-variance action dimensions. The action dimensions have very uneven target energy, with per-dimension zero baselines from about `0.025` to `0.716`.

Preferred experiment:

- Compute action mean/std on the training split.
- Train the SNN against normalized actions.
- Denormalize output for reporting/export/rollout compatibility.

Lighter alternative:

```python
loss = ((rate - actions) ** 2 / action_var.clamp_min(1e-6)).mean()
```

Keep reporting raw validation MSE even if optimizing normalized or weighted MSE.

Expected signal:

- If normalized/weighted training improves raw MSE, previous training was ignoring low-variance action dimensions.
- If normalized loss improves but raw MSE worsens, tune the weighting strength or use a combined raw-plus-normalized objective.

### 4. Test Output Calibration

The ANN checkpoint has biases, while the SNN state dict mostly contains input bias, neuron parameters, and synapse weights. Hidden-layer ANN biases may not be faithfully represented in the pure SNN path.

Practical test:

- Add a small learned output affine calibration during training:

```python
calibrated_rate = rate * output_scale + output_bias
loss = F.mse_loss(calibrated_rate, actions)
```

Expected signal:

- If calibration sharply lowers MSE, the SNN dynamics may be preserving shape but missing scale/offset.
- If calibration does little, focus on earlier spiking layers and neuron dynamics.

### 5. Use A Two-Stage Training Schedule

The long run already improved substantially over the shorter checkpoint. A staged schedule is likely useful:

- Stage 1: larger learning rate, e.g. `2e-3`, with higher `time_steps`.
- Stage 2: fine-tune from the best checkpoint with `2e-4` or `5e-5`.

Keep full validation checks frequent enough to avoid selecting a lucky validation subset checkpoint.

## Diagnostics To Add

Before changing too much, add logging for:

- Raw train and validation MSE.
- Normalized or weighted training loss if used.
- Mean and median percentage error:

```python
percentage_error = 100.0 * torch.linalg.norm(predictions - targets, dim=1) / torch.linalg.norm(targets, dim=1).clamp_min(1e-8)
```

- Per-action-dimension MSE.
- Output mean, output std, and output absolute mean.
- Target mean, target std, and target absolute mean.

These make it easier to distinguish under-activity, scale mismatch, and per-dimension imbalance.

## Suggested First Implementation Batch

1. Add helper functions for action statistics, percentage error, and per-dimension MSE.
2. Add optional config flags for:
   - action normalization,
   - weighted MSE,
   - output calibration,
   - loading an existing SNN checkpoint for fine-tuning.
3. Keep defaults equivalent to the current behavior.
4. Run a short sanity job on a subset before the full run.

Example short run:

```bash
python snn/train_v2.py \
  --config snn/train_v2_config_default.yaml \
  --set training.epochs=2 \
  --set training.max_train_samples=4096 \
  --set training.max_val_samples=2048 \
  --set runtime.export_hdf5=false \
  --set runtime.save_plot=false
```

## Stop Conditions

Stop and ask the user before:

- Deleting or overwriting important checkpoints without creating a new output directory.
- Changing rollout-time action scaling or robot policy semantics.
- Replacing the Lava bootstrap blocks with a different SNN implementation.
- Introducing new dependencies.

## Practical Ranking

Most promising order:

1. `model.time_steps=10`.
2. Action normalization or weighted MSE.
3. Joint sweep of `input_weight`, `threshold`, `current_decay`, and `voltage_decay`.
4. Output affine calibration.
5. Two-stage fine-tuning from the best SNN checkpoint.

The main working hypothesis is that the current `0.0214` MSE is limited by coarse SNN rate readout and target scaling imbalance, not by absence of learnable signal.
