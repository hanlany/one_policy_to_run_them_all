# Goal-Mode Guideline: 24-Bit QuanTrick SNNv2

## Goal-mode contract

**Goal:** deliver a trained, rollout-compatible 24-bit (3x8-bit) QuanTrick SNNv2 and collect validated videos for every configured robot.

**Terminal success condition:** mark the goal complete only after every item in the final completion checklist is checked and its evidence is present on disk. Implementation, unit tests, or a successful training process alone are not completion.

**Non-negotiable gates:** preserve current artifacts; keep legacy 8-bit behavior compatible; do not record until the selected 24-bit model meets the validation-MSE gate; do not count pre-existing videos; do not silently relax a gate.

**Blocked-state rule:** if all specified fallback training runs finish and still fail the MSE gate, retain logs/artifacts, report the measured candidates and failure cause, and request a decision. Do not replace the candidate with the legacy model merely to finish recording.

## Goal condition

Implement PRECiSE-style 24-bit weight quantization as three 8-bit chunks in `/app/lava-dl`, integrate it with SNNv2 training and PyTorch rollout, train a quantized SNNv2 without regressing from the current best full-validation MSE, and record all 16 configured robots.

The goal is complete only when:

- Lava-DL and project tests pass.
- The winning checkpoint uses 24-bit, 3x8-bit decomposed weights during both training and rollout.
- Full-validation MSE is `<= 0.00592308546602726`.
- All predictions and weights are finite and no 24-bit saturation occurs.
- Sixteen new, readable MP4 files—one per `record_robots` entry—exist directly under `/app/one_policy_to_run_them_all/experiments/videos`.
- A manifest identifies the checkpoint, configuration, metrics, robot-to-video mapping, and video validation results.
- Existing checkpoints and videos are preserved and are not counted as new evidence.

True NetX/Loihi multi-synapse deployment is out of scope. Legacy 8-bit HDF5 export must remain unchanged; decomposed mode must reject HDF5 export clearly rather than produce an invalid single-weight representation.

## Ordered TODO checklist

Work through this list in order. Tick an item only after its stated evidence is available.

- [ ] **0. Establish baseline and workspace safety.** Record `git status` for both repositories, preserve unrelated dirty files, snapshot the existing video filenames, and copy the long-best metric/config into the new run manifest.
- [ ] **1. Implement Lava-DL decomposition primitives.** Add the public decomposition API, quantizer, validation, diagnostics, and legacy-compatible behavior described below.
- [ ] **2. Verify Lava-DL behavior.** Pass all focused decomposition/CUBA tests, then the full Lava-DL suite; save command output and versions in the run manifest.
- [ ] **3. Integrate SNNv2.** Wire the feature flag, diagnostics, checkpoints, HDF5 guard, rollout loader, PPO settings, recording preset, and all-robot wrapper.
- [ ] **4. Verify project integration.** Pass focused compatibility/rollout tests and the full project test suite; save results with the implementation artifacts.
- [ ] **5. Run training smoke test.** Complete the two-epoch smoke run with finite loss, gradients, outputs, and zero saturation.
- [ ] **6. Run matched candidate trials.** Complete all three 40-epoch candidates, rank decomposed candidates, and record the selection rationale.
- [ ] **7. Train and accept a final candidate.** Finish the selected 300-epoch run, or the defined fallback sequence, and satisfy the full-validation MSE gate.
- [ ] **8. Prove rollout parity.** Load the final checkpoint through the rollout path and satisfy the fixed-batch parity and finite-output checks.
- [ ] **9. Record every robot.** Produce exactly one new video for each of the 16 configured `record_robots` entries.
- [ ] **10. Validate and hand off.** Validate videos, create the final manifest/contact sheets, review completion evidence, and only then mark the goal complete.

## Implementation changes

### Lava-DL quantization API

Add a public utility under `lava.lib.dl.slayer.utils` with:

- `SignMode`: `MIXED`, `EXCITATORY`, and `INHIBITORY`.
- `WeightChunk`: chunk tensor, exponent, sign mode, and MSB metadata.
- `WeightDecomposer(target_bits=24, chunk_bits=8)`.
- `DecomposedWeightQuantizer`, a callable compatible with a synapse `pre_hook_fx`.

Required behavior:

- Validate that bit widths are positive and `target_bits` is divisible by `chunk_bits`.
- Support NumPy arrays for diagnostic use and PyTorch tensors for training.
- Use `torch.int64` internally so unsigned chunks remain portable across CPU and CUDA.
- With CUBA scale `64`, calculate `m = clamp(round(weight * 64))`, decompose `m`, reconstruct it, and return `m / 64`.
- For 24/8 mixed mode, reconstruct `m = c0 * 2^0 + c1 * 2^8 + c2 * 2^16`.
- In mixed mode, interpret lower chunks as unsigned and the most-significant chunk as signed.
- In excitatory mode, use unsigned chunks and reject or clamp negative inputs according to the documented saturation policy.
- In inhibitory mode, decompose magnitudes and negate each chunk's contribution.
- Use a straight-through estimator so the discrete reconstructed value is used in the forward pass while gradients pass to the original floating-point parameter.
- Preserve tensor shape, device, and floating dtype.
- Expose reconstruction error and saturation counts for diagnostics.
- Keep existing `quantize_8bit` behavior as the default and backward-compatible path.

Golden vectors must include `8388607 -> [255, 255, 127]`, `-8388608 -> [0, 0, -128]`, mixed-mode `-1`, zero, boundary overflows, and inhibitory values.

### SNNv2 training and checkpoint integration

Extend `snn/train_v2.py` and its configuration with:

```yaml
model:
  weight_quantization:
    mode: legacy_8bit
    target_bits: 24
    chunk_bits: 8
    sign_mode: mixed
    scope: all
```

Rules:

- Existing configs default to `legacy_8bit`.
- The new QuanTrick config explicitly selects `mode: decomposed`, `target_bits: 24`, and `chunk_bits: 8`.
- `scope` supports `all` and `first`; it applies only to Dense/Affine synaptic weights, not the scalar input transform.
- Install the decomposed quantizer as the selected layers' `pre_hook_fx` in ANN, sampler, and SNN modes.
- Reject unsupported modes, signs, scopes, or widths before training.
- Reject `runtime.export_hdf5: true` in decomposed mode with an actionable message.
- Log per-layer raw range, quantized range, reconstruction error, and saturation count at initialization, each full-validation epoch, and final checkpoint selection.
- Save self-describing checkpoint payloads containing `state_dict`, architecture, neuron settings, input encoding, and quantization metadata.
- Continue loading old raw state dictionaries and existing metadata checkpoints.
- Update final best-checkpoint reload logic to extract `state_dict` from either format.

Create `snn/train_v2_config_quantrick.yaml` from the long-best defaults:

- `time_steps: 20`
- `input_weight: 1.0`
- threshold `0.2`
- current decay `0.3`
- voltage decay `0.02`
- 300 epochs, batch size 1024, learning rate `2e-3`
- full validation every 10 epochs
- scheduler factor `0.7`, patience `10`, threshold `5e-5`, minimum LR `2e-5`
- decomposed 24/8 mixed quantization
- HDF5 export disabled
- unique output directories under `snn/experiments/snn_v2_quantrick/<UTC_RUN_ID>`.

### Rollout and recording integration

Propagate quantization metadata through `BootstrapTrainingConfig`, `BootstrapStudentPolicy`, checkpoint loading, PPO defaults, and experiment overrides. Metadata stored in a checkpoint takes precedence; explicit preset values provide compatibility for raw checkpoints.

Add:

- A `record_snnv2_quantrick` preset pointing to the winning checkpoint with matching `time_steps`, neuron dynamics, input encoding, and quantization settings.
- A `record_snnv2_quantrick_all_robots.sh` wrapper following the existing all-robot recording loop.
- Preset and wrapper tests proving the new checkpoint and 24/8 settings reach rollout construction.

Before recording, compare trainer and rollout-loader predictions on the same fixed batch. Require identical outputs within `atol=1e-6`, finite values, and confirmation that every selected synapse has the decomposed pre-hook.

## Verification and training procedure

### Feature tests

In `/app/lava-dl`, add focused tests for:

- All sign modes and golden vectors.
- Exact integer reconstruction for randomized 24-bit values.
- Quantization error bounded by `1 / (2 * 64)`.
- Saturation and invalid configuration handling.
- NumPy and PyTorch parity.
- CPU/CUDA shape, dtype, and device preservation.
- Straight-through gradients.
- Dense/Affine forward parity with a reference convolution using reconstructed weights.
- Unchanged legacy 8-bit outputs and HDF5 export.
- Explicit decomposed-HDF5 rejection.

Run the focused Lava-DL suite, relevant CUBA block tests, then the full Lava-DL test suite.

In `one_policy_to_run_them_all`, test:

- Config loading and CLI overrides.
- Old raw and new metadata checkpoint compatibility.
- Trainer/rollout output parity.
- A one-epoch CPU optimization smoke test.
- Invalid quantization and HDF5 combinations.
- Experiment preset and shell-wrapper wiring.
- The full project test suite.

### Training gates

Use deterministic split seeds `0` and `1` throughout.

1. Run a two-epoch/4096-sample smoke job. Require decreasing or finite loss, nonzero gradients, finite outputs, and zero saturation.
2. Run matched 40-epoch, 50k-train/10k-validation trials:
   - Legacy 8-bit control.
   - Decomposed 24/8 on the first weighted layer.
   - Decomposed 24/8 on all weighted layers.
3. Rank decomposed candidates by full-validation MSE. Break ties using median percentage error, then mean percentage error.
4. Train the winning decomposed scope for 300 epochs on the complete dataset and validation split.
5. Accept only a checkpoint with full-validation MSE `<= 0.00592308546602726`. Report mean and median percentage error against the existing `13.90697%` and `10.76049%` baselines, but do not make them hard gates.
6. If the full run regresses:
   - Fine-tune the existing long-best checkpoint under the winning decomposed mode for up to 100 epochs at `2e-4`.
   - If needed, repeat with the runner-up decomposition scope.
   - Never substitute the legacy checkpoint, relax the MSE gate, or proceed to recording while the gate fails.

Store commands, resolved config, environment versions, metrics, logs, plots, checkpoints, and candidate ranking in the run directory.

## Video validation and final evidence

After the accuracy gate passes:

1. Snapshot the existing MP4 list so older videos cannot satisfy completion.
2. Run the new all-robot wrapper for the 16 robots listed in `record_robots`.
3. Capture one new MP4 per robot directly in `experiments/videos`.
4. Validate each video with OpenCV's FFmpeg-enabled backend:
   - File opens successfully and is nonempty.
   - Width, height, FPS, and frame count are positive.
   - Duration is approximately the configured 10 seconds.
   - First, middle, and final frames decode and are not blank.
5. Build contact sheets from those frames and visually check that the intended robot is visible and rendering is not frozen or corrupted.
6. Require recording logs to contain no traceback, NaN/Inf action, checkpoint mismatch, or fallback-to-teacher message.
7. Write a manifest containing the run ID, checkpoint hash/path, resolved quantization configuration, accepted validation metrics, all 16 robot names and MP4 paths, frame metadata, and validation status.

Do not overwrite or delete existing Lava tutorial changes, checkpoints, experiment runs, or videos. Do not mark the goal complete after implementation, tests, or training alone; completion occurs only after all 16 validated videos and the manifest are present.

## Final completion checklist

- [ ] Lava-DL focused and full test suites passed, with saved logs.
- [ ] Project focused and full test suites passed, with saved logs.
- [ ] Final 24/8 checkpoint metadata, diagnostics, and reproducible training command are saved.
- [ ] Final full-validation MSE is `<= 0.00592308546602726`; output is finite and saturation count is zero.
- [ ] Trainer and rollout-loader predictions satisfy `atol=1e-6` parity on the fixed validation batch.
- [ ] Manifest records one newly generated, OpenCV-validated MP4 for each of the 16 configured robots.
- [ ] Contact sheets and recording logs show no rendering failure, traceback, NaN/Inf action, checkpoint mismatch, or teacher fallback.
- [ ] All items in the ordered TODO checklist are checked.
