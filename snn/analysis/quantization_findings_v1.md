# SNNv2 Quantization Findings V1

Date: 2026-06-19

Scope: `snn/train_v2.py`, Lava CUBA bootstrap blocks, and `snn/student_model_latest.pth` used to initialize SNNv2 training.

## Summary

`train_v2.py` does not expose an explicit quantization config, but Lava CUBA blocks implicitly use an 8-bit quantized forward path for synapse weights. The SNN training therefore already sees quantized weights during forward passes, even though PyTorch stores float parameters.

Default Lava CUBA quantized weight range observed:

```text
[-4.0, 3.96875]
step = 2 / 64 = 0.03125
```

Because `train_v2.py` maps the first ANN layer into signed-split form with `torch.cat([W, -W], dim=1)`, a safe symmetric bound for the original first-layer ANN weights is approximately:

```text
[-3.96875, 3.96875]
```

## Findings

Raw ANN checkpoint: `snn/student_model_latest.pth`

Only the first layer has weights outside the quantized valid range.

```text
net.0.weight   min -23.389170  max 40.639568  out 530 / 684032  (0.07748176%)
net.2.weight   out 0
net.4.weight   out 0
net.6.weight   out 0
net.8.weight   out 0
net.10.weight  out 0

Total raw ANN out of range: 530 / 4,902,912 = 0.0108099024%
```

After signed-split mapping used for SNN initialization:

```text
net.0.weight mapped shape: (1024, 1336)
min -40.639568  max 40.639568
below -4.0: 527
above 3.96875: 540
out 1067 / 1,368,064 = 0.07799343%

Total mapped SNN-init out of range: 1067 / 5,586,944 = 0.0190980973%
```

Distribution summary from `student_model_latest_out_of_range_weights.csv`:

```text
raw_net0:
  n out of range: 530
  below: 263
  above: 267
  median clip distance: 1.677
  max clip distance: 36.671

mapped_signed_split_net0:
  n out of range: 1067
  below: 527
  above: 540
  median clip distance: 1.672
  max clip distance: 36.671
```

Interpretation: the fraction of offending weights is tiny, but a few first-layer weights are very far outside range and will be heavily clipped by Lava quantization.

## Generated Analysis Files

- `snn/analysis/student_model_latest_out_of_range_weights.csv`
- `snn/analysis/student_model_latest_out_of_range_weight_distribution.png`
- `snn/analysis/student_model_latest_mapped_first_layer_weight_distribution.png`

## Recommended Next Experiments

Start with first-layer-only interventions. Do not alter deeper layers; they are already in range.

1. Hard clamp at SNN initialization

```python
weight = weight.clamp(-3.96875, 3.96875)
```

Apply after the signed-split mapping in `initialize_from_ann_checkpoint()`.

2. Hard clamp after each optimizer step

```python
with torch.no_grad():
    model.blocks[2].synapse.weight.clamp_(-3.96875, 3.96875)
```

This keeps the first weighted SNN layer deployable throughout training.

3. Soft range penalty

```python
limit = 3.96875
w0 = model.blocks[2].synapse.weight
range_penalty = torch.relu(w0.abs() - limit).pow(2).mean()
loss = mse_loss + lambda_range * range_penalty
```

Suggested sweep: `lambda_range = 1e-4, 1e-3, 1e-2`.

4. Avoid full first-layer rescaling unless clipping hurts

Global rescaling would shrink the first layer by about `3.96875 / 40.639568 ~= 0.098`, which may be too destructive unless compensated with `input_weight`.

## Suggested Validation

Use the current long-best SNN hyperparameters for an apples-to-apples short run:

```text
time_steps: 20
input_weight: 1.0
threshold: 0.2
current_decay: 0.3
voltage_decay: 0.02
```

Compare baseline vs. clamp/penalty variants on:

- best full-val SNN MSE
- mean percentage error
- median percentage error
- rollout behavior with `record_snnv2_longbest`

Recommended first implementation: add optional config flags with defaults preserving current behavior, then run a 40-epoch short comparison before any new 300-epoch run.
