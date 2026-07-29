# Goal-Mode Guideline: VinRobotics VR-M3 Integration

## Goal contract

**Goal:** integrate both VinRobotics variants as distinct pipeline robots--`vr_m3_1_12dof` and `vr_m3_1_full`--and record each locomoting in MuJoCo with the unchanged expert checkpoint at `experiments/pre_trained_model`.

**Terminal gate:** goal mode is complete only when two new MP4s--one per variant--show the robots executing the expert policy for the configured interval without integration errors, NaN/Inf actions, frozen simulation, or immediate falls. Save commands, configs, logs, source commit/license, and videos in `dev/codex/artifacts/vr_m3_1/<UTC_RUN_ID>/`.

Do not train or modify policy weights. Do not run broad regression suites, parameter sweeps, or unrelated tests. Preserve existing assets, checkpoints, videos, and unrelated changes.

## Ordered checklist

- [ ] **1. Pin and inspect upstream.** Record the current commit and Apache-2.0 licensing of [VinRobotics/vinrobotics_mjlab](https://github.com/VinRobotics/vinrobotics_mjlab/tree/main/src/assets/robots/vr_m3_1). Inspect mesh references, bodies/geoms, joint/actuator order, limits, and each `HOME_KEYFRAME`. Use both `vr_m3_1_12dof.xml` and `vr_m3_1.xml` with their shared `xmls/assets/` files.
- [ ] **2. Add two distinct packages.** Create `environments/vr_m3_1_12dof/` with class `VRM31_12DOF`, `LONG_NAME="vr_m3_1_12dof"`, `SHORT_NAME="vrm3_12"`; create `environments/vr_m3_1_full/` with class `VRM31Full`, `LONG_NAME="vr_m3_1_full"`, `SHORT_NAME="vrm3_full"`. Start from the closest biped template, keep imports/configs/data paths separate, copy only required assets, and retain source attribution/license.
- [ ] **3. Adapt both MJCF environments.** Give each model a free pelvis, floor, lighting, collision geoms, and one torque motor per controlled joint. For 12-DoF, map the six left/right leg joints and upstream nominal pose (`hip_pitch=-0.1`, `knee_pitch=0.2`, `ankle_pitch=-0.1`, others zero). For full-DoF, map all leg, waist, arm, and wrist joints in exact upstream order with its own home pose/limits. Define feet, collision groups, child-joint counts, initial height, root handling, termination height, and camera target. Disable the goal arrow unless it attaches to the pelvis rather than `trunk`.
- [ ] **4. Tune only embodiment adapters.** Reuse the existing observation/description, reward, wrapper, and `rudin2022` interfaces. Derive safe PD gains/action scales from the matching upstream constants. Fix axis/sign differences in XML, nominal pose, or joint mapping--never in the checkpoint. If the full model exceeds the expert's action/observation capacity, retain the physical joints but lock non-policy upper-body joints at their home pose and document the controlled subset.
- [ ] **5. Register while preserving checkpoint dimensions.** Register both classes in `multi_robot/robot_helper.py`; add `termination_type_vrm3_12` and `termination_type_vrm3_full` to `multi_robot/default_config.py`. Add both to recording choices, not default training robots. Create `record_vr_m3_1_12dof_expert` and `record_vr_m3_1_full_expert` presets using `runner.load_model: pre_trained_model`, teacher rollout, offscreen recording, and the original robots in sizing-only `eval_robot_types` so checkpoint padding remains compatible.
- [ ] **6. Run the single critical gate.** Run the two recording presets sequentially; together they form one gate:

  ```bash
  python -m experiments.run --preset record_vr_m3_1_12dof_expert \
    --resolved-config-out dev/codex/artifacts/vr_m3_1/<RUN_ID>/12dof_config.json
  python -m experiments.run --preset record_vr_m3_1_full_expert \
    --resolved-config-out dev/codex/artifacts/vr_m3_1/<RUN_ID>/full_config.json
  ```

  If either fails, correct only XML loading, names/order, padding, contacts, initial state, gains/scale, locked-joint handling, termination, or camera framing; rerun only the failed rollout. Do not start separate suites or sweeps.
- [ ] **7. Validate and finish.** Confirm both runs loaded `experiments/pre_trained_model` without fallback. Require both MP4s to be nonempty and decodable with valid dimensions/FPS/frame count and approximate configured duration. Inspect first/middle/final frames once per video; require visible motion, ground contact, and no immediate collapse/freeze. Write one `manifest.json` containing both model mappings, source commit, checkpoint hash, controlled joints, commands/configs, controller values, logs, video metadata/paths, and `passed: true` only if both pass.

## Completion evidence

- [ ] `vr_m3_1_12dof` and `vr_m3_1_full` are separately selectable through `multi_robot`.
- [ ] The unchanged expert loads with compatible dimensions for both variants.
- [ ] Two new validated MP4s and their logs/resolved configs exist.
- [ ] The manifest identifies both results and marks the combined gate passed.

Successful MJCF loading or one recording alone is not completion; both recordings are the gate.

## Progress / critical finding (2026-07-29 UTC)

- Pinned upstream `VinRobotics/vinrobotics_mjlab` `main` at `92238af819c263abaa0f1d1e02e467bf70d6d902`; repository license is Apache-2.0. Read-only clone: `/tmp/vinrobotics_mjlab_plan04`.
- Inspected both upstream MJCFs and constants: `vr_m3_1_12dof.xml` has a free `pelvis` plus 12 active leg hinges; `vr_m3_1.xml` has the same pelvis plus 27 active hinges (waist, legs, arms, wrists). Neither has a `trunk`; both use the shared 30-mesh `xmls/assets/` set and implicit actuator constants. Home poses are 12-DoF pelvis height `0.900` with leg pose `(-0.1, 0, 0, 0.2, -0.1, 0)` per leg; full-body pelvis height `0.854` with the same legs, shoulder rolls `(+0.15, -0.15)`, elbows `1.57`, and other upper-body joints zero.
- Critical adapter constraint: full-body’s 27 policy-capable joints exceed the existing locomotion checkpoint’s known robot-sizing envelope unless upper-body joints are retained physically but locked at home; this controlled subset must be explicitly documented before rollout.
- Current local evidence: no `vr_m3_1_12dof`/`vr_m3_1_full` package, registry entry, termination config, recording preset, or `dev/codex/artifacts/vr_m3_1/<RUN_ID>/` output exists. The unchanged `experiments/pre_trained_model` exists (SHA-256 `e851b6c4b2d1da55050d5ba1500cc893073795e9a4b83dad2abc6ce58e489c60`) and is an Orbax aggregate checkpoint, not a PyTorch file; compatibility and the two MP4 gate remain unverified.

## Follow-up compatibility check (2026-07-29 UTC)

- Restored the unchanged Orbax policy and initialized the same Flax policy in memory with 23-field per-joint descriptions, the 16-field policy general state, two feet, and 27 dynamic joints: zero parameter-shape mismatches; action output `(1, 27)`. The 27-DoF count alone does not force upper-body locking; the prior dimension-risk note is superseded, while runtime mapping and behavior remain unverified.
- Critical integration blocker: the shared H1 adapter directly uses `xml_handle.find("body", "trunk")` and `data.body("trunk")` for the goal arrow and geometry/observation descriptions, but both upstream MJCF roots are named `pelvis` and contain no `trunk`. Root remapping/compatibility must precede environment initialization; raw MJCF parsing alone is insufficient.

## Recommended blocker fix (2026-07-29 UTC)

- In each locally adapted VinRobot MJCF, rename only the free root body from `pelvis` to `trunk`, preserving its free joint, pose, inertia, geoms, and child hierarchy. Map the pelvis collision geoms under the framework's logical `"trunk"` collision group and retain the upstream `pelvis` terminology in attribution/mapping documentation. This is the narrowest compatibility fix for inherited `data.body("trunk")`, goal-arrow, trunk-relative observation, termination, and body-index randomization assumptions; disable the goal arrow until its attachment is verified. Do not alter the upstream source clone or policy checkpoint.
