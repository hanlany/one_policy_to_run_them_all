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
