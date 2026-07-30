# Goal-Mode Guideline: VinRobotics VR-M3 Integration

## Goal contract

**Goal:** integrate both VinRobotics variants as distinct pipeline robots--`vr_m3_1_12dof` and `vr_m3_1_full`--and record each locomoting in MuJoCo with the unchanged expert checkpoint at `experiments/pre_trained_model`.

**Terminal gate:** goal mode is complete only when two new MP4s--one per variant--show the robots executing the expert policy for the configured interval without integration errors, NaN/Inf actions, frozen simulation, or immediate falls. Save commands, configs, logs, source commit/license, and videos in `dev/codex/artifacts/vr_m3_1/<UTC_RUN_ID>/`.

Do not train or modify policy weights. Do not run broad regression suites, parameter sweeps, or unrelated tests. Preserve existing assets, checkpoints, videos, and unrelated changes.

## Goal-mode execution rules

- The ordered checklist is the work queue. Always advance the earliest incomplete item that can be acted on.
- Limit read-only investigation to information required for the next implementation action. Once a viable fix exists, implement it before investigating additional risks.
- Cap design investigation at 10% of the goal budget unless an implementation or focused test exposes a new blocker.
- Do not append speculative audits, repeated readiness reviews, or standalone risks. Append a finding only when an actual implementation or test fails, the finding explains that failure, and it changes the next action.
- Every continuation must produce at least one source/config/asset change, focused test or rollout result, checked checklist item, or concrete blocker that cannot be resolved locally.
- After two consecutive continuations without implementation progress, stop investigating and implement the best-supported solution.
- Keep new progress notes to five bullets maximum and update existing status instead of adding another audit section.
- Measure progress by completed checklist items and terminal-gate artifacts, not token usage or the number of findings.
- Budget the remaining work approximately as follows: 45% packages/MJCF/adapters, 20% registration/presets, 20% rollout debugging/MP4 validation, 10% focused investigation triggered by failures, and 5% manifest/documentation.

**Mandatory next action:** implement the 12-DoF vertical slice: create its package and attributed asset bundle, adapt/compile its MJCF, instantiate its environment, restore the unchanged expert, and run a focused short rollout. Do not add another design audit before this attempt. Once the 12-DoF slice works, apply the established pattern to the full variant.

## Ordered checklist

- [x] **1. Pin and inspect upstream.** Recorded commit `92238af819c263abaa0f1d1e02e467bf70d6d902` and Apache-2.0 licensing of [VinRobotics/vinrobotics_mjlab](https://github.com/VinRobotics/vinrobotics_mjlab/tree/main/src/assets/robots/vr_m3_1). Inspected mesh references, bodies/geoms, joint/actuator order, limits, controller constants, and each `HOME_KEYFRAME` for both `vr_m3_1_12dof.xml` and `vr_m3_1.xml` with their shared `xmls/assets/` files.
- [x] **2. Add two distinct packages.** Create `environments/vr_m3_1_12dof/` with class `VRM31_12DOF`, `LONG_NAME="vr_m3_1_12dof"`, `SHORT_NAME="vrm3_12"`; create `environments/vr_m3_1_full/` with class `VRM31Full`, `LONG_NAME="vr_m3_1_full"`, `SHORT_NAME="vrm3_full"`. Start from the closest biped template, keep imports/configs/data paths separate, copy only required assets, and retain source attribution/license.
- [x] **3. Adapt both MJCF environments.** Give each model a free pelvis, floor, lighting, collision geoms, and one torque motor per controlled joint. For 12-DoF, map the six left/right leg joints and upstream nominal pose (`hip_pitch=-0.1`, `knee_pitch=0.2`, `ankle_pitch=-0.1`, others zero). For full-DoF, map all leg, waist, arm, and wrist joints in exact upstream order with its own home pose/limits. Define feet, collision groups, child-joint counts, initial height, root handling, termination height, and camera target. Disable the goal arrow unless it attaches to the pelvis rather than `trunk`.
- [ ] **4. Tune only embodiment adapters.** Reuse the existing observation/description, reward, wrapper, and `rudin2022` interfaces. Derive safe PD gains/action scales from the matching upstream constants. Fix axis/sign differences in XML, nominal pose, or joint mapping--never in the checkpoint. If the full model exceeds the expert's action/observation capacity, retain the physical joints but lock non-policy upper-body joints at their home pose and document the controlled subset.
- [x] **5. Register while preserving checkpoint dimensions.** Register both classes in `multi_robot/robot_helper.py`; add `termination_type_vrm3_12` and `termination_type_vrm3_full` to `multi_robot/default_config.py`. Add both to recording choices, not default training robots. Create `record_vr_m3_1_12dof_expert` and `record_vr_m3_1_full_expert` presets using `runner.load_model: pre_trained_model`, teacher rollout, offscreen recording, and the original robots in sizing-only `eval_robot_types` so checkpoint padding remains compatible.
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

## Frozen investigation record

The investigation notes below are retained as implementation reference and are now frozen. Do not append another audit unless an implementation or focused test fails and the new finding directly changes the next action. New progress belongs in the ordered checklist or a single concise current-status update.

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

## MJCF and controller gate evidence (2026-07-29 UTC)

- Focused MuJoCo loading of both pinned XMLs succeeds, but neither is directly runnable: `vr_m3_1_12dof` is `nq=19,nv=18,njnt=13,nu=0` and `vr_m3_1` is `nq=34,nv=33,njnt=28,nu=0`; both have root `pelvis` and no `floor`. Each local MJCF must add the framework world elements and exactly 12/27 actuators in XML joint order.
- The inherited H1 path assigns `qpos[7:]`, `qvel[6:]`, and `data.ctrl` as the controlled joint/action vector. Since the unchanged policy accepts 27 dynamic joints, the full variant should retain and actuate all 27 hinges; leaving upper-body hinges physical while omitting their actuators would create a dimension mismatch. This supersedes the earlier upper-body-lock suggestion.
- MuJoCo reports 20 named foot collision geoms, ten per side (`left_ankle_roll_link_collision_1..10` and `right_ankle_roll_link_collision_1..10`), not H1's `left_foot_1..3`/`right_foot_1..3`. Map all ten per side into logical `left_foot`/`right_foot` collision groups. Upstream controller constants are per-joint; derive adapter gains and action scales from `0.25 * effort_limit / stiffness`, rather than copying H1's scalar values, without changing the checkpoint.

## H1 contract probe (2026-07-29 UTC)

- An explicit name/shape probe found both pinned models missing H1-required `trunk`, `floor`, and `left_foot_1..3`/`right_foot_1..3` names, while their dynamic dimensions are correct (`12/12/nu=0` and `27/27/nu=0`). Renaming `pelvis` to `trunk` therefore fixes only the root lookup; each local adapter must also provide a floor and logical foot lookup/aliases while preserving all 10 real foot collision geoms per side.
- H1 `get_name_to_description_vector()` dereferences `left_foot_2`/`right_foot_2`, and its seen-robot randomizer mutates fixed H1 geom IDs `[14,15,16,29,30,31]`; those assumptions cannot be inherited by name/order. Use VinRobot name-derived IDs. The reward's `qpos[17:]` versus nominal `[10:]` slice is also H1-specific and needs an embodiment-specific slice/term, especially for 12-DoF.
- H1’s policy description contract has 23 per-joint fields and a 3-scalar general gain/action-scale context. Upstream per-joint gains may be used internally by the controller, but passing gain/scale vectors directly through the existing context construction would change or break the checkpoint input shape; retain the 23-field layout and exact scalar general-context shape in the adapter.

## In-memory adapted-MJCF preflight (2026-07-29 UTC)

- The first synthetic adaptation failed at MuJoCo sensor parsing after the root rename: both upstream files contain `<subtreeangmom body="pelvis">`. The root remap must update that sensor reference (and any future object-name references), not only the `<body>` name.
- After remapping the sensor, adding floor geom id 0, non-colliding H1 foot aliases, and one motor per active hinge, both variants compiled in memory: 12-DoF `nq=19,nv=18,nu=12`; full `nq=34,nv=33,nu=27`; root `trunk`; foot aliases resolved. This confirms the remaining pre-rollout work is adapter/controller/observation mapping rather than an unresolvable MJCF structural conflict.
- The verified active joint order is left leg 6, right leg 6, then for full-body `waist_yaw` followed by left arm 7 and right arm 7. Active direct-child counts are `1,1,1,1,1,0` per leg, `2` for waist, and `1,1,1,1,1,1,0` per arm; use these values in the 23-field descriptions.

## Expert forward preflight (2026-07-29 UTC)

- Restored the unchanged aggregate `experiments/pre_trained_model` and called the unchanged Flax `Policy` with the checkpoint contract: 23-field joint descriptions, 3-field joint state, 10-field foot descriptions, 2-field foot state, and 16-field policy general state. Both 12-joint and 27-joint inputs produced finite mean/logstd outputs with shapes `(1,12)` and `(1,27)`.
- This confirms both action dimensions are accepted by the actual checkpoint without fallback or parameter reshaping. Remaining risk is runtime observation/value mapping, controller behavior, contacts, termination, and locomotion; no MP4 gate has been run.

## In-memory expert dynamics probe (2026-07-29 UTC)

- With the structural fixes, upstream per-joint PD gains/action scales, and the actual restored checkpoint, both synthetic adapters produced finite policy actions from model-derived 23-field descriptions: `(12,23)` for 12-DoF and `(27,23)` for full-body.
- This is diagnostic only: the 12-DoF probe fell to root `z=0.1073` with maximum roll/pitch norm `2.5864`; the isolated full-body probe remained finite at root `z=0.7466` but reached `1.1143` rad tilt and ended without contact. It used manually constructed descriptions, zero foot-state/command context, and no repository environment, so it is not a rollout/MP4 pass.
- Recommended fix remains to implement and verify an embodiment-specific runtime adapter: exact VinRobot joint/foot/root observation mapping, contact/termination and reward slices, and a controller using the upstream per-joint gains/effort-derived action scales. The finite forward pass proves checkpoint compatibility; the unstable synthetic dynamics prove the runtime mapping/controller gate is still the current blocker.

## Template contract audit (2026-07-29 UTC)

- The read-only H1 template audit confirms this is not an XML-only adaptation: its package hardcodes 19 H1 joint names, `qpos[7:]`/`qvel[6:]`, fixed foot aliases, foot geom IDs `[14,15,16,29,30,31]`, body index `1`, and the reward slice `qpos[17:]`. A VinRobot package must parameterize or fork these paths for 12/27 joints and use name-derived geometry/body IDs.
- `Rudin2022Control` currently applies one scalar `p_gain`, `d_gain`, and action scale to every joint, whereas VinRobot publishes per-joint gains and effort limits. The adapter must use vector PD gains and `0.25 * effort_limit / stiffness` action scales while preserving the checkpoint’s 23-field description and 3-field general context.
- Runner evidence: `record_robots` contains no VinRobot entry, and `python -m experiments.run --preset record_vr_m3_1_12dof_expert --print-only` rejects the planned preset as an invalid choice. Registration, per-robot termination keys, recording presets, and artifact wiring remain the pre-MP4 blocker.

## Multi-robot routing compatibility audit (2026-07-29 UTC)

- `multi_robot.create_env` instantiates every `train_robot_types + eval_robot_types` once to compute shared maximum observation/action sizes; `eval_robot_types` contributes sizing even when `nr_eval_envs=0`, but no evaluation rollout is created. Each train adapter then receives the shared maxima and pads its local spaces.
- The unchanged Flax `Policy` emits one action per dynamic joint. PPO appends zeros for each adapter’s `missing_nr_of_actions` and slices back to the local action count before `env.step`. Thus 12-DoF and 27-DoF VinRobot variants can use the unchanged checkpoint without upper-body locking; exact `model.nu`, joint order, and padding counts remain gate invariants.
- For the planned one-environment recording presets, `record_robot_index=0` selects the sole VinRobot train environment; original robots in `eval_robot_types` should remain sizing-only. This reduces checkpoint/padding risk but does not remove the missing registration, adapter, and recording-preset work.

## Recording/artifact gate audit (2026-07-29 UTC)

- The recorder defaults `algorithm.record_dir` to `experiments/videos`; `--resolved-config-out` writes only the resolved JSON and does not relocate MP4s or capture logs. Each gate preset/command must set the run-specific artifact video directory and capture stdout/stderr into the same run directory; the runner creates no manifest.
- `get_record_timing()` ignores `algorithm.record_fps` and derives both FPS and frame count from simulation `dt`: at `timestep=0.005`, a 10-second recording is 200 FPS and 2,000 frames. Record and validate the actual video metadata rather than trusting the configured FPS field.
- MuJoCo confirms body `0` is `world` and the H1 root `trunk` is body `1`; the generic offscreen recorder sets `trackbodyid=0`. VinRobot recording must explicitly target the remapped root body (or prove framing from first/middle/final frames), otherwise a world-tracked camera can lose the robot. The MP4 gate remains unrun.

## Recording timing correction (2026-07-29 UTC)

- The prior `200 FPS/2,000 frames` example applies only when the environment `dt` itself is `0.005`. With the existing 50 Hz controller at `timestep=0.005`, `nr_substeps=4`, `env.dt=0.02`, and `get_record_timing(10.0, env.dt)` yields `50 FPS/500 frames`; use each adapter’s reported `dt` in the manifest.
- Both pinned upstream XMLs confirm `world` body `0` and `pelvis` body `1`; renaming the root to `trunk` preserves the camera target ID `1`. The generic recorder still needs an explicit root-target override or frame-level verification.

## Expert input-context mapping audit (2026-07-29 UTC)

- The unchanged policy mask consumes exactly 16 general fields: trunk angular velocity (3), command velocity (3), projected gravity (3), scalar `p_gain`/`d_gain`/`action_scaling_factor` (3), mass (1), and robot dimensions (3). Preserve this order and shape; the gain triple is not a vector input.
- The 23-field joint description places per-joint `p_gain`, `d_gain`, and action scale at indices `16:19` (position `0:3`, axis `3:6`, child count `6`, nominal `7`, limits/physical fields `8:16`, mass `19`, dimensions `20:23`). Recommended mapping: encode VinRobot’s upstream per-joint controller values in each joint’s `16:19` fields, while supplying a documented scalar summary in the three general gain fields; never concatenate gain vectors into the general context.
- The prior finite forward probe constructed these descriptions manually, so the mapping is still an adapter preflight requirement. Verify field normalization and the chosen general-state scalar summary against the actual repository observation builder before attempting MP4s.

## Upstream controller-scale audit (2026-07-29 UTC)

- Pinned VinRobot constants generate each normalized-action offset as `0.25 * effort_limit / stiffness`; do not reuse H1’s scalar `0.75` scale.
- In exact leg order `[hip_pitch, hip_roll, hip_yaw, knee_pitch, ankle_pitch, ankle_roll]`, the 12-DoF vector is `[0.600000, 0.600000, 0.270833, 0.450000, 0.150000, 0.150000]` repeated for left/right.
- Full-body keeps those 12 values, then uses waist `0.069482`, and each arm `[0.117857, 0.117857, 0.212500, 0.212500, 0.091667, 0.091667, 0.091667]` in the verified order `waist_yaw`, left arm, right arm. The adapter must pair these scales with the matching per-joint P/D gains and XML action order; this remains unverified in repository code.

## Initial-state/contact clearance audit (2026-07-29 UTC)

- The pinned `HOME_KEYFRAME` roots are `z=0.900` for 12-DoF and `z=0.854` for full-body, with leg pose `(-0.1, 0, 0, 0.2, -0.1, 0)` per side; full-body additionally has shoulder rolls `(+0.15, -0.15)` and elbows `1.57`.
- At those exact poses, the lowest named foot-collision bound is about `0.0492 m` and `0.0032 m` above a zero-height floor for 12-DoF and full-body respectively. The raw XML has no floor, so the adapter must deliberately choose a calibrated floor/contact margin rather than assume the keyframe is already grounded.
- H1’s inherited `DefaultInitialState` instead uses `initial_drop_height=0.985`, which is not either VinRobot home pose and can create an avoidable drop before policy control. Initialize the upstream keyframe, set the VinRobot-specific root/floor height, and verify initial contacts, termination height, and first-step stability before the expert rollout.

## Domain-randomization ID audit (2026-07-29 UTC)

- H1’s default domain randomization hardcodes geom IDs `[14,15,16,29,30,31]` for foot-size/friction changes and geom `31` for foot friction. In both pinned VinRobot XMLs those IDs resolve to mixed ankle/knee collision and visual geoms—not a stable semantic foot set—so copying the H1 randomizer would silently perturb the wrong geometry.
- The recommended fix is to make VinRobot randomization name-derived: resolve the intended collision geoms from the adapter’s explicit foot aliases, resolve the trunk body by name, and derive joint/DoF address ranges from the mapped joint names. Keep the first expert-recording gate deterministic (no geometry/mass/gravity sampling) until contacts and rollout stability pass; add randomized evaluation only after these semantic mappings are verified.

## Runtime observation/reward contract audit (2026-07-29 UTC)

- The actual H1 observation builder looks up one `<foot>_2` geom for each foot description, while `check_collision("floor", foot)` uses the collision-group IDs. Non-colliding aliases are therefore safe only if each VinRobot foot group also contains all 10 real ankle collision geoms; otherwise contact observations, air-time, and reward terms stay false even when the robot touches the floor.
- `RudinOwnVarReward` hardcodes `data.qpos[17:] - seen_joint_nominal_position[10:]`. For a free-root VinRobot this scores only the last 2 of 12 dynamic joints or the last 17 of 27, so the adapter must replace that H1 offset with all mapped joint qpos addresses (and the matching nominal vector).
- The H1 description code builds the general gain context with `np.array([seen_p_gain, seen_d_gain, seen_scaling_factor])`; vectorizing those fields for VinRobot would break the fixed 23-field description concatenation. Keep vector P/D/scale arrays in a separate controller path, encode them at per-joint description indices `16:19`, and provide documented scalar summaries in the fixed three-field general context.

## Joint-order/axis audit (2026-07-29 UTC)

- Compiled MuJoCo addresses are contiguous and authoritative: both variants use dynamic qpos `7:19`/dof `6:18` for the 12 legs; full-body continues qpos `19:34`/dof `18:33` as waist, left arm, right arm. Build actuators and observation vectors by these joint IDs, not by a separately sorted semantic list.
- The XML contains real mirrored/asymmetric coordinates and limits (for example right hip/ankle pitch axes differ slightly from left, and left/right hip-roll ranges are reversed). Preserve the raw `model.jnt_axis`, `jnt_range`, nominal positions, and signs; do not mirror or canonicalize them while adapting the checkpoint.
- Upstream action configuration is a joint-position offset around `HOME_KEYFRAME`, with per-joint scale `0.25 * effort_limit / stiffness`, followed by P/D control. The full-body source keyframe mentions nonexistent `waist_roll_joint`, while XML/actuators contain only `waist_yaw_joint`; ignore that stale key rather than adding an extra action or shifting the 27-joint order.

## Runner registration/preset audit (2026-07-29 UTC)

- Current `robot_helper.ROBOTS` and `presets.yaml:record_robots` contain neither VinRobot variant, and the runner rejects `record_vr_m3_1_12dof_expert` as an invalid preset. `multi_robot.create_env` imports every registered class, instantiates every train/eval type for shared sizing, and immediately reads `termination_type_<SHORT_NAME>`; both exact registry specs and termination keys are required before even a print-only or initial-sizing check can pass.
- Existing `record_default` proves the required teacher/checkpoint shape (`runner.load_model=pre_trained_model`, `runner.mode=test`, `environment.mode=test`, `algorithm.rollout_policy_stage=teacher`, one train env, `record_robot_index=0`), but the base config also enables `add_goal_arrow=True` and leaves videos at the global default. New VinRobot presets must explicitly set `add_goal_arrow=False` until root attachment is verified and set a run-specific `algorithm.record_dir`.
- Add both names to the recording choices, but keep each named gate preset’s `train_robot_types` as the sole VinRobot variant and its `eval_robot_types` explicit/original for sizing-only compatibility; retain `record_robot_index=0` so the recorder selects that sole train environment.

## Termination/root-collision audit (2026-07-29 UTC)

- H1 `HeightTermination` hardcodes `data.qpos[2] < 0.545`; that absolute threshold is not derived from VinRobot’s `0.900`/`0.854` home roots or the chosen floor. Use a per-variant calibrated threshold relative to the floor/home pose and verify that the initial keyframe is non-terminated while a fallen root terminates promptly.
- H1 `Rudin2022Termination` reads `collision_groups["trunk"]`, but the H1 constructor registers only `floor`, `left_foot`, and `right_foot`; copying that termination path can raise `KeyError`. The adapted model’s root collision group must be explicit and name-derived (`pelvis_collision_1` becomes the logical `trunk` geom after the root remap), or the variant should use a dedicated termination implementation.
- Both raw VinRobot models have root body ID `1` and one root collision geom; preserve that mapping after renaming, and validate height plus root-contact termination at reset, first step, and after a controlled fall before the expert recording gate.

## Torque-actuator contract audit (2026-07-29 UTC)

- H1 compiles one `<motor>` per dynamic joint with `gear=1`; its `data.ctrl` is therefore a direct torque command. The pinned VinRobot XMLs compile with `nu=0`, so their `actuatorfrcrange` joint metadata alone does not actuate anything.
- Add exactly 12/27 named motors in compiled joint order, with `gear=1` and `ctrlrange` from the VinRobot limits: legs ±360/±130/±120 by joint type; full-body additionally waist ±102, shoulder ±66, shoulder-yaw/elbow ±34, and wrist ±11. Assert `model.nu == len(joint_names) == policy_action_dim` after compilation.
- Clip the vector P/D torque output to those per-joint ranges and verify actuator target IDs, gear, ranges, finite torques, and first-step motion before recording. Do not inherit H1’s torque limits or rely on the raw XML’s no-actuator state.

## Asset-packaging audit (2026-07-29 UTC)

- Both pinned XMLs use `<compiler meshdir="assets">` and declare the same 30 STL meshes, with no includes, textures, or external files beyond that directory. The upstream `get_assets()` helper returns an empty mapping, so a local `from_xml_path` depends on preserving this relative layout.
- Each distinct package should keep its attributed XML under its own `data/` tree with the shared `data/assets/*.stl` files (or explicitly prune unused declarations and recompile); copying only the 12-DoF leg meshes is insufficient because that XML still declares arm/head/waist meshes.
- Before registration, compile each packaged XML from its local path and record the asset manifest plus source commit/license. This isolates the runner from `/tmp`/upstream paths and proves the copied meshes, renamed root/sensor references, floor, aliases, and actuators are the same model being gated.

## Description-normalization domain audit (2026-07-29 UTC)

- The unchanged H1 builder uses fixed physical reference denominators in the 23-field joint descriptions: stiffness `15`, damping `5`, armature `0.1`, friction loss `0.6`, torque `500`, velocity `17.5`, and per-joint P/D/scale references `50/1/0.4`; the policy receives these values directly after the description Dense/LayerNorm path, with no variant-specific rescaling layer.
- VinRobot consequently produces large distribution shifts that are finite but not H1-like: full-body waist values alone give approximately normalized P `6.34` (`367/50-1`), D `28` (`29/1-1`), stiffness `23.47` (`367/15-1`), and friction loss `29` (`18/0.6-1`). Do not silently rescale or clip per variant; that would change the checkpoint’s physical-context semantics.
- Before the MP4 gate, dump min/max of every 23-field row and the fixed 16-field general context for both variants, assert exact masks/shapes and finiteness, and record the scalar general-context summary. Treat unexpected out-of-distribution values or rollout instability as an adapter blocker rather than hiding it with normalization changes.

## Explicit-torque versus implicit-PD audit (2026-07-29 UTC)

- Upstream `BuiltinPositionActuatorCfg` defines VinRobot stiffness/damping/effort/armature/friction as actuator behavior, while the raw XML compiles with zero passive damping, armature, and stiffness. The repository instead computes explicit P/D torque and writes it to direct motors; these are different control contracts.
- The adapter must choose one source of each effect: use gear-1 torque motors plus the vector P/D controller to reproduce the upstream position-offset behavior, and do not also add implicit position actuators or copy their P/D values into passive joint fields unless the resulting combined dynamics are intentional and measured. Preserve inertial/friction terms only once.
- Keep the source controller arrays available separately for the 23-field description and manifest; verify `model.dof_*`/`jnt_*`, computed torques, and zero-action equilibrium together. A finite forward pass with duplicated or missing passive/actuator gains is not evidence of a valid expert rollout.

## MP4 validation-tooling audit (2026-07-29 UTC)

- The recorder writes OpenCV `mp4v` files named `<UTC-second>_<stage>_envNN_<robot>.mp4`, at FPS derived from `dt`, with `round(record_seconds / dt)` frames and a nominal 1280×720 frame capped by MuJoCo offscreen limits; `algorithm.record_fps` is not authoritative.
- `ffprobe`/`ffmpeg` are unavailable in the current environment, but `cv2` and `imageio` are installed. Validate each future file with `cv2.VideoCapture`: nonzero size, successful first/middle/final reads, decoded width/height, FPS, frame count, and duration against the adapter’s reported `dt`; use `imageio` only as a fallback decoder.
- `dev/codex/artifacts/vr_m3_1/` is currently absent. The gate remains incomplete until each run captures stdout/stderr and resolved config beside its MP4, and a manifest records the actual filename, decoder metadata, selected robot, checkpoint-load evidence, and visual pass/fail.

## Grounding invariant and recommended initial-state fix (2026-07-29 UTC)

- Exact pinned upstream home roots are `0.900` (12-DoF) and `0.854` (full-body). The lowest named ankle-roll collision bounds are approximately `0.0490` and `0.0030` m, respectively, so both poses encode the same root-to-foot clearance of approximately `0.8510` m. The raw XMLs contain no floor.
- Recommended adapter fix: preserve each source joint pose and place the per-variant floor just below its measured lowest-foot bound, with an explicit 1–3 mm clearance; alternatively use a common zero floor and set both root heights to approximately `0.8510` m plus that clearance. Do not combine the source root heights with a zero floor, because that gives the 12-DoF and full-body resets materially different ground clearance.
- Make reset a gate: assert lowest-foot clearance, no root/floor penetration, expected foot-floor contacts, no masked self-contact, and false height/orientation termination before the first action. Record floor/root heights, lowest-foot bound, contact pairs, and reset termination flags in the run manifest.

## Current terminal-gate readiness audit (2026-07-29 UTC)

- Rechecked the current tree: no `vr_m3_1`/`vrm3` files exist under `one_policy_to_run_them_all/environments`, and no VinRobot registry, termination, or `record_vr` preset matches exist in the source or `experiments/configs/presets.yaml`.
- `python -m experiments.run --preset record_vr_m3_1_12dof_expert --print-only` still fails during CLI parsing with `invalid choice`; `dev/codex/artifacts/` is absent and no VinRobot MP4 or manifest exists. Only the plan file is modified.
- The in-memory MJCF/checkpoint probes therefore remain design evidence, not terminal-gate evidence. The immediate blocker is implementation/registration/preset wiring; checklist items 2–7 and all completion evidence remain unfulfilled until both packaged adapters produce validated recordings.

## Adapter initialization-order audit (2026-07-29 UTC)

- `UnitreeH1.__init__` compiles the XML and then calls `self.get_name_to_description_vector()`, `self.get_observation_space(...)`, and `self.get_initial_observation(...)` before it returns. Those methods assume `body("trunk")`, contiguous `qpos[7:]`/`qvel[6:]`, H1’s 19-joint arrays, and scalar controller fields.
- A VinRobot subclass cannot safely set address, foot, root, controller, or home-state mappings after `super().__init__`; Python dynamic dispatch would invoke overridden builders during base construction with incomplete adapter state. Resolve all model names/addresses and per-joint values before those builders run (or use a dedicated shared base/factory), then override every runtime slice, control, reward, and termination path that is H1-specific.
- `multi_robot.create_env` first constructs every train/eval type once for max sizing, reads `termination_type_<SHORT_NAME>`, then constructs normal environments and resets them. The adapter must pass this constructor smoke test before checkpoint/video logic, with asserted 12/27 action counts, observation lengths, finite initial observations, and correct padding counts.

## Teacher-action semantic audit (2026-07-29 UTC)

- The recording path in `PPO.test()` uses the deterministic policy mean, appends only multi-robot zero padding, and passes the result through an identity `get_processed_action`; the environment receives the raw checkpoint action values.
- H1 interprets those values as normalized joint position offsets, applying one scalar `0.75` scale before scalar P/D torque. VinRobot must preserve the same normalized-action contract but apply its ordered vector `0.25 * effort_limit / stiffness` scales and vector P/D gains inside the adapter; do not rescale the checkpoint in PPO or write the unitless action directly to `data.ctrl`.
- The first runtime gate must log raw action, target position offset, clipped torque, actuator index/range, and finite-state checks for both 12- and 27-joint paths. A shape-compatible deterministic policy call is insufficient if this conversion is missing, duplicated, or applied in the wrong joint order.

## Checkpoint-config merge audit (2026-07-29 UTC)

- `PPO.load` removes `rollout_policy_stage`, `record`, `record_robot_index`, `record_seconds`, `record_dir`, `record_fps`, and student/SNN keys from the aggregate restore target, so checkpoint-saved values cannot silently replace those recording and teacher controls.
- Each gate preset must still explicitly pin `algorithm.rollout_policy_stage: teacher`, `algorithm.record: true`, `algorithm.record_robot_index: 0`, and the run-specific directory/timing. Other unmasked algorithm fields may be restored from the checkpoint unless explicitly set; capture the resolved post-load configuration and checkpoint identifier/hash.
- `record_default` is the current proof-of-pattern for teacher/test recording. The VinRobot presets must reproduce those fields while adding variant-specific train/eval sizing and artifact paths; checkpoint restoration alone does not prove the selected stage or output location.

## Terrain XML-selection audit (2026-07-29 UTC)

- The H1 plane terrain handler chooses the first `.xml` returned by unsorted `os.listdir(data_dir)` and fixes `center_height=0.0`; the environment compiles that path before model setup. Reusing it with multiple variant XMLs can select the wrong model, and the pinned VinRobot XMLs themselves contain no floor.
- Recommended adapter contract: use a variant-local terrain handler with an explicit XML filename, assert the resolved file is the intended package asset, add the calibrated floor to that model, and set `center_height` to the same floor reference. Keep one authoritative XML per variant if the generic handler is retained.
- The constructor gate must record the resolved absolute XML path, asset manifest, floor/root IDs and heights, `center_height`, and `model.nu`; fail if any do not match the selected `vr_m3_1_12dof` or `vr_m3_1_full` package.

## Multi-robot wrapper-protocol audit (2026-07-29 UTC)

- PPO initialization queries each vector environment through `env.call` for `robot_type`, `model`, `data`, `dt`, `observation_name_to_id`, dynamic joint/foot counts and lengths, description sizes, and `missing_nr_of_actions`; the vector wrappers do not synthesize missing adapter metadata. The episode wrapper supplies `robot_type`, but the VinRobot class must expose the rest.
- The wrapper path also requires valid Gym spaces, reset/step outputs, finite padded observations, and a `data` handle that remains synchronized with the selected model during recording. A direct class constructor or standalone Flax forward pass does not test this protocol.
- Add an actual `multi_robot.create_env` smoke test for each named preset before checkpoint loading: query every metadata field, assert the 12/27 dynamic counts and 23/10 description sizes, verify `missing_nr_of_actions`/observation padding, reset once, and close all environments cleanly.

## Domain-randomization initialization audit (2026-07-29 UTC)

- `mode="test"` disables later sampling, but H1 still calls `DefaultDomainSeenRobotFunction.init()` and `DefaultDomainMuJoCoModel.init()`. Those initializers read H1 assumptions: root/body index `1`, contiguous `dof[6:]`/`jnt[1:]`, passive damping/armature/stiffness, actuator ranges, and fixed geom `31`/foot IDs.
- The pinned VinRobot XMLs have zero passive damping, armature, and stiffness and no actuators. If the adapter uses the required direct motors plus explicit vector P/D, blindly reusing these initializers will put zero or semantically wrong physical/controller fields into the 23-field policy descriptions even when randomization is “off.”
- Recommended fix: provide variant-specific, name/order-derived seen-robot state from the upstream controller constants (and deterministic no-op mutation hooks for the first gate), or intentionally populate each MuJoCo field exactly once. At reset, assert description fields match the source arrays and that every mutation target resolves semantically, not just by fixed index.

## Constructor/model dependency audit (2026-07-29 UTC)

- In `UnitreeH1.__init__`, control, command, reward, termination, domain, observation, and terrain factories are constructed before the XML is compiled into `self.model`; only afterward are model addresses, names, and initial state available.
- A VinRobot vector controller/description/domain adapter needs compiled joint/body/geom/actuator addresses and source limits. Forking H1 and deriving those values inside an early factory or `super().__init__` therefore creates an order-dependent failure or silently falls back to H1 assumptions.
- Recommended fix: make model selection/compilation and name-derived mappings the first phase; then construct model-dependent helpers; then initialize home state, domain state, descriptions, spaces, and cached observation indices. Keep explicit XML path and root/floor/actuator assertions before helper initialization.
- Gate this sequence with a constructor-only smoke test for each variant: compile the intended XML, assert `nq/nv/nu`, resolve every controlled joint/foot/trunk target, initialize finite observations, and close before loading the checkpoint.

## Motor velocity/limit mapping audit (2026-07-29 UTC)

- Pinned VinRobot constants expose per-joint `max_vel` distinct from effort/stiffness: each leg is `[14.653, 14.653, 31.4, 14.653, 16.747, 16.747]` in `[hip_pitch, hip_roll, hip_yaw, knee_pitch, ankle_pitch, ankle_roll]` order; full-body then adds waist `4.18` and each arm `[4.29, 4.29, 5.13, 5.13, 6.17, 6.17, 6.17]`.
- H1 hardcodes different velocity arrays and uses them both for qvel clipping and the 23-field seen-velocity description. Reusing them would distort VinRobot policy context and permit/clamp dynamics at the wrong physical limits, even with correct vector P/D and action scales. Use source `max_vel` by name/order; use source `effort_limit` for motor `ctrlrange`, not `saturation_tau`.
- Recommended fix: keep the framework checkpoint normalizers (`qvel/35`, etc.) unchanged for compatibility, but feed source `max_vel` into physical clipping/context, log raw versus normalized qvel, and assert the arrays match the compiled XML/controller order in the manifest.

## General-context scalarization audit (2026-07-29 UTC)

- PPO’s policy mask accepts exactly three controller context fields—`p_gain`, `d_gain`, and `action_scaling_factor`—while VinRobot supplies vectors; concatenating those vectors would change the unchanged checkpoint input contract.
- H1 uses one `gains_and_action_scaling_factor` triple both in every joint/foot description and in the general observation. Replacing it with a vector would also change description width and fail the 23-field contract.
- Recommended fix: use the arithmetic mean of each ordered VinRobot P/D/scale vector as the first-gate scalar general context, keep the exact per-joint vectors in description fields `16:19`, and assert/manifest both the vectors and their scalar means. Do not fall back to H1 defaults or the first joint.

## Visual-scene/offscreen audit (2026-07-29 UTC)

- Both pinned VinRobot XMLs contain no explicit `<light>` or `<camera>` (and no floor/actuator); H1 data includes lights and a track camera, while the generic recorder creates a tracking camera at body `0`.
- A valid, decodable MP4 can still be dark or frame the world instead of the robot, so codec/frame-count success is not evidence of the visual terminal gate.
- Recommended fix: add explicit ambient/key lighting and a variant-local camera/follow target after root remapping, assert the expected light/camera IDs and root target, and require visible robot pixels/nonuniformity at first/middle/final frames before `passed: true`; record these settings in the manifest.

## Geom-zero/floor ordering audit (2026-07-29 UTC)

- A read-only MuJoCo compile of both pinned XMLs reports `ngeom=71` and the same first geoms: `pelvis_visual_0`, `pelvis_collision_1`, `left_hip_pitch_link_visual_0`, `left_hip_roll_link_visual_0`, and `left_hip_roll_link_collision_1`; neither has a floor.
- H1’s description builder skips geom `0`, and its default model-domain state reads geom `0` friction/solver fields as the floor. Appending a floor after the VinRobot body would therefore omit the wrong geom and treat the pelvis visual as ground.
- Recommended fix: prepend the floor as the first world geom and assert `geom_id(0).name == "floor"` before descriptions/domain initialization, or replace every geom-zero assumption with name-derived IDs; record the resolved floor and first-geom IDs in the manifest.

## Contact-filtering/collision semantics audit (2026-07-29 UTC)

- The pinned XMLs contain no `<contact>`/`<exclude>` matrix; visual geoms explicitly disable contact, while collision geoms inherit defaults. The upstream task supplies separate `FULL_COLLISION` and `FEET_ONLY_COLLISION` configs with different `condim`, priority, friction, and contact behavior.
- H1’s `collision_groups` only label/query contacts; they do not establish MuJoCo filtering, so mapping VinRobot names into `left_foot`/`right_foot`/`trunk` does not by itself reproduce the source dynamics or termination behavior.
- Recommended fix: choose and encode the intended source contact mode explicitly, assert floor/foot/root contact pairs and unexpected self-contact at reset and during a short controlled rollout, and do not suppress self-collision merely to make the initial pose pass.

## Joint-level versus actuator-level torque-limit audit (2026-07-29 UTC)

- Both raw XMLs preserve `actuatorfrclimited="true"` and per-joint `actuatorfrcrange` values even though they compile with `nu=0`; adding direct motors introduces a second actuator-level `ctrlrange` path.
- If those ranges diverge—such as using saturation torque for the motor while retaining source effort limits—MuJoCo can clip the same command differently at joint and actuator level, changing the policy’s effective torque and description context.
- Recommended fix: preserve the source joint force limits, set each gear-1 motor `ctrlrange` to the matching ordered `effort_limit`, assert both arrays agree after compilation, and log which limit clipped every first-step diagnostic torque.

## Recording-list/preset-coupling audit (2026-07-29 UTC)

- The shared YAML `record_robots` anchor is reused as `eval_robot_types` by at least ten existing student/SNN recording presets; `experiments.run` also exposes it as the integer-indexed `--record-robot` list.
- `multi_robot.create_env` constructs every train/eval type in that list to compute maximum observation/action sizes even when `nr_eval_envs=0`, so adding VinRobot names to the shared anchor changes unrelated preset construction, padding, and failure surface.
- Recommended fix: append VinRobot names to the CLI recording list without reordering existing indices, freeze the legacy eval-sizing list for existing presets, and make only the two new VinRobot presets opt in to their intended sizing-only eval set; run the legacy print-only smoke checks after registration.

## Auto-reset/fall-visibility audit (2026-07-29 UTC)

- `MultiSingleVectorEnv.step_wait()` saves a terminated/truncated state, immediately calls `env.reset()`, and returns the reset observation; `PPO._record_vectorized_policy_videos()` then ignores the termination flags and captures the post-reset `data` handle.
- A policy that falls early can therefore produce a decodable video of repeated reset poses, masking the terminal fall and falsely satisfying a frame-only motion/contact check.
- Recommended fix: capture the terminal state before auto-reset or fail the gate on any termination/truncation before the configured horizon; record first-termination step, count, final info/observation, and reset events in the manifest.

## Sizing-constructor mode audit (2026-07-29 UTC)

- `multi_robot.create_env` calls each train/eval type once for space sizing through `make_env(..., purpose_initial_check=True)` without passing `mode`, so the constructor receives its default `mode="train"` even when the actual recording environment is configured as `test`.
- A VinRobot adapter that only disables randomization or hardware/display-dependent paths under `mode="test"` can fail during this sizing pass before PPO/checkpoint loading; eval-only types have the same constructor requirement.
- Recommended fix: make model/name/controller/domain initialization deterministic and safe in both sizing `train` and rollout `test`/`eval` modes, and run a constructor smoke gate in all three modes with finite spaces/observations, exact 12/27 `nu`, and clean close.

## Evaluation-mode propagation audit (2026-07-29 UTC)

- `create_env` passes `mode="eval"` to real evaluation environments, but the shared robot constructor initializes `self.eval = False` and PPO's `set_eval_mode()` is a no-op; reward/domain paths that check `self.env.eval` therefore never receive the evaluation state.
- `mode="eval"` also bypasses the constructor's `mode=="test"` deterministic switch, so a real eval rollout can retain training randomization/noise/curriculum even when the checkpoint gate assumes deterministic behavior; sizing-only eval types do not exercise this path.
- Recommended fix: define and propagate explicit mode semantics so VinRobot `test` recordings and real `eval` rollouts are deterministic/noise-free (or record the intentional difference), set eval state before reset, and smoke one actual eval environment whenever eval types are configured; do not rely on the no-op `set_eval_mode()` or the mode string alone.

## Global-description-width audit (2026-07-29 UTC)

- PPO stores `single_dynamic_joint_observation_length`, `single_dynamic_foot_observation_length`, and both description widths from environment index 0, then reuses them to reshape and split every robot type; only total dynamic lengths/counts are tracked per environment.
- A VinRobot can therefore pass its standalone Gym-space check yet be misparsed in a mixed train/eval configuration if its per-item widths differ from the first robot, silently feeding shifted fields to the unchanged policy.
- Recommended fix: assert across every train/eval adapter before PPO construction that joint descriptions are exactly 23 fields with a 3-field state, foot descriptions exactly 10 fields with a 2-field state, and the corresponding per-item lengths are identical; record these cross-robot assertions in the manifest.

## Reward-height embodiment audit (2026-07-29 UTC)

- The shared `RudinOwnVarReward` uses a fixed `nominal_trunk_z=0.985` in its base-height term, while the pinned VinRobot home roots are `0.900` and `0.854` relative to the source reference; reusing it unchanged assigns a nonzero home-pose height penalty to both variants.
- This reward also runs during the recording reset/steps, so a finite, shape-compatible rollout can still report misleading baseline/height metrics even when the teacher action path is correct; reward and termination must use the same calibrated floor/home reference.
- Recommended fix: make nominal trunk height an explicit per-variant value relative to `terrain_function.center_height`, evaluate every reward term at the exact home reset, and record baseline height/reward components in the manifest; keep this adapter correction separate from checkpoint inputs and weights.

## Unattended-command reproducibility audit (2026-07-29 UTC)

- In `mode="test"`, the shared step path checks for a joystick and then reads a cwd `commands.txt` before falling back to the seeded random command sampler; the current worktree has no `commands.txt`, but the inherited external-input branch remains active.
- A VinRobot recording can therefore vary with host input or working-directory files even when the preset, seed, checkpoint, and XML are unchanged, making expert behavior and manifest evidence non-reproducible.
- Recommended fix: give offscreen gate presets an explicit deterministic command source, disable joystick/file overrides for unattended recording, and log the exact per-step command sequence or command seed alongside the resolved config.

## Zero-command tracking-metric audit (2026-07-29 UTC)

- `RudinOwnVarReward` computes `track_perf_perc` by dividing each velocity error by `abs(desired_global_velocities)` without an epsilon or zero-component mask; a valid deterministic command with zero lateral/yaw velocity therefore emits NaN/Inf metrics, and near-zero commands are numerically unstable.
- This can contaminate the recording log/manifest even when teacher actions and MuJoCo state remain finite, so a decodable MP4 is not sufficient evidence of a clean gate.
- Recommended fix: use an epsilon-safe denominator or exclude zero command components, assert reward/info finiteness at reset and every recorded step, and store the command vector plus finite-metric result in the manifest.

## Installed-asset packaging audit (2026-07-29 UTC)

- `setup.py` includes environment assets only from files under `one_policy_to_run_them_all/environments/*/data`; a source-tree XML can therefore compile successfully while an installed wheel omits the VinRobot XML or STL meshes if they are placed elsewhere or the package lacks `__init__.py`.
- The recording gate must exercise the same installed/package-resolved paths that `multi_robot` imports, because `/tmp` or an editable source checkout can mask missing wheel data and relative `meshdir="assets"` failures.
- Recommended fix: keep each variant’s XML/meshes under its package `data/` tree, build a no-dependency wheel, inspect that both asset sets are present, and run the constructor/MJCF smoke from the packaged path before accepting either MP4.

## Actuator-limit enforcement audit (2026-07-29 UTC)

- The pinned VinRobot XMLs contain `actuatorfrcrange` on joints but no `<motor>` elements and no explicit actuator `ctrllimited`/`autolimits`; H1’s working XML explicitly enables both for its direct motors.
- Adding a `ctrlrange` without proving that MuJoCo enforces it can leave the vector P/D torque path unsaturated while the manifest incorrectly reports the intended effort limits.
- Recommended fix: set `ctrllimited="true"` with the ordered VinRobot effort ranges (and explicit compiler autolimits if used), then assert `model.actuator_ctrllimited`, `model.actuator_ctrlrange`, gear, and observed clipped `data.ctrl`/torque values before the MP4 gate.

## Upstream-license provenance audit (2026-07-29 UTC)

- The project `setup.py` declares the wheel license as MIT, while the copied VinRobotics XML/mesh source is Apache-2.0; recording the upstream URL and commit in a transient `/tmp` clone does not preserve redistribution attribution.
- A source-tree rollout can therefore appear compliant while the installed variant packages omit the upstream license/NOTICE beside the adapted assets and manifest.
- Recommended fix: carry the upstream Apache-2.0 license/attribution with each VinRobot asset bundle, record source file hashes plus the pinned commit in the manifest, and verify those provenance files are included in the built wheel before the gate.

## Expert critic preflight (2026-07-29 UTC)

- The unchanged aggregate checkpoint at `experiments/pre_trained_model` restores through the repository's Orbax path convention; its critic produced finite `(1, 1)` values for synthetic 12-joint/2-foot and 27-joint/4-foot inputs using the recorded 23/3, 10/2, and 20-field critic contract.
- This closes the checkpoint-side policy/critic shape preflight, but it is not rollout evidence: the VinRobot adapters, registration/preset wiring, packaged-asset smoke, termination/reward validation, and both MP4 artifacts still have to be implemented and gated.
- Recommended fix: preserve the checkpoint input contract in both adapters, then make the acceptance command instantiate each packaged variant, restore policy and critic, run finite reset-plus-rollout assertions, and emit the manifest and decodable MP4 only after those checks pass.

## Actuator-physics equivalence audit (2026-07-29 UTC)

- The pinned upstream task config uses `BuiltinPositionActuatorCfg` with per-joint stiffness, damping, effort limit, armature, and static-friction values; its action scale is derived as `0.25 * effort_limit / stiffness`. In contrast, both raw VinRobot XMLs compile with dynamic `dof_armature`, `dof_frictionloss`, and joint stiffness all equal to zero.
- A direct gear-1 motor plus explicit vector P/D controller therefore reproduces only the command conversion unless the adapter also restores the intended physical armature/friction exactly once; adding implicit position actuators as well would double-apply P/D/limits and change the expert dynamics.
- Recommended fix: choose one control path per adapter. For the repository’s direct-torque path, set name/order-derived armature and friction-loss fields, compute vector P/D torque, and clip with the source effort limits; assert zero-action equilibrium and manifest the full arrays. Do not add a second implicit position actuator path.

## Policy/critic general-mask audit (2026-07-29 UTC)

- The shared schema has 16 policy fields but 20 critic fields; the critic additionally requires `trunk_x_velocity`, `trunk_y_velocity`, `trunk_z_velocity`, and `height_0`. PPO builds these masks by exact observation name, then passes the 16-field and 20-field slices separately to the restored networks.
- An adapter that constructs only the policy context, or concatenates a 16-field array for both networks, will either fail mask construction or feed the critic shifted/missing state even though the policy forward shape remains valid.
- Recommended fix: publish one canonical full observation containing all six root velocities, relative height, commands, gravity, scalar controller context, mass, and dimensions; assert all 20 critic names and all 16 policy names resolve before PPO construction, verify their normalization/frame, and let PPO apply the masks.

## Reset/step observation-normalization audit (2026-07-29 UTC)

- The shared H1 `get_initial_observation()` inserts raw joint position offsets, raw joint velocities, and raw previous actions, while `get_observation()` later divides those fields by `4.6`, `35.0`, and `10.0`; reset height is `(qpos[2]-center_height)/robot_height`, whereas step height becomes `(height/robot_height)-1`.
- Copying that path to VinRobot gives the unchanged expert a discontinuous first observation at reset, with source-scale home offsets/velocities and a shifted height field before the first action; a shape-compatible forward pass therefore does not prove first-step safety.
- Recommended fix: centralize the checkpoint-compatible normalization and apply it identically to reset and step observations; assert the reset observation equals the normalized same-state observation with the same action/history, all fields are finite, and record the normalizers in the manifest before the rollout gate.

## Free-root quaternion/frame audit (2026-07-29 UTC)

- Both pinned MJCFs use a MuJoCo free joint, whose identity state is stored as `[w,x,y,z]=[1,0,0,0]`; the shared adapter explicitly reorders this to SciPy `[x,y,z,w]` before inverting it for root-frame velocity and projected gravity.
- A read-only VinRobot check gives projected gravity `[0,0,-1]` with that reorder but `[0,0,1]` when `qpos[3:7]` is passed directly to SciPy. The latter is finite and shape-correct yet represents an upside-down root to the policy and termination logic.
- Recommended fix: preserve the exact quaternion reorder and root/world frame conventions in both adapters; at the home reset assert identity orientation, projected gravity `[0,0,-1]`, finite body-frame velocities, and the expected command frame before accepting the first expert action.

## Reset-command timing audit (2026-07-29 UTC)

- `reset()` calls the shared random command function’s no-op `setup()` and builds the initial observation while the command fields are still zero; on the first `step()`, the `total_timesteps == 0` branch samples a new command before processing the action.
- The teacher therefore receives a zero-command observation but its first action is evaluated against a newly sampled command, creating a policy/reward/observation mismatch at the most sensitive transition; deterministic command files do not fix this ordering unless the reset observation is rebuilt.
- Recommended fix: select and store the recording command before constructing the reset observation, use that same vector for the first policy action and reward, assert observation-command equality at every step, and log the initial command plus command changes in the manifest.

## Action-history/delay semantics audit (2026-07-29 UTC)

- The shared `step()` applies `delay_action()` before P/D control, then stores that applied (possibly delayed) vector as `current_action` and `last_action`; the dynamic `previous_action` observation and action-rate reward therefore describe the applied action, not necessarily the raw PPO output.
- A VinRobot adapter that delays torque but exposes the raw teacher action in `previous_action`, or that leaves a nonzero delay path active during recording, changes the checkpoint’s recurrent action context and can destabilize the first transitions while all shapes remain valid.
- Recommended fix: pin the teacher presets to `domain_randomization_action_delay_type: none`, or reproduce the exact post-delay history semantics; log raw, delayed, applied, and previous-action vectors and assert the normalized observation matches the applied vector at every gate step.

## Foot-state reset-normalization audit (2026-07-29 UTC)

- `get_initial_observation()` emits boolean foot contact and raw touchdown timers, while `get_observation()` maps contact to `-1/1` and touchdown time through `(t/2.5)-1` with clipping. A no-contact reset is therefore `0` before the first action but `-1` immediately afterward, and a reset timer of `0` becomes `-1` on the first step.
- This is an additional dynamic-foot discontinuity that can alter the restored policy’s first action even when joint/root normalization is corrected; VinRobot’s calibrated initial contacts make the exact branch variant-dependent.
- Recommended fix: run one shared foot-state normalization for reset and step, assert contact/timer equality at the unchanged home state, and record the resolved foot aliases, contact booleans, timers, and normalized values before accepting a rollout.


## Current implementation status (2026-07-30 UTC)

- Added distinct attributed `vr_m3_1_12dof` and `vr_m3_1_full` packages from pinned VinRobotics commit `92238af819c263abaa0f1d1e02e467bf70d6d902` (Apache-2.0), with adapted MJCFs, separate classes/registry entries, termination keys, recording choices, and named expert presets. The models compile as `nq=19,nv=18,nu=12` and `nq=34,nv=33,nu=27`; wheel-content verification remains for the final packaging gate.
- Both variants restore the unchanged `experiments/pre_trained_model` checkpoint (SHA-256 `e851b6c4b2d1da55050d5ba1500cc893073795e9a4b83dad2abc6ce58e489c60`) and emit finite actions/videos. Presets now explicitly pass `environment.episode_length_in_seconds: 20`, `record_seconds: 10`, teacher rollout, one VinRobot train environment, original robots as sizing-only eval types, and run-scoped artifacts under `dev/codex/artifacts/vr_m3_1/20260730T194945Z/`. Ambient `experiments/commands.txt` input is disabled in both adapters.
- The 12-DoF path currently uses the upstream vector P/D scale `0.25 * effort_limit / stiffness` and deterministic `[0.2,0,0]` test command. Earlier short recordings exist, but no fresh uninterrupted 10-second run with the explicit 20-second horizon has been executed/accepted; rerun this only after resolving the full path, then validate 500 frames at 50 FPS plus first/middle/final frames.
- The full path is the active blocker. It retains all 27 physical joints/actions for checkpoint sizing but locks actions `12:` (waist/arms) at home, uses `0.0625 * effort_limit / stiffness` for leg offsets, and currently has a zero test command. Explicit diagnostics proved prior apparent step-100 resets were not horizon truncations: horizon is 1,000 steps. Every attempted full configuration still physically terminated below the `0.545` height threshold: locked upper at scale 0.0625/command 0.2 fell at step 182; scale 0.04/command 0.2 at step 134; all-27 actions at 0.0625/command 0.2 at step 158; locked upper at command 0.05 at step 117; locked upper at zero command at step 109. The last background run was stopped; no process should be assumed active and no full MP4 passes. Temporary `VRM31Full motion/done` prints remain in `environment.py` and must be removed after debugging.
- Resume at checklist item 4/6: correct only the full embodiment adapter (mapping, normalization/reset-command timing, PD/torque clipping, contacts/initial state, or locked-joint handling), then run the failed full preset until one uninterrupted 500-frame/10-second video passes without reset. Next run the 12-DoF 10-second gate, inspect both videos, remove temporary diagnostics/unused `forward.py` helpers and copied template cruft, perform focused compile/print-only/package checks, write `manifest.json` with mappings/controller arrays/commands/logs/configs/video metadata/provenance, and check items 2-7/completion evidence only when objectively satisfied. Artifact directory contains many failed diagnostic MP4s; select only final validated filenames in the manifest.

