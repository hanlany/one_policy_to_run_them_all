# Independent Codex Agent Brief: Optimization And Cleanup

Use this file as the complete handoff brief for an independent Codex agent working on the `one_policy_to_run_them_all` repository. The agent should treat the critique below as context, but should execute changes incrementally and verify each step.

Audit date: 2026-06-15

Scope: `one_policy_to_run_them_all` only. I sampled the experiment scripts, PPO implementation, student distillation code, packaging, multi-robot factory, and representative robot environments.

## Agent Mission

Improve maintainability, portability, and safety of the repository without changing training semantics or robot behavior unless a change is explicitly covered by tests or parity checks.

Primary goal: make the codebase easier to install, run, test, and refactor safely.

Secondary goal: prepare the codebase for larger architecture cleanup, especially PPO decomposition, student distillation separation, and robot environment deduplication.

Do not start with large behavioral rewrites. Build guardrails first, then refactor behind those guardrails.

## Required Operating Rules

- Work inside `/app/one_policy_to_run_them_all`.
- Before editing, run `git status --short` and inspect any files you plan to touch.
- Do not revert or overwrite user changes.
- Prefer small, reviewable commits or change batches.
- Preserve existing public command behavior unless this document explicitly asks for a migration.
- Do not delete robot assets, checkpoints, datasets, videos, or result files unless the user explicitly asks for cleanup of tracked artifacts.
- Avoid broad formatting-only edits across generated, vendored, or robot asset files.
- When refactoring numerical code, add parity checks before or alongside the refactor.
- Treat MuJoCo/JAX/Torch behavior as high risk. Keep changes narrow and verify imports/tests after every phase.

## First Task For The Agent

Start with **Phase 1: Safe Infrastructure Cleanup**. The first useful change set should be intentionally small:

1. Fix Python packaging so subpackages are included.
2. Add or improve lightweight tests that do not require a full training run.
3. Add a central path helper only if needed by the touched code.
4. Tighten ignore rules for generated artifacts without deleting existing files.
5. Document what was changed and what was deliberately deferred.

Do not begin with `ppo.py` decomposition or robot base-class extraction. Those are later phases after tests exist.

## Suggested Initial Implementation Plan

### Step 1: Baseline Inspection

Run:

```bash
git status --short
find . -maxdepth 3 -type f \( -name "pyproject.toml" -o -name "setup.cfg" -o -name "tox.ini" -o -name ".pre-commit-config.yaml" \) -print
find . -maxdepth 4 -type f \( -name "test_*.py" -o -name "*_test.py" \) -print
```

Read:

- `setup.py`
- `requirements.txt`
- `experiments/run.py`
- `experiments/presets.py`
- `student/train_student.py`
- `student/bootstrap_backend.py`
- `one_policy_to_run_them_all/environments/multi_robot/create_env.py`
- `one_policy_to_run_them_all/environments/multi_robot/robot_helper.py`

### Step 2: Packaging Fix

Expected direction:

- Replace the single-package list in `setup.py` with `setuptools.find_packages()`.
- Include non-Python environment assets through package data or `MANIFEST.in` if packaging tests reveal missing XML/mesh files.
- Keep this backwards compatible with editable installs.

Minimal success criteria:

- `python -c "import one_policy_to_run_them_all; import one_policy_to_run_them_all.environments.multi_robot.create_env"` works from the repo.
- A non-editable install path is considered or tested if dependencies permit it.

### Step 3: Lightweight Tests

Add tests that avoid GPU, MuJoCo rendering, full training, and network access.

Good first tests:

- `experiments.run.build_command()` resolves every preset.
- `--record-robot` maps a robot index to a single-robot recording command.
- `StudentCheckpointManager` can save/load a tiny `StudentPolicy`.
- `infer_student_architecture_from_state_dict()` works on a tiny synthetic state dict.
- Import smoke test for `one_policy_to_run_them_all.environments.multi_robot.robot_helper`.

Use `pytest` if available. If test dependencies are missing, add a clear note and provide import-level smoke commands.

### Step 4: Ignore Rules

Review `.gitignore` and add patterns for generated artifacts if missing:

```gitignore
experiments/videos/
experiments/log/
student/snn_exports/
*.pth
*.npz
*.hdf5
dev/bash_history
```

Do not remove existing tracked files unless the user asks. If a generated artifact is already tracked, mention it in the final report.

### Step 5: Validation

Run the narrowest available checks:

```bash
python -m pytest tests -q
python -c "import one_policy_to_run_them_all; import one_policy_to_run_them_all.environments.multi_robot.robot_helper"
python experiments/run.py --preset test_default --print-only
```

If these fail because heavy dependencies are unavailable, report the exact missing dependency and keep any pure-Python tests passing.

## Deliverables Expected From The Agent

At the end of each change batch, report:

- Files changed.
- What behavior was intentionally preserved.
- Tests or smoke checks run.
- Any checks that could not run and why.
- Recommended next phase.

## Stop Conditions

Stop and ask the user before:

- Deleting tracked data/model/video artifacts.
- Renaming public experiment presets or robot names.
- Changing reward functions, observations, action scaling, termination behavior, or model architecture.
- Performing a mass robot-environment rewrite.
- Introducing a new dependency that requires network install.

## Definition Of Done For Phase 1

Phase 1 is done when:

- Packaging includes the actual subpackages.
- At least a few fast tests or smoke checks cover presets/imports/checkpoint metadata.
- Generated artifact ignore rules are in place or documented.
- No training semantics, robot dynamics, rewards, observations, or policy outputs were intentionally changed.
- The final report clearly lists residual risks.

# Supporting Audit: Codebase Critique And Improvement Plan

## Executive Summary

This repository is valuable research code with a clear scientific goal, but it has grown by cloning patterns instead of extracting stable interfaces. The main risks are reproducibility drift, path/environment coupling, very large mixed-responsibility modules, and duplicated robot simulation code.

The highest-leverage cleanup is not micro-optimization. It is separating concerns:

1. Make packaging and paths portable.
2. Move experiment presets/configuration out of Python constants and absolute paths.
3. Split the 1,380-line PPO module into training, rollout, recording, student distillation, checkpointing, and evaluation components.
4. Replace per-robot copy-pasted environment classes with a shared base class plus robot specs.
5. Add fast smoke tests around config resolution, factories, observation shapes, checkpoint loading, and preset command generation.

## Current Strengths

- The project has a consistent conceptual layout: algorithms, environments, experiments, and student distillation are recognizable.
- The per-robot folder convention is easy to discover.
- `experiments/run.py` provides a useful named-preset entrypoint instead of requiring users to remember long CLI overrides.
- The multi-robot environment factory already centralizes robot selection through `multi_robot/robot_helper.py`.
- The student distillation code uses dataclasses for several configuration objects, which is a good direction.

## Main Problems

### 1. Packaging Is Incomplete

`setup.py` declares:

```python
packages=["one_policy_to_run_them_all"]
```

That excludes subpackages such as `one_policy_to_run_them_all.environments`, `one_policy_to_run_them_all.algorithms`, and their nested modules in a normal wheel/sdist install. Editable installs can mask this locally, but proper packaging will be fragile.

Recommended changes:

- Replace `packages=["one_policy_to_run_them_all"]` with `find_packages()`.
- Move packaging metadata to `pyproject.toml`.
- Include XML/mesh assets explicitly through package data or `MANIFEST.in`.
- Add a packaging smoke test: install into a clean venv and import `one_policy_to_run_them_all.environments.multi_robot.create_env`.

### 2. The Code Is Hard-Coupled To `/app`

Absolute paths appear in configuration and scripts, for example:

- `experiments/presets.py`
- `one_policy_to_run_them_all/algorithms/uni_ppo/ppo/default_config.py`
- shell scripts in `experiments/*.sh`
- experiment result JSON files

This makes the code difficult to run outside this container/workspace and makes tests brittle.

Recommended changes:

- Define a single project-root resolver, for example `paths.py`, based on `Path(__file__).resolve()`.
- Allow overrides through environment variables such as `OPTRA_ROOT`, `OPTRA_DATA_DIR`, and `OPTRA_ARTIFACT_DIR`.
- Store paths in config as relative paths where possible.
- Make shell scripts call module entrypoints, for example `python -m one_policy_to_run_them_all.experiments.run`, after moving experiments into the package.

### 3. `ppo.py` Is A God Module

`one_policy_to_run_them_all/algorithms/uni_ppo/ppo/ppo.py` is 1,380 lines and mixes:

- PPO training loop
- JAX model state setup
- rollout/evaluation
- video recording
- MuJoCo offscreen rendering
- DAgger data collection
- Torch student inference
- SNN conversion/training integration
- checkpointing
- CPU affinity
- W&B/logging

This makes behavior changes risky: a video recording change can accidentally affect PPO training or student rollout behavior.

Recommended split:

- `ppo/trainer.py`: PPO update loop and train state.
- `ppo/rollout.py`: rollout collection, advantage computation, batching.
- `ppo/evaluation.py`: evaluation episodes and metrics.
- `ppo/recording.py`: `VideoFrameWriter`, `OffscreenMujocoVideoRecorder`.
- `ppo/distillation.py`: student/teacher stage controller, DAgger hooks, SNN conversion hooks.
- `ppo/checkpointing.py`: Orbax save/load utilities.
- `ppo/observation_schema.py`: masks and observation slicing.

Keep `PPO` as a thin orchestrator until call sites are migrated.

### 4. Robot Environments Are Mostly Copy-Paste

There are 20 robot environment directories. Many `environment.py` files are around 650 lines and share the same step/reset/observation/domain-randomization logic. The main differences are robot constants, XML paths, collision groups, joint names, foot names, and a few robot-specific masks.

This design increases maintenance cost: a bug in observation normalization, command handling, collision checks, or domain randomization must be patched in many places.

Recommended changes:

- Introduce `BaseMujocoRobotEnv`.
- Move robot-specific data into `RobotSpec` dataclasses:
  - names: long/short name
  - XML file and terrain defaults
  - nominal joint positions
  - max joint velocities
  - foot names
  - joint names
  - collision groups
  - initial drop height
  - optional joint masks
- Make per-robot `environment.py` files only define `RobotSpec` and subclass/instantiate the base env.
- Extract shared `viewer.py`, wrappers, handler factories, and no-op/default functions.

This is the single biggest maintainability win in the repo.

### 5. Handler Factories Are Repeated And Stringly Typed

Many folders define handlers like:

```python
if name == "default":
    return DefaultThing(...)
elif name == "none":
    return NoneThing(...)
else:
    raise NotImplementedError
```

This pattern is repeated across robots and feature types.

Recommended changes:

- Replace `if/elif` handlers with registry dictionaries.
- Raise `ValueError(f"Unknown ...: {name}. Available: {sorted(REGISTRY)}")`.
- Move common `none`, `default`, and sampling implementations into shared modules.
- Add tests that all configured handler names resolve.

### 6. Configuration Is Spread Across Python, Shell, And CLI Overrides

Configuration currently lives in:

- `default_config.py` files
- `experiments/presets.py`
- shell scripts
- hardcoded defaults in student dataclasses
- direct CLI overrides

This makes experiments hard to reproduce because the effective config is assembled from several places.

Recommended changes:

- Introduce structured experiment config files in YAML/TOML.
- Keep Python preset builders only as compatibility wrappers.
- Save the fully resolved config to each run directory.
- Validate config at startup with clear errors.
- Use enums/literals for fields such as `student_backend`, `rollout_policy_stage`, `readout`, and handler names.

### 7. Student Distillation Needs Separation From Runtime PPO

`student/train_student.py` is 1,237 lines and `student/bootstrap_backend.py` is 889 lines. There is also duplicated dataset/checkpoint logic between them.

Recommended changes:

- Create a package: `one_policy_to_run_them_all/student/`.
- Split into:
  - `datasets.py`
  - `models.py`
  - `checkpoints.py`
  - `trainers/ann.py`
  - `trainers/bootstrap.py`
  - `conversion/snn.py`
  - `metrics.py`
  - `cli.py`
- Remove `sys.path.append(...)` imports.
- Share `TeacherStudentDataset`, checkpoint metadata, and dataset loading code.
- Gate optional Lava imports behind explicit backend selection so normal PPO import does not touch SNN dependencies.

### 8. Performance Risks Are Hidden In Python Loops And Copies

Some likely hotspots:

- Rebuilding observations by copying `initial_observation` every step.
- Repeated collision group checks through Python loops over MuJoCo contacts.
- `MultiSingleVectorEnv.step_wait()` uses `deepcopy`.
- `AsyncVectorEnvWithSkipping.step_wait()` polls pipes in a tight loop and returns existing `self.observations` even when some envs are skipped.
- The PPO module crosses JAX, NumPy, Torch, OpenCV, MuJoCo, and Python control flow in one place, making profiling noisy.

Recommended changes:

- Add a small benchmark suite: env reset, env step, vector step, PPO rollout collection.
- Replace `deepcopy` with explicit `np.copy` only where needed.
- Cache collision lookups and observation index arrays in shared base env initialization.
- Profile before changing numerical code; preserve reward and observation parity with regression tests.
- Keep JAX-critical paths free of Torch/OpenCV/MuJoCo recording imports.

### 9. Tests Are Too Thin For The Blast Radius

Only two Python files look test-like in the top-level experiment folder:

- `experiments/test_bootstrap_parity.py`
- `experiments/test_snn_conversion_loss.py`

They are closer to experiment validation scripts than general CI tests.

Recommended test additions:

- Import smoke tests for all packages after a non-editable install.
- Config tests for every preset in `experiments/presets.py`.
- Handler-resolution tests for every robot and function category.
- Environment construction tests with `render=False` and `mode="test"` for one cheap robot and one multi-robot config.
- Observation schema tests: shape, finite values after reset/step, expected padding.
- Student checkpoint tests: save/load round trip with tiny models.
- `experiments/run.py --print-only` tests to guarantee generated commands stay stable.

### 10. Repository Hygiene Needs Boundaries

The repo contains generated/runtime artifacts such as:

- `experiments/videos/*.mp4`
- `student/snn_exports/runtime_snn.hdf5`
- experiment result JSON files
- `dev/bash_history`

The repository is about 419 MB in this checkout, mostly because it includes many robot assets. Some assets are necessary, but runtime outputs should not live beside source by default.

Recommended changes:

- Add or tighten `.gitignore` for videos, checkpoints, datasets, generated SNN exports, logs, and local shell history.
- Move experiment artifacts to `artifacts/` or an external run directory.
- Keep small reference result files only when they are used by tests or documentation.
- Document asset licensing and provenance per robot.

## Prioritized Roadmap

### Phase 1: Safe Infrastructure Cleanup

- Replace `setup.py` package list with `find_packages()` and add package data.
- Add `pyproject.toml` with formatting/test tooling.
- Add `.gitignore` entries for generated artifacts.
- Create a central `paths.py` and remove `/app/...` from new code.
- Add smoke tests for imports, preset command generation, and student checkpoint round trip.

### Phase 2: Configuration And Entrypoints

- Convert experiment presets to data files.
- Save resolved configs for each run.
- Replace shell scripts with thin wrappers around package module entrypoints.
- Validate handler/config names at startup.

### Phase 3: PPO Decomposition

- Move video recording classes out of `ppo.py`.
- Move student/DAgger/SNN code out of `ppo.py`.
- Extract observation schema/mask construction.
- Add regression tests around action prediction, rollout stage switching, and data collection.

### Phase 4: Environment Deduplication

- Build `BaseMujocoRobotEnv` from one representative robot.
- Migrate one simple robot first.
- Add parity tests comparing old and new reset observation shape, one step output, and core metadata.
- Migrate remaining robots incrementally.

### Phase 5: Performance Work

- Add benchmarks before optimizing.
- Profile env step and vector env stepping.
- Remove avoidable copies and repeated Python work.
- Only then tune MuJoCo/JAX/Torch boundaries.

## Suggested First Pull Request

Start small and boring:

1. Add `pyproject.toml` with `pytest`, `ruff`, and formatting configuration.
2. Fix packaging with `find_packages()`.
3. Add `dev/codex` to documentation space if desired, and ignore generated videos/checkpoints/logs.
4. Add tests for:
   - `experiments.run.build_command`
   - all preset names resolving
   - `StudentCheckpointManager` save/load round trip
   - import of `one_policy_to_run_them_all.environments.multi_robot.create_env`

This gives the repo guardrails before touching the risky training and simulator code.

## Bottom Line

The codebase is doing too much in too few abstractions. The core idea is strong, but implementation velocity is now being taxed by copy-paste robot environments, hardcoded paths, and giant modules. The cleanup should focus first on portability and tests, then on carving stable interfaces around PPO, student distillation, and robot simulation.
