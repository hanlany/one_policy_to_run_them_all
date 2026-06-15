import numpy as np
import pytest

from one_policy_to_run_them_all.algorithms.uni_ppo.ppo.distillation import PolicyStageController
from one_policy_to_run_them_all.algorithms.uni_ppo.ppo.observation_schema import build_general_state_mask
from one_policy_to_run_them_all.algorithms.uni_ppo.ppo.recording import (
    build_recording_path,
    get_record_env_ids,
    get_record_timing,
    sanitize_video_fragment,
)


def test_policy_stage_controller_routes_teacher_labels_to_teacher():
    calls = []

    def teacher(policy_state, state, env_id=None):
        calls.append((policy_state, state, env_id))
        return "teacher"

    controller = PolicyStageController(teacher, initial_stage="student")
    controller.register_student(lambda state: "student")

    assert controller.predict("policy", "state", env_id=2, role="teacher_label") == "teacher"
    assert calls == [("policy", "state", 2)]


def test_policy_stage_controller_prefers_snn_then_student_then_teacher():
    controller = PolicyStageController(lambda policy_state, state, env_id=None: "teacher")

    controller.set_stage("student")
    assert controller.predict(None, "state") == "teacher"

    controller.register_student(lambda state: "student")
    assert controller.predict(None, "state") == "student"

    controller.set_stage("snn")
    assert controller.predict(None, "state") == "student"

    controller.register_snn(lambda state: "snn")
    assert controller.predict(None, "state") == "snn"


def test_build_general_state_mask_selects_named_columns_per_env():
    observation_name_to_ids = [
        {"a": 0, "b": 2},
        {"a": 1, "b": 3},
    ]

    mask = build_general_state_mask(observation_name_to_ids, (2, 4), ("a", "b"))

    expected = np.array(
        [
            [True, False, True, False],
            [False, True, False, True],
        ]
    )
    np.testing.assert_array_equal(mask, expected)
    assert mask.dtype == bool


def test_recording_helpers_sanitize_paths_and_select_envs():
    assert sanitize_video_fragment("stage/snn v1") == "stage_snn_v1"
    assert build_recording_path("videos", "snn v1", 3, "unitree/go1", "ts").endswith(
        "ts_snn_v1_env03_unitree_go1.mp4"
    )
    assert get_record_env_ids(-1, 3) == [0, 1, 2]
    assert get_record_env_ids(1, 3) == [1]

    with pytest.raises(ValueError, match="out of range"):
        get_record_env_ids(3, 3)


def test_record_timing_uses_environment_dt():
    assert get_record_timing(2.0, 0.5) == (2, 4)
    assert get_record_timing(0.0, 0.5) == (2, 1)

    with pytest.raises(ValueError, match="Invalid environment dt"):
        get_record_timing(1.0, 0.0)



def test_ppo_default_student_paths_use_project_path():
    from one_policy_to_run_them_all.algorithms.uni_ppo.ppo.default_config import get_config
    from one_policy_to_run_them_all.paths import project_path

    config = get_config("uni_ppo.ppo")

    assert config.student_model_path == str(project_path("student", "student_model_best.pth"))
    assert config.student_checkpoint_dir == str(project_path("student"))
    assert config.student_dataset_path == str(project_path("student", "teacher_student_dagger_dataset.npz"))
    assert config.snn_export_dir == str(project_path("student", "snn_exports"))
    assert config.record_dir == str(project_path("experiments", "videos"))
