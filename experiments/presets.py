from copy import deepcopy


RECORD_ROBOTS = (
    "unitree_a1",
    "unitree_go1",
    "unitree_go2",
    "anymal_b",
    "anymal_c",
    "barkour_v0",
    "barkour_vb",
    "badger",
    "bittle",
    "unitree_h1",
    "unitree_g1",
    "talos",
    "robotis_op3",
    "nao_v5",
    "cassie",
    "hexapod",
)

STUDENT_MODEL_PATH = "/app/one_policy_to_run_them_all/student/student_model_best.pth"
D300K_STUDENT_MODEL_PATH = (
    "/app/one_policy_to_run_them_all/experiments/teacher-student/"
    "conversion_target_search_d300k/1024x1024x1024x1024x1024/student_model_best.pth"
)
ONLINE_DAGGER_SNN_DIR = "/app/one_policy_to_run_them_all/experiments/teacher-student/online_dagger_snn"
BOOTSTRAP_MODEL_PATH = "/app/one_policy_to_run_them_all/experiments/teacher-student/bootstrap_parity/student_model_best.pth"
ONLINE_DAGGER_BOOTSTRAP_DIR = "/app/one_policy_to_run_them_all/experiments/teacher-student/online_dagger_bootstrap"

DEFAULT_SNN_OVERRIDES = {
    "algorithm.snn_enabled": True,
    "algorithm.rollout_policy_stage": "snn",
    "algorithm.snn_threshold": 0.2,
}

DEFAULT_BOOTSTRAP_OVERRIDES = {
    "algorithm.student_backend": "bootstrap",
    "algorithm.bootstrap_timesteps": 3,
    "algorithm.bootstrap_readout": "mean",
}

DEFAULT_RECORD_ROBOT = "unitree_h1"

DEFAULT_RECORD_OVERRIDES = {
    "environment.nr_envs": 1,
    "environment.train_robot_types": (DEFAULT_RECORD_ROBOT,),
    "environment.multi_render": False,
    "algorithm.record": True,
    "algorithm.record_robot_index": 0,
}


def get_record_robot(robot_index):
    try:
        return RECORD_ROBOTS[int(robot_index)]
    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"Invalid record robot index {robot_index}. Use an index from 0 to {len(RECORD_ROBOTS) - 1}."
        ) from exc


BASE_EXPERIMENT = {
    "algorithm.name": "uni_ppo.ppo",
    "environment.name": "multi_robot",
    "runner.track_console": True,
    "runner.load_model": "pre_trained_model",
    "algorithm.determine_fastest_cpu_for_gpu": False,
    "algorithm.nr_epochs": 1,
    "algorithm.record": False,
    "runner.mode": "test",
    "environment.mode": "test",
    "environment.add_goal_arrow": True,
    "environment.nr_envs": 16,
    "environment.render": False,
}


def build_preset(**overrides):
    preset = deepcopy(BASE_EXPERIMENT)
    preset.update(overrides)
    return preset


def build_student_test_preset(student_model_path, *, multi_render=True, record=False):
    overrides = {
        "environment.multi_render": multi_render,
        "algorithm.use_student": True,
        "algorithm.student_model_path": student_model_path,
    }
    if record:
        overrides.update(DEFAULT_RECORD_OVERRIDES)
    return build_preset(**overrides)


def build_snn_test_preset(student_model_path, *, snn_timesteps, multi_render=True, record=False):
    overrides = {
        "environment.multi_render": multi_render,
        "algorithm.student_model_path": student_model_path,
        "algorithm.snn_timesteps": snn_timesteps,
        **DEFAULT_SNN_OVERRIDES,
    }
    if record:
        overrides.update(DEFAULT_RECORD_OVERRIDES)
    return build_preset(**overrides)


def build_bootstrap_student_test_preset(student_model_path, *, multi_render=True, record=False):
    overrides = {
        "environment.multi_render": multi_render,
        "algorithm.use_student": True,
        "algorithm.student_model_path": student_model_path,
        **DEFAULT_BOOTSTRAP_OVERRIDES,
    }
    if record:
        overrides.update(DEFAULT_RECORD_OVERRIDES)
    return build_preset(**overrides)


def build_bootstrap_snn_test_preset(student_model_path, *, bootstrap_timesteps, multi_render=True, record=False):
    overrides = {
        "environment.multi_render": multi_render,
        "algorithm.student_model_path": student_model_path,
        "algorithm.rollout_policy_stage": "snn",
        "algorithm.bootstrap_timesteps": bootstrap_timesteps,
        **DEFAULT_BOOTSTRAP_OVERRIDES,
    }
    if record:
        overrides.update(DEFAULT_RECORD_OVERRIDES)
    return build_preset(**overrides)


PRESETS = {
    "test_default": build_preset(
        **{
            "environment.multi_render": True,
        },
    ),
    "record_default": build_preset(
        **DEFAULT_RECORD_OVERRIDES,
    ),
    "collect_data_dagger": build_preset(
        **{
            "environment.multi_render": False,
            "algorithm.save_data": True,
            "algorithm.data_points": 300000,
            "algorithm.dagger_style": True,
        },
    ),
    "train_dagger": build_preset(
        **{
            "environment.multi_render": False,
            "algorithm.save_data": True,
            "algorithm.data_points": 20000,
            "algorithm.dagger_style": True,
            "algorithm.dagger_online": 50,
        },
    ),
    "train_dagger_snn": build_preset(
        **{
            "environment.multi_render": False,
            "algorithm.data_points": 20000,
            "algorithm.dagger_online": 50,
            "algorithm.snn_timesteps": 5,
            "algorithm.snn_convert_every_iteration": True,
            "algorithm.student_model_path": STUDENT_MODEL_PATH,
            "algorithm.student_checkpoint_dir": ONLINE_DAGGER_SNN_DIR,
            "algorithm.student_dataset_path": f"{ONLINE_DAGGER_SNN_DIR}/teacher_student_dagger_dataset.npz",
            "algorithm.snn_export_dir": f"{ONLINE_DAGGER_SNN_DIR}/snn_exports",
            **DEFAULT_SNN_OVERRIDES,
        },
    ),
    "train_dagger_bootstrap": build_preset(
        **{
            "environment.multi_render": False,
            "algorithm.data_points": 20000,
            "algorithm.dagger_online": 50,
            "algorithm.student_model_path": BOOTSTRAP_MODEL_PATH,
            "algorithm.student_checkpoint_dir": ONLINE_DAGGER_BOOTSTRAP_DIR,
            "algorithm.student_dataset_path": f"{ONLINE_DAGGER_BOOTSTRAP_DIR}/teacher_student_dagger_dataset.npz",
            **DEFAULT_BOOTSTRAP_OVERRIDES,
        },
    ),
    "test_student": build_student_test_preset(STUDENT_MODEL_PATH),
    "test_snn": build_snn_test_preset(STUDENT_MODEL_PATH, snn_timesteps=5),
    "test_student_d300k": build_student_test_preset(D300K_STUDENT_MODEL_PATH),
    "test_snn_d300k": build_snn_test_preset(D300K_STUDENT_MODEL_PATH, snn_timesteps=3),
    "test_bootstrap_student": build_bootstrap_student_test_preset(BOOTSTRAP_MODEL_PATH),
    "test_bootstrap_snn": build_bootstrap_snn_test_preset(BOOTSTRAP_MODEL_PATH, bootstrap_timesteps=3),
    "record_bootstrap_student": build_bootstrap_student_test_preset(
        BOOTSTRAP_MODEL_PATH,
        multi_render=True,
        record=True,
    ),
    "record_bootstrap_snn": build_bootstrap_snn_test_preset(
        BOOTSTRAP_MODEL_PATH,
        bootstrap_timesteps=3,
        multi_render=True,
        record=True,
    ),
    "record_student_d300k": build_student_test_preset(
        D300K_STUDENT_MODEL_PATH,
        multi_render=True,
        record=True,
    ),
    "record_snn_d300k": build_snn_test_preset(
        D300K_STUDENT_MODEL_PATH,
        snn_timesteps=3,
        multi_render=True,
        record=True,
    ),
}
