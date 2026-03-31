from copy import deepcopy


BASE_EXPERIMENT = {
    "algorithm.name": "uni_ppo.ppo",
    "environment.name": "multi_robot",
    "runner.track_console": True,
    "runner.load_model": "pre_trained_model",
    "algorithm.determine_fastest_cpu_for_gpu": False,
    "algorithm.nr_epochs": 1,
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


PRESETS = {
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
            "algorithm.snn_enabled": True,
            "algorithm.rollout_policy_stage": "snn",
            "algorithm.snn_threshold": 0.2,
            "algorithm.snn_timesteps": 5,
            "algorithm.snn_convert_every_iteration": True,
            "algorithm.student_model_path": "/app/one_policy_to_run_them_all/student/student_model_best.pth",
            "algorithm.student_checkpoint_dir": "/app/one_policy_to_run_them_all/experiments/teacher-student/online_dagger_snn",
            "algorithm.student_dataset_path": "/app/one_policy_to_run_them_all/experiments/teacher-student/online_dagger_snn/teacher_student_dagger_dataset.npz",
            "algorithm.snn_export_dir": "/app/one_policy_to_run_them_all/experiments/teacher-student/online_dagger_snn/snn_exports",
        },
    ),
    "test_student": build_preset(
        **{
            "environment.multi_render": True,
            "algorithm.use_student": True,
        },
    ),
    "test_snn": build_preset(
        **{
            "environment.multi_render": True,
            "algorithm.snn_enabled": True,
            "algorithm.rollout_policy_stage": "snn",
            "algorithm.snn_threshold": 0.2,
            "algorithm.snn_timesteps": 5,
            "algorithm.student_model_path": "/app/one_policy_to_run_them_all/student/student_model_best.pth",
        },
    ),
    "test_student_d300k": build_preset(
        **{
            "environment.multi_render": True,
            "algorithm.use_student": True,
            "algorithm.student_model_path": "/app/one_policy_to_run_them_all/experiments/teacher-student/conversion_target_search_d300k/1024x1024x1024x1024x1024/student_model_best.pth",
        },
    ),
    "test_snn_d300k": build_preset(
        **{
            "environment.multi_render": True,
            "algorithm.snn_enabled": True,
            "algorithm.rollout_policy_stage": "snn",
            "algorithm.snn_threshold": 0.2,
            "algorithm.snn_timesteps": 3,
            "algorithm.student_model_path": "/app/one_policy_to_run_them_all/experiments/teacher-student/conversion_target_search_d300k/1024x1024x1024x1024x1024/student_model_best.pth",
        },
    ),
}
