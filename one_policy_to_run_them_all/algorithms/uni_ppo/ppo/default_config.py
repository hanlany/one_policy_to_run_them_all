from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"  # cpu, gpu
    config.total_timesteps = 1e9
    config.start_learning_rate = 3e-4
    config.end_learning_rate = 3e-4
    config.nr_steps = 2048
    config.nr_epochs = 10
    config.minibatch_size = 64
    config.gamma = 0.99
    config.gae_lambda = 0.95
    config.clip_range = 0.2
    config.softmax_temperature = 1.0
    config.softmax_temperature_min = 0.015
    config.stability_epsilon = 1e-8
    config.missing_value = 0.0
    config.entropy_coef = 0.0
    config.critic_coef = 0.5
    config.max_grad_norm = 0.5
    config.std_dev = 1.0
    config.policy_mean_abs_clip = 10.0
    config.policy_std_min_clip = 1e-8
    config.policy_std_max_clip = 2.0
    config.nr_hidden_units = 256
    config.evaluation_frequency = 204800  # -1 to disable
    config.evaluation_episodes = 10
    config.save_latest_frequency = 204800
    config.determine_fastest_cpu_for_gpu = False

    config.save_data = False
    config.data_points = 1000000
    config.use_student = False
    config.dagger_style = False
    config.dagger_online = 0

    config.student_model_path = "/app/one_policy_to_run_them_all/student/student_model_best.pth"
    config.student_checkpoint_dir = "/app/one_policy_to_run_them_all/student"
    config.student_dataset_path = "/app/one_policy_to_run_them_all/student/teacher_student_dagger_dataset.npz"
    config.student_hidden_dims = [1024, 1024, 1024, 1024, 1024]
    config.student_batch_size = 64
    config.student_learning_rate = 1e-4
    config.student_train_epochs = 60
    config.student_num_workers = 0
    config.rollout_policy_stage = "student"
    config.snn_enabled = False
    config.snn_threshold = 0.2
    config.snn_timesteps = 3
    config.snn_convert_every_iteration = True
    config.snn_export_dir = "/app/one_policy_to_run_them_all/student/snn_exports"

    return config
