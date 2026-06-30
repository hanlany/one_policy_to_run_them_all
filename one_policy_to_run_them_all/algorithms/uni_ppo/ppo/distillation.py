from student.train_student import (
    DEFAULT_HIDDEN_DIMS,
    BootstrapTrainingConfig,
    PolicyDistillationPipeline,
    SNNConversionConfig,
    StudentTrainingConfig,
)


class PolicyStageController:
    def __init__(self, teacher_predict, initial_stage="teacher"):
        self.teacher_predict = teacher_predict
        self.stage = initial_stage
        self.student_predict = None
        self.snn_predict = None

    def register_student(self, predict_fn):
        self.student_predict = predict_fn

    def register_snn(self, predict_fn):
        self.snn_predict = predict_fn

    def set_stage(self, stage):
        self.stage = stage

    def predict(self, policy_state, state, env_id=None, role="behavior"):
        if role == "teacher_label":
            return self.teacher_predict(policy_state, state, env_id)

        if self.stage == "snn" and self.snn_predict is not None:
            return self.snn_predict(state)
        if self.stage in {"student", "snn"} and self.student_predict is not None:
            return self.student_predict(state)
        return self.teacher_predict(policy_state, state, env_id)


def get_online_dagger_expert_ratio(iteration_index):
    if iteration_index < 0:
        raise ValueError("DAgger iteration index must be non-negative.")
    if iteration_index == 0:
        return 1.0
    if iteration_index == 1:
        return 0.5
    if iteration_index == 2:
        return 0.25
    return 0.0
