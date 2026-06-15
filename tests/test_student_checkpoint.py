import torch

from student.train_student import StudentCheckpointManager, StudentPolicy, infer_student_architecture_from_state_dict


def test_student_checkpoint_round_trip(tmp_path):
    model = StudentPolicy(input_dim=3, output_dim=2, hidden_dims=[4])
    checkpoint_path = tmp_path / "student_model_best.pth"
    manager = StudentCheckpointManager(tmp_path)

    manager.save(model, checkpoint_path, extra_metadata={"iteration": 7})
    loaded_model = manager.load(checkpoint_path)

    assert loaded_model.get_architecture_config() == {
        "input_dim": 3,
        "output_dim": 2,
        "hidden_dims": [4],
    }
    observation = torch.ones(1, 3)
    torch.testing.assert_close(loaded_model(observation), model(observation))


def test_infer_student_architecture_from_synthetic_state_dict():
    state_dict = {
        "net.0.weight": torch.zeros(4, 3),
        "net.0.bias": torch.zeros(4),
        "net.2.weight": torch.zeros(5, 4),
        "net.2.bias": torch.zeros(5),
        "net.4.weight": torch.zeros(2, 5),
        "net.4.bias": torch.zeros(2),
    }

    assert infer_student_architecture_from_state_dict(state_dict) == (3, 2, [4, 5])
