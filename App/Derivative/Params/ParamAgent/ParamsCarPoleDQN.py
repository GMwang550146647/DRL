import torch

ParamsCarPoleDQN = {
    "learning_rate": 2e-3,
    "gamma": 0.98,
    "epsilon": 0.01,
    "state_dim": 4,
    "action_dim": 2,
    "hidden_dim": 128,
    "target_update": 10,
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
}
