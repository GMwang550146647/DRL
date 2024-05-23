import torch

ParamsFuturesSAC = {
    "critic_lr": 1e-2,
    "actor_lr": 1e-3,
    "alpha_lr": 1e-2,
    "tau": 0.2,
    "gamma": 0.98,
    "target_entropy": -1,
    "state_dim": 8,
    "action_dim": 3,
    "hidden_dim": 256,
    "target_update": 10,
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
}
