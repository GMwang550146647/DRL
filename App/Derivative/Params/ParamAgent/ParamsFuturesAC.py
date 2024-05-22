import torch

ParamsFuturesAC = {
    "actor_lr": 1e-3,
    "critic_lr": 1e-2,
    "gamma": 0.98,
    # "epsilon": 0.01,
    "state_dim": 8,
    "action_dim": 3,
    "hidden_dim": 256,
    "target_update": 10,
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
}
