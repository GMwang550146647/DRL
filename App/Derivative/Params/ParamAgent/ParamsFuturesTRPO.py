import torch

ParamsFuturesTRPO = {
    "critic_lr": 1e-2,
    "lmbda": 0.95,
    "gamma": 0.98,
    "kl_constraint": 0.0005,
    "alpha": 0.5,
    "hidden_dim": 256,
    "target_update": 10,
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
}
