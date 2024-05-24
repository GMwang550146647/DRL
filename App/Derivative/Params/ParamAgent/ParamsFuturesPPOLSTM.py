import torch

ParamsFuturesPPOLSTM = {
    "critic_lr": 1e-2,
    "actor_lr": 1e-3,
    "lmbda": 0.95,
    "gamma": 0.98,
    "epochs": 10,
    "eps": 0.2,
    "hidden_dim": 256,
    "target_update": 10,
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
}
