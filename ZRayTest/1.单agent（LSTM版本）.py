from App.Derivative.Env.EnvFutures import EnvFutures
from App.Derivative.Params.ParamEnv.ParamsFuturesEnv import ParamsFuturesEnv
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.checkpoints import convert_to_msgpack_checkpoint
import os
import gymnasium as gym
import ray
from ray.rllib.algorithms.ppo import PPOConfig

save_dir = "./tmp"
save_checkpoint_dir = "./tmp/checkpoints"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_checkpoint_dir, exist_ok=True)
env_params = {
    'save_dir' :save_dir,
    'params' : ParamsFuturesEnv,
}
ParamsFuturesEnv['save_dir'] = save_dir
# def CreateEnv(*args,**kwargs):
#     env = EnvFutures(save_dir = save_dir,**)
#     return env
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

register_env("my_env",lambda param: EnvFutures(**param))


algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=4)
    .resources(num_gpus=0)
    .environment(env="my_env",env_config=ParamsFuturesEnv)
    .build()
)

for i in range(20):
    result = algo.train()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = algo.save(checkpoint_dir=save_checkpoint_dir)
        print(f"Checkpoint saved in directory {checkpoint_dir}")


# algo1 = Algorithm.from_checkpoint(save_checkpoint_dir)
#
# for i in range(20):
#     result = algo1.train()
#     print(pretty_print(result))
#
#     if i % 5 == 0:
#         checkpoint_dir = algo1.save(checkpoint_dir=save_checkpoint_dir)
#         print(f"Checkpoint saved in directory {checkpoint_dir}")
