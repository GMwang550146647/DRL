
from App.DataSet.DemoDataSet import DemoDateset
import numpy as np
import torch
import matplotlib.pyplot as plt
import gym
import random
from App.Deriatives.Agents.DQN import DQN
from App.Deriatives.TrainTest.TrainTest import TrainTest

if __name__ == '__main__':
    lr = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")


    random.seed(0)
    np.random.seed(0)
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    # env.seed(0)
    torch.manual_seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
                target_update, device)


    tr = TrainTest(
        agent,env,
        num_episodes=num_episodes,
        replay_buffer_cap=buffer_size,
        minimal_size=minimal_size,
        batch_size = batch_size,
        on_policy=False
    ).train()