import os
import numpy as np
import torch
import torch.nn.functional as F

from App.Base.AgentBase import AgentBase


class Qnet(torch.nn.Module):
    """只有一层隐藏层的Q网络 """

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)


class DoubleDQN(AgentBase):
    """ DQN算法 """

    def __init__(
            self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device, save_dir,
            model_dir,
            *args, **kwargs
    ):
        """
        :param state_dim: dim of states
        :param hidden_dim: hidden layers
        :param action_dim: dim of actions
        :param learning_rate:
        :param gamma:
        :param epsilon:
        :param target_update: update frequency of target_network
        :param device:
        """
        super(DoubleDQN, self).__init__(save_dir, model_dir)
        self.MODEL_FILE = os.path.join(self.MODEL_DIR, "dqn_net.pth")
        self.OPTIMIZER_FILE = os.path.join(self.MODEL_DIR, "opt.pth")
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device
        self.load_model()
        self.loss_dict = {"dqn_loss": []}

    def take_action(self, state, *args, **kwargs):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict, *args, **kwargs):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
        max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()
        self.loss_dict["dqn_loss"].append(dqn_loss.item())

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1

    def save_model(self):

        torch.save(self.q_net.state_dict(), self.MODEL_FILE)
        torch.save(self.optimizer.state_dict(), self.OPTIMIZER_FILE)

    def load_model(self):
        if os.path.exists(self.MODEL_FILE):
            # self.q_net = self.build_nn(self.layer_sizes)
            self.optimizer.load_state_dict(torch.load(self.OPTIMIZER_FILE))
            self.q_net.load_state_dict(torch.load(self.MODEL_FILE))
