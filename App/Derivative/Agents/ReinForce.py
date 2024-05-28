import os
import torch
import torch.nn.functional as F
from App.Base.AgentBase import AgentBase


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ReinForce(AgentBase):
    """
    reinforce算法
    """

    def __init__(
            self, state_dim, hidden_dim, action_dim, learning_rate, gamma, target_update, device, save_dir, model_dir,
            *args, **kwargs
    ):
        """
        :param state_dim: dim of states
        :param hidden_dim: hidden layers
        :param action_dim: dim of actions
        :param learning_rate:
        :param gamma:
        :param target_update: update frequency of target_network
        :param device:
        """
        super(ReinForce, self).__init__(save_dir, model_dir)
        self.SAVE_DIR = save_dir
        self.MODEL_DIR = model_dir
        self.MODEL_FILE = os.path.join(self.MODEL_DIR, "dqn_net.pth")
        self.OPTIMIZER_FILE = os.path.join(self.MODEL_DIR, "opt.pth")
        self.action_dim = action_dim
        self.policy_net = PolicyNet(state_dim, hidden_dim, self.action_dim).to(device)  # Q网络
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device
        self.load_model()
        self.loss_dict = {"reinforce_loss":[]}

    def take_action(self, state, *args, **kwargs):  # 根据动作概率分布随机采样
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict, *args, **kwargs):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            G = self.gamma * G + reward
            loss = -log_prob * G  # 每一步的损失函数
            loss.backward()  # 反向传播计算梯度
            self.loss_dict["reinforce_loss"].append(loss.item())
        self.optimizer.step()  # 梯度下降

    def save_model(self):
        torch.save(self.policy_net.state_dict(), self.MODEL_FILE)
        torch.save(self.optimizer.state_dict(), self.OPTIMIZER_FILE)

    def load_model(self):
        if os.path.exists(self.MODEL_FILE):
            self.optimizer.load_state_dict(torch.load(self.OPTIMIZER_FILE))
            self.policy_net.load_state_dict(torch.load(self.MODEL_FILE))
