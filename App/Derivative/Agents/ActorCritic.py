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


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ActorCritic(AgentBase):
    """ActorCritic算法"""

    def __init__(
            self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, target_update, device, save_dir,
            model_dir,
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
        super(ActorCritic, self).__init__(save_dir, model_dir)
        self.MODEL_FILE_ACTOR = os.path.join(self.MODEL_DIR, "actor_net.pth")
        self.MODEL_FILE_CRITIC = os.path.join(self.MODEL_DIR, "critic_net.pth")
        self.OPTIMIZER_ACTOR_FILE = os.path.join(self.MODEL_DIR, "opt_actor.pth")
        self.OPTIMIZER_CRITIC_FILE = os.path.join(self.MODEL_DIR, "opt_critic.pth")
        self.action_dim = action_dim
        self.actor = PolicyNet(state_dim, hidden_dim, self.action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)  # 使用Adam优化器
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device
        self.load_model()
        self.loss_dict = {"actor_loss": [], "critic_loss": []}

    def take_action(self, state, *args, **kwargs):  # 根据动作概率分布随机采样
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict, *args, **kwargs):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)  # 时序差分误差
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        # 均方误差损失函数
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        actor_loss.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        self.optimizer_actor.step()  # 更新策略网络的参数
        self.optimizer_critic.step()  # 更新价值网络的参数
        self.loss_dict["actor_loss"].append(actor_loss.item())
        self.loss_dict["critic_loss"].append(critic_loss.item())

    def save_model(self):
        torch.save(self.actor.state_dict(), self.MODEL_FILE_ACTOR)
        torch.save(self.critic.state_dict(), self.MODEL_FILE_CRITIC)
        torch.save(self.optimizer_actor.state_dict(), self.OPTIMIZER_ACTOR_FILE)
        torch.save(self.optimizer_critic.state_dict(), self.OPTIMIZER_CRITIC_FILE)

    def load_model(self):
        if os.path.exists(self.MODEL_FILE_ACTOR) and os.path.exists(self.MODEL_FILE_CRITIC):
            self.optimizer_actor.load_state_dict(torch.load(self.OPTIMIZER_ACTOR_FILE))
            self.optimizer_critic.load_state_dict(torch.load(self.OPTIMIZER_CRITIC_FILE))
            self.actor.load_state_dict(torch.load(self.MODEL_FILE_ACTOR))
            self.critic.load_state_dict(torch.load(self.MODEL_FILE_CRITIC))
