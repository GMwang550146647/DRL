import os
import torch
import torch.nn.functional as F
import numpy as np

from App.Base.AgentBase import AgentBase


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class QValueNet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SAC(AgentBase):
    ''' DQN算法 '''

    def __init__(
            self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma, device, save_dir,
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
        super(SAC, self).__init__()
        self.SAVE_DIR = save_dir
        self.MODEL_DIR = model_dir
        self.MODEL_FILE_ACTOR = os.path.join(self.MODEL_DIR, "actor_net.pth")
        self.MODEL_FILE_CRITIC1 = os.path.join(self.MODEL_DIR, "critic_net1.pth")
        self.MODEL_FILE_CRITIC2 = os.path.join(self.MODEL_DIR, "critic_net2.pth")
        self.OPTIMIZER_ACTOR_FILE = os.path.join(self.MODEL_DIR, "opt_actor.pth")
        self.OPTIMIZER_CRITIC1_FILE = os.path.join(self.MODEL_DIR, "opt_critic1.pth")
        self.OPTIMIZER_CRITIC2_FILE = os.path.join(self.MODEL_DIR, "opt_critic2.pth")


        self.action_dim = action_dim
        self.actor = PolicyNet(state_dim, hidden_dim, self.action_dim).to(device)
        self.critic_1 = QValueNet(state_dim, hidden_dim,action_dim).to(device)
        self.critic_2 = QValueNet(state_dim, hidden_dim,action_dim).to(device)
        self.target_critic_1 = QValueNet(state_dim, hidden_dim,action_dim).to(device)
        self.target_critic_2 = QValueNet(state_dim, hidden_dim,action_dim).to(device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)  # 使用Adam优化器
        self.optimizer_critic1 = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)  # 使用Adam优化器
        self.optimizer_critic2 = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)  # 使用Adam优化器

        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.optimizer_log_alpha = torch.optim.Adam([self.log_alpha],   lr=alpha_lr)

        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.load_model()

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

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_q_values = self.critic_1(states).gather(1, actions)
        critic_1_loss = torch.mean(F.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_q_values = self.critic_2(states).gather(1, actions)
        critic_2_loss = torch.mean(F.mse_loss(critic_2_q_values, td_target.detach()))
        self.optimizer_critic1.zero_grad()
        critic_1_loss.backward()
        self.optimizer_critic1.step()
        self.optimizer_critic2.zero_grad()
        critic_2_loss.backward()
        self.optimizer_critic2.step()

        # 更新策略网络
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        # 直接根据概率计算熵
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)  #
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)  # 直接根据概率计算期望
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # 更新alpha值
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.optimizer_log_alpha.zero_grad()
        alpha_loss.backward()
        self.optimizer_log_alpha.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

    def save_model(self):
        torch.save(self.actor.state_dict(), self.MODEL_FILE_ACTOR)
        torch.save(self.critic_1.state_dict(), self.MODEL_FILE_CRITIC1)
        torch.save(self.critic_2.state_dict(), self.MODEL_FILE_CRITIC2)
        torch.save(self.optimizer_actor.state_dict(), self.OPTIMIZER_ACTOR_FILE)
        torch.save(self.optimizer_critic1.state_dict(), self.OPTIMIZER_CRITIC1_FILE)
        torch.save(self.optimizer_critic2.state_dict(), self.OPTIMIZER_CRITIC2_FILE)

    def load_model(self):
        if os.path.exists(self.MODEL_FILE_ACTOR) and os.path.exists(self.MODEL_FILE_CRITIC1)and os.path.exists(self.MODEL_FILE_CRITIC2):
            self.optimizer_actor.load_state_dict(torch.load(self.OPTIMIZER_ACTOR_FILE))
            self.optimizer_critic1.load_state_dict(torch.load(self.OPTIMIZER_CRITIC1_FILE))
            self.optimizer_critic2.load_state_dict(torch.load(self.OPTIMIZER_CRITIC2_FILE))
            self.actor.load_state_dict(torch.load(self.MODEL_FILE_ACTOR))
            self.critic_1.load_state_dict(torch.load(self.MODEL_FILE_CRITIC1))
            self.critic_2.load_state_dict(torch.load(self.MODEL_FILE_CRITIC2))
            self.target_critic_1.load_state_dict(self.critic_1.state_dict())
            self.target_critic_2.load_state_dict(self.critic_2.state_dict())

    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def calc_target(self, rewards, next_states, dones):
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
