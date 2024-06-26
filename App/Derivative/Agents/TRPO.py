from App.Base.AgentBase import AgentBase
import copy
import os
import numpy as np
import torch
import torch.nn.functional as F


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

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


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


class TRPO(AgentBase):
    """ TPRO 算法 """

    def __init__(
            self, state_dim, hidden_dim, action_dim, lmbda, kl_constraint, alpha, critic_lr, gamma, target_update,
            device, save_dir,
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
        super(TRPO, self).__init__(save_dir, model_dir)
        self.MODEL_FILE_ACTOR = os.path.join(self.MODEL_DIR, "actor_net.pth")
        self.MODEL_FILE_CRITIC = os.path.join(self.MODEL_DIR, "critic_net.pth")
        self.OPTIMIZER_ACTOR_FILE = os.path.join(self.MODEL_DIR, "opt_actor.pth")
        self.OPTIMIZER_CRITIC_FILE = os.path.join(self.MODEL_DIR, "opt_critic.pth")
        self.action_dim = action_dim
        self.actor = PolicyNet(state_dim, hidden_dim, self.action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子
        self.target_update = target_update  # 目标网络更新频率
        self.lmbda = lmbda  # GAE参数
        self.kl_constraint = kl_constraint  # KL距离最大限制
        self.alpha = alpha  # 线性搜索参数
        self.count = 0  # 计数器,记录更新次数
        self.device = device
        self.load_model()
        self.loss_dict = {"critic_loss": []}

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
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        old_action_dists = torch.distributions.Categorical(self.actor(states).detach())
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()  # 更新价值函数
        self.loss_dict["critic_loss"].append(critic_loss.item())
        # 更新策略函数
        self.policy_learn(states, actions, old_action_dists, old_log_probs, advantage)

    def save_model(self):
        torch.save(self.actor.state_dict(), self.MODEL_FILE_ACTOR)
        torch.save(self.critic.state_dict(), self.MODEL_FILE_CRITIC)
        torch.save(self.optimizer_critic.state_dict(), self.OPTIMIZER_CRITIC_FILE)

    def load_model(self):
        if os.path.exists(self.MODEL_FILE_ACTOR) and os.path.exists(self.MODEL_FILE_CRITIC):
            self.optimizer_critic.load_state_dict(torch.load(self.OPTIMIZER_CRITIC_FILE))
            self.actor.load_state_dict(torch.load(self.MODEL_FILE_ACTOR))
            self.critic.load_state_dict(torch.load(self.MODEL_FILE_CRITIC))

    def hessian_matrix_vector_product(self, states, old_action_dists, vector):
        # 计算黑塞矩阵和一个向量的乘积
        new_action_dists = torch.distributions.Categorical(self.actor(states))
        kl = torch.mean(
            torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))  # 计算平均KL距离
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        # KL距离的梯度先和向量进行点积运算
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product, self.actor.parameters())
        grad2_vector = torch.cat([grad.view(-1) for grad in grad2])
        return grad2_vector

    def conjugate_gradient(self, grad, states, old_action_dists):  # 共轭梯度法求解方程
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()

        r = torch.where(torch.isnan(r), torch.full_like(r, 0), r)
        p = torch.where(torch.isnan(p), torch.full_like(p, 0), p)
        rdotr = torch.dot(r, r)
        for i in range(10):  # 共轭梯度主循环
            Hp = self.hessian_matrix_vector_product(states, old_action_dists, p)
            alpha = rdotr / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        if x.data[0].numpy() == np.nan:
            a = 0
        return x

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs, actor):  # 计算策略目标
        log_probs = torch.log(actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    def line_search(self, states, actions, advantage, old_log_probs, old_action_dists, max_vec):  # 线性搜索
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(self.actor.parameters())
        old_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)
        for i in range(15):  # 线性搜索主循环
            coef = self.alpha ** i
            new_para = old_para + coef * max_vec
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(new_para, new_actor.parameters())
            new_action_dists = torch.distributions.Categorical(new_actor(states))
            kl_div = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))
            new_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, new_actor)
            if new_obj > old_obj and kl_div < self.kl_constraint:
                return new_para
        return old_para

    def policy_learn(self, states, actions, old_action_dists, old_log_probs, advantage):  # 更新策略函数
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        # 用共轭梯度法计算x = H^(-1)g
        descent_direction = self.conjugate_gradient(obj_grad, states, old_action_dists)

        Hd = self.hessian_matrix_vector_product(states, old_action_dists, descent_direction)
        max_coef = torch.sqrt(2 * self.kl_constraint / (torch.dot(descent_direction, Hd) + 1e-8))
        new_para = self.line_search(
            states, actions, advantage, old_log_probs, old_action_dists, descent_direction * max_coef)  # 线性搜索
        torch.nn.utils.convert_parameters.vector_to_parameters(new_para, self.actor.parameters())  # 用线性搜索后的参数更新策略
