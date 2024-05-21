from App.Base.TrainTestBase import TrainTestBase

from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import time
import logging


class TrainTest(TrainTestBase):
    def __init__(
            self, agent, env, num_episodes, replay_buffer_cap, minimal_size=128, batch_size=16, on_policy=False,
            *args, **kwargs
    ):
        super().__init__(agent, env)
        self._env = env
        self._agent = agent
        self._num_episodes = num_episodes
        self._replay_buffer = ReplayBuffer(replay_buffer_cap)
        self._minimal_size = minimal_size
        self._batch_size = batch_size
        self._on_policy = on_policy

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

    def train_on_policy_agent(self, env, agent, num_episodes):
        """
        Yes, It Trains On Policy
        :param env:
        :param agent:
        :param num_episodes:
        :return:
        """
        return_list = []
        for i in range(10):
            with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(num_episodes / 10)):
                    episode_return = 0
                    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                    state = env.reset()[0]
                    done = False
                    truncated = False
                    while (not done) and (not truncated):
                        action = agent.take_action(state)
                        next_state, reward, done, truncated, _ = env.step(action)
                        transition_dict['states'].append(state)
                        transition_dict['actions'].append(action)
                        transition_dict['next_states'].append(next_state)
                        transition_dict['rewards'].append(reward)
                        transition_dict['dones'].append(done)
                        state = next_state
                        episode_return += reward
                    return_list.append(episode_return)
                    agent.update(transition_dict)
                    if (i_episode + 1) % 10 == 0:
                        pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                          'return': '%.3f' % np.mean(return_list[-10:])})
                    pbar.update(1)
            self._agent.save_model()
        return return_list

    def train_off_policy_agent(self, env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
        """
        Trains Off Policy
        :param env:
        :param agent:
        :param num_episodes:
        :return:
        """
        return_list = []
        for i in range(10):
            t0 = time.time()
            with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(num_episodes / 10)):
                    episode_return = 0
                    state = env.reset()[0]
                    done = False
                    truncated = False

                    while (not done) and (not truncated):
                        action = agent.take_action(state)
                        next_state, reward, done, truncated, _ = env.step(action)
                        replay_buffer.add(state, action, reward, next_state, done)
                        state = next_state
                        episode_return += reward
                        if replay_buffer.size() > minimal_size:
                            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                            transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,'dones': b_d}
                            agent.update(transition_dict)

                    return_list.append(episode_return)
                    if (i_episode + 1) % 10 == 0:
                        pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                          'return': '%.3f' % np.mean(return_list[-10:])})
                    pbar.update(1)
            print(f"Used {time.time() - t0} s")
            self._agent.save_model()
        return return_list

    def train(self, *args, **kwargs):
        if self._on_policy:
            self.train_on_policy_agent(self._env, self._agent, self._num_episodes)
        else:
            self.train_off_policy_agent(
                self._env, self._agent, self._num_episodes, self._replay_buffer, self._minimal_size, self._batch_size)

    def test(self, epochs=10, time_interval=0.01, *args, **kwargs):
        for i in tqdm(range(epochs)):
            truncated = False
            state, done, rew = self._env.reset()[0], False, 0
            while (not done and not truncated):
                self._env.render()
                action = self._agent.take_action(state)
                state, reward, done, truncated, _ = self._env.step(action)
                rew += reward
                time.sleep(time_interval)
            self._env.display()
            logging.info(f"Epoch {i + 1} : reward -> {rew}")
            rew = 0


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)
