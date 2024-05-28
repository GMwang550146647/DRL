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
        self._episode_count = 0
        self._replay_buffer = ReplayBuffer(replay_buffer_cap)
        self._minimal_size = minimal_size
        self._batch_size = batch_size
        self._on_policy = on_policy

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

    def train_on_policy_agent(self, env, agent):
        """
        Yes, It Trains On Policy
        :param env:
        :param agent:
        :param num_episodes:
        :return:
        """
        self.start_time = time.time_ns()
        return_list = []
        for i in range(self.log_interval):
            with tqdm(total=int(self._num_episodes / self.log_interval), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(self._num_episodes / self.log_interval)):
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

                    self._episode_count += 1
                    time_elapsed = time.time() - self.start_time
                    agent.logger.record("time/episodes", self._episode_count, exclude="tensorboard")
                    agent.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                    agent.logger.record("time/return", np.mean(return_list[-self.log_interval:]))
                    for loss_name, loss_value in agent.loss_dict.items():
                        agent.logger.record(f"time/{loss_name}", np.mean(loss_value))
                    agent._reset_loss()

                    if self._episode_count % self.log_interval == 0:
                        pbar.set_postfix({'episode': '%d' % self._episode_count,
                                          'return': '%.3f' % np.mean(return_list[-self.log_interval:])})

                    pbar.update(1)

            # dump log every num_episodes / log_interval
            agent.logger.dump(step=self._episode_count)
            # TODO: check the performance and save the best model only
            self._agent.save_model()
        return return_list

    def train_off_policy_agent(self, env, agent):
        """
        Trains Off Policy
        :param env:
        :param agent:
        :param num_episodes:
        :return:
        """
        return_list = []
        for i in range(self.log_interval):
            with tqdm(total=int(self._num_episodes / self.log_interval), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(self._num_episodes / self.log_interval)):
                    episode_return = 0
                    state = env.reset()[0]
                    done = False
                    truncated = False

                    while (not done) and (not truncated):
                        action = agent.take_action(state)
                        next_state, reward, done, truncated, _ = env.step(action)
                        self._replay_buffer.add(state, action, reward, next_state, done)
                        state = next_state
                        episode_return += reward
                        if self._replay_buffer.size() > self._minimal_size:
                            b_s, b_a, b_r, b_ns, b_d = self._replay_buffer.sample(self._batch_size)
                            transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,'dones': b_d}
                            agent.update(transition_dict)

                        self._episode_count += 1
                        time_elapsed = time.time() - self.start_time
                        agent.logger.record("time/episodes", self._episode_count, exclude="tensorboard")
                        agent.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                        agent.logger.record("time/return", np.mean(return_list[-self.log_interval:]))
                        for loss_name, loss_value in agent.loss_dict.items():
                            agent.logger.record(f"time/{loss_name}", np.mean(loss_value))
                        agent._reset_loss()

                    return_list.append(episode_return)
                    if self._episode_count % self.log_interval == 0:
                        pbar.set_postfix({'episode': '%d' % self._episode_count,
                                          'return': '%.3f' % np.mean(return_list[-self.log_interval:])})
                    pbar.update(1)

            agent.logger.dump(step=self._episode_count)
            # TODO: check the performance and save the best model only
            self._agent.save_model()
        return return_list

    def train(self, *args, **kwargs):
        if self._on_policy:
            self.train_on_policy_agent(self._env, self._agent)
        else:
            self.train_off_policy_agent(self._env, self._agent)

    def test(self, epochs=10, time_interval=0.01, *args, **kwargs):
        for i in tqdm(range(len(self._env))):
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
