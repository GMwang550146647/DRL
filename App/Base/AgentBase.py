from abc import ABC
from stable_baselines3.common.logger import configure


class AgentBase(ABC):
    def __init__(self, save_dir: str, model_dir: str):
        self.SAVE_DIR = save_dir
        self.MODEL_DIR = model_dir
        self.logger = configure(folder=self.SAVE_DIR, format_strings=['stdout', 'log', 'csv', 'tensorboard'])
        self.loss_dict = {}

    def _reset_loss(self):
        for k, _ in self.loss_dict.items():
            self.loss_dict[k] = []

    def take_action(self, state, *args, **kwargs):
        """
        return Action According to given state
        :param state:
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def update(self, transition_dict, *args, **kwargs):
        """
        Update Params According to given transition_dict to Modify your Agent
        :param transition_dict:
        transition_dict = {
            'states': state_i,
            'actions': action_i,
            'next_states': n_state_i,
            'rewards': reward_i,
            'dones': done_i
        }
        """
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError
