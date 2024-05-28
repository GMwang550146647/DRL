import os
import matplotlib.pyplot as plt


class TrainTestBase():

    def __init__(self, agent, env, save_dir=None, *args, **kwargs):
        self._agent = agent
        self._env = env
        self.log_interval = 10
        self.start_time = 0
        self._return_list = []
        self._OUTPUT_DIR = save_dir
        self._plot_file = os.path.join(self._OUTPUT_DIR,"RewardEpochs")


    def train(self, *args, **kwargs):
        """
        Train Your Agent
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def test(self, *args, **kwargs):
        """
        Test Your Agent
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def plot_reward(self):
        if self._return_list:
            plt.plot(self._return_list)
        plt.savefig(self._plot_file,dpi=100)