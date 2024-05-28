class TrainTestBase():
    def __init__(self, agent, env, *args, **kwargs):
        self._agent = agent
        self._env = env
        self.log_interval = 10
        self.start_time = 0

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
