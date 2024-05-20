import gym


class EnvBase(gym.Env):
    def __init__(self, render_mode=None, *args, **kwargs):
        self.render_mode = render_mode

