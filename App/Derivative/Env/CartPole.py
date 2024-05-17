import gym
from App.Base.EnvBase import EnvBase


class CartPole(EnvBase):
    def __init__(self,render_mode=None, *args, **kwargs):
        super(CartPole, self).__init__(render_mode)

    def __new__(cls, *args, **kwargs):
        env_name = 'CartPole-v1'
        render_mode = kwargs.get('render_mode',None)
        if render_mode:
            env = gym.make(env_name,render_mode = render_mode)
        else:
            env = gym.make(env_name)
        return env
