import gym
from App.Base.EnvBase import EnvBase


class EnvFutures(EnvBase):
    def __init__(self, render_mode=None, *args, **kwargs):
        super(EnvFutures, self).__init__(render_mode)

    def reset(self,*args,**kwargs):

        return None,{}

    def step(self,action, *args, **kwargs):

        return None