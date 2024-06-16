import gymnasium
import os

class EnvBase(gymnasium.Env):
    def __init__(self, render_mode=None,save_dir=None, *args, **kwargs):
        self.render_mode = render_mode
        self.OUTPUT_PATH = os.path.join(save_dir,"Env")
        os.makedirs(self.OUTPUT_PATH,exist_ok=True)

    def display(self,*args,**kwargs):
        """
        Display After Game is Done
        :return:
        """
        raise NotImplementedError