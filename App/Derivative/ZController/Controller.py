import importlib
import logging
import datetime

from App.Base.ControllerBase import ControllerBase
from App.Configs.Configs import *
from App.Utils.utils import *


class Controller(ControllerBase):
    def __init__(
            self, info_level=logging.INFO,
            *args, **kwargs
    ):
        super(Controller, self).__init__()
        self.DT_CONFIGS_APP = DT_CONFIGS_APP
        self.TRAIN_MODE = self.DT_CONFIGS_APP.get(TRAIN_MODE_COL)
        self.AGENT_NAME = self.DT_CONFIGS_APP.get(AGENT_COL)
        self.ENV_NAME = self.DT_CONFIGS_APP.get(ENV_COL)
        self.TT_NAME = self.DT_CONFIGS_APP.get(TT_COL)
        self.AGENT_CONFIG_NAME = self.DT_CONFIGS_APP.get(AGENT_CONFIG_COL)
        self.ENV_CONFIGS_NAME = self.DT_CONFIGS_APP.get(ENV_CONFIGS_COL)
        self.TT_CONFIGS_NAME = self.DT_CONFIGS_APP.get(TT_CONFIGS_COL)

        self.task_name = f"{self.ENV_NAME}_{self.AGENT_NAME}"

        self.logger = logging.getLogger(self.task_name)
        self.datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        self.OUTPUT_DIR = os.path.join(OUTPUT_PATH, self.task_name, self.datetime)
        self.SAVE_MODEL_DIR = os.path.join(OUTPUT_MODEL_PATH, self.task_name)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.SAVE_MODEL_DIR, exist_ok=True)
        self._load_objects()
        self.logger.setLevel(info_level)
        logging.info(f"Task Activated : AGENT({self.AGENT_NAME}) | ENV({self.ENV_NAME}) | TT_NAME({self.TT_NAME})")

    def _load_objects(self):
        self.AGENT = self._import_module(self.AGENT_NAME)
        self.ENV = self._import_module(self.ENV_NAME)
        self.TT = self._import_module(self.TT_NAME)
        self.AGENT_CONFIG = self._import_module(self.AGENT_CONFIG_NAME)
        self.ENV_CONFIGS = self._import_module(self.ENV_CONFIGS_NAME)
        self.TT_CONFIGS = self._import_module(self.TT_CONFIGS_NAME)

    def run(self):
        if self.TRAIN_MODE:
            env = self.ENV(save_dir=self.OUTPUT_DIR, **self.ENV_CONFIGS)
        else:
            env = self.ENV(render_mode='human', save_dir=self.OUTPUT_DIR, **self.ENV_CONFIGS)
        agent = self.AGENT(save_dir=self.OUTPUT_DIR, model_dir=self.SAVE_MODEL_DIR, **self.AGENT_CONFIG)
        tt = self.TT(agent=agent, env=env, save_dir=self.OUTPUT_DIR, **self.TT_CONFIGS)
        if self.TRAIN_MODE:
            tt.train()
        else:
            tt.test()
