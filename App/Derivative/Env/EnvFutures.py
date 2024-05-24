import gym
import logging
import random

import pandas as pd
import numpy as np

from App.Configs.Configs import DATA_PATH
from App.Base.EnvBase import EnvBase
from App.Utils.utils import get_all_pattern_files
from App.Utils.parallel_computing import MultiProcess
from App.Configs.ConfigsFutures import *
from App.Base.PlotBase import PlotBase

class ActionSpace(gym.spaces.Space):
    def __init__(self,n_space):
        super().__init__()
        self.n = n_space
class ObservationSpace(gym.spaces.Space):
    def __init__(self,n_osv):
        super().__init__(shape = (n_osv,))

class EnvFutures(EnvBase):
    def __init__(
            self, file_pattern, data_random_mode=False, load_when_use=True, render_mode=None, save_dir=None,
            *args, **kwargs
    ):
        super(EnvFutures, self).__init__(render_mode, save_dir, *args, **kwargs)
        self._data_path = DATA_PATH
        self._load_when_use = load_when_use
        self._render_mode = render_mode
        self._data_random_mode = data_random_mode
        self._files = get_all_pattern_files(self._data_path, file_pattern)
        random.shuffle(self._files)
        self._N_FILES = len(self._files)
        self._lt_dt_data = MultiProcess.multi_process(self._load_convert_dfdata2dict,
                                                      self._files) if not load_when_use else []
        self._CUR_FILE_IDX = 0
        self._LONG_SHORT_MODE = 0
        self._features_cols = [
            # SPECIAL_TIME_COL, ASK_VIB_COL, BID_VIB_COL, DIFF_AB_COL,
            # HIGH_COL, LOW_COL, CLOSE_COL, MEAN_DEVIATION_COL,
            GT_MIN_RATE_COL, LT_MAX_RATE_COL, PRICE_VIB_COL,
            DIFF_PRICE_COL, TREND_COL
        ]
        self.action_space = ActionSpace(3)
        self.observation_space = ObservationSpace(len(self._features_cols)+1)

    def __len__(self):
        return self._N_FILES

    def _get_next_file_idx(self):
        if self._data_random_mode:
            self._CUR_FILE_IDX = np.random.randint(0, self._N_FILES)
        else:
            self._CUR_FILE_IDX = (self._CUR_FILE_IDX + 1) % self._N_FILES
        return self._CUR_FILE_IDX

    def _get_next_file_data(self):
        next_idx = self._get_next_file_idx()
        flag_success = True
        data = None
        try:
            if self._load_when_use:
                data = self._load_convert_dfdata2dict(self._files[next_idx])
            else:
                data = self._lt_dt_data[next_idx]

        except Exception as err:
            flag_success = False
            logging.error(f"EnvFutures(GetData) Error : error when opening {self._files[next_idx]} -> {err}")
        if not flag_success:
            return self._get_next_file_data()
        else:
            return data

    def _load_convert_dfdata2dict(self, file_name):
        df_data = pd.read_csv(file_name)
        return list(df_data.T.to_dict().values())

    def _get_next_data(self):
        data2return = self._lt_cur_data[self._cur_data_idx]
        self._cur_data_idx += 1
        end = (self._cur_data_idx == len(self._lt_cur_data))
        return data2return, end

    def reset(self, *args, **kwargs):
        self._lt_cur_data = self._get_next_file_data()
        self._cur_data_idx = 0
        self._pre_action = 0
        if self._render_mode:
            self.lt_action = []
        return self.step(0)[0], None

    def step(self, action, *args, **kwargs):
        """
        以下均假设 时间采样频率为10s，其值均是变化的百分比（乘以10000），例如变化了0.0001，记为1
        ASK_VIB:
            前10s 和前20-10s 的AskPrice1 变化
        BID_VIB：
            前10s 和前20-10s 的BidPrice1 变化
        DIFF_AB_FEE:
            前10s 的AskPrice1 和 BidPrice1 的价差均值
        HIGH:
            前10s内最高价 相对于 OPEN 的变化(记为0，可忽略）
        LOW:
            前10s内最低价 相对于 OPEN 的变化(记为0，可忽略)
        CLOSE:
            第10s的最低价 相对于 OPEN 的变化(记为0，可忽略)
        :param action: 采取的动作
            0  : 平仓状态
            1  : 多仓
            2 : 空仓
        :param args:
        :param kwargs:
        :return:state, reward, done, truncated, None 五元组
        """
        cur_data, done = self._get_next_data()
        state = [cur_data[key_i] for key_i in self._features_cols]
        state = [action] + state
        fee = cur_data[FEE_COL] *5
        if action == 1:
            if self._pre_action == 1:
                # 1.Keep Long Pos
                reward = cur_data[BID_VIB_NXT_COL]
            elif self._pre_action == 2:
                # 2.Change From Short To Long Pos
                reward = -cur_data[ASK_VIB_NXT_COL] - fee - cur_data[DIFF_AB_NXT_COL] - fee
            else:
                # 3.Open Long Pos
                reward = -cur_data[DIFF_AB_NXT_COL] - fee
        elif action == 2:
            if self._pre_action == 1:
                # 4.Change From Long To Short Pos
                reward = cur_data[BID_VIB_NXT_COL] - fee - cur_data[DIFF_AB_NXT_COL] - fee
            elif self._pre_action == 2:
                # 5.Keep Short Pos
                reward = -cur_data[ASK_VIB_NXT_COL]
            else:
                # 6.Open Short Pos
                reward = -cur_data[DIFF_AB_NXT_COL] - fee
        else:
            if self._pre_action == 1:
                # 7.Close Long Pos
                reward = cur_data[BID_VIB_NXT_COL] - fee
            elif self._pre_action == 2:
                # 8.CLose Short Pos
                reward = -cur_data[ASK_VIB_NXT_COL] - fee
            else:
                # 9.Keep 0 Pos
                reward = 0
        self._pre_action = action
        if self._render_mode:
            self.lt_action.append((action, reward))
        return state, reward, done, None, None

    def render(self, *args, **kwargs):
        pass

    def display(self, *args, **kwargs):
        cur_file = self._files[self._CUR_FILE_IDX].split("/")[-1].split(".")[0]
        cur_data = self._lt_cur_data
        buy_vol = pd.Series([(1 if a_i[0] == 1 else 0) for a_i in self.lt_action])
        sell_vol = pd.Series([(1 if a_i[0] == 2 else 0) for a_i in self.lt_action])
        rewards = pd.Series([a_i[1] for a_i in self.lt_action]).cumsum()
        update_time = pd.Series([item[UPDATE_TIME_COL] for item in cur_data])
        last_price = pd.Series([item[LAST_PRICE_COL] for item in cur_data])
        new_df_data = pd.DataFrame()
        new_df_data[UPDATE_TIME_COL] = update_time
        new_df_data[LAST_PRICE_COL] = last_price
        new_df_data["BUY_VOL"] = buy_vol
        new_df_data["SELL_VOL"] = sell_vol
        new_df_data["POS_REWARDS"] = rewards
        new_df_data["NEG_REWARDS"] = rewards
        new_df_data.loc[new_df_data["POS_REWARDS"] < 0, "POS_REWARDS"] = 0
        new_df_data.loc[new_df_data["NEG_REWARDS"] > 0, "NEG_REWARDS"] = 0
        new_df_data['NEG_REWARDS'] = -new_df_data['NEG_REWARDS']
        plot_cols = [LAST_PRICE_COL]
        bar_cols = ["BUY_VOL", "SELL_VOL"]
        PlotBase().plot_bar_curve(new_df_data, plot_cols, bar_cols, self.OUTPUT_PATH, title=f"{cur_file}_{round(rewards[len(rewards)-1])}")

        plot_cols = [LAST_PRICE_COL]
        bar_cols = ["POS_REWARDS", "NEG_REWARDS"]
        PlotBase().plot_bar_curve(new_df_data, plot_cols, bar_cols, self.OUTPUT_PATH, title=f"{cur_file}_{round(rewards[len(rewards)-1])}_reward")
