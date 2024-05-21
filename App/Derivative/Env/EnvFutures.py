import gym
import logging
import random

import pandas as pd
import numpy as np

from App.Configs.Configs import DATA_PATH
from App.Base.EnvBase import EnvBase
from App.Utils.utils import get_all_pattern_files
from App.Configs.ConfigsFutures import *


class EnvFutures(EnvBase):
    def __init__(self, file_pattern, data_random_mode=False, load_when_use=True, render_mode=None, *args, **kwargs):
        super(EnvFutures, self).__init__(render_mode)
        self._data_path = DATA_PATH
        self._load_when_use = load_when_use
        self._data_random_mode = data_random_mode
        self._files = get_all_pattern_files(self._data_path, file_pattern)
        random.shuffle(self._files)
        self._N_FILES = len(self._files)
        self._lt_df_data = [pd.read_csv(file_i) for file_i in self._files] if load_when_use else []
        self._CUR_FILE_IDX = 0
        self._LONG_SHORT_MODE = 0

    def __len__(self):
        return self._N_FILES

    def _get_next_file_idx(self):
        if self._data_random_mode:
            self._CUR_FILE_IDX = np.random.randint(0, self._N_FILES)
        else:
            self._CUR_FILE_IDX = (self._CUR_FILE_IDX + 1) % self._N_FILES
        logging.info(f"CurFile({self._CUR_FILE_IDX}) : {self._files[self._CUR_FILE_IDX]}")
        return self._CUR_FILE_IDX

    def _get_next_file_data(self):
        next_idx = self._get_next_file_idx()
        flag_success = True
        data = None
        try:
            if self._load_when_use:
                df_data = pd.read_csv(self._files[next_idx])
            else:
                df_data = self._lt_df_data[next_idx]
            data = list(df_data.T.to_dict().values())

        except Exception as err:
            flag_success = False
            logging.error(f"EnvFutures(GetData) Error : error when opening {self._files[next_idx]} -> {err}")
        if not flag_success:
            return self._get_next_file_data()
        else:
            return data

    def _get_next_data(self):
        data2return = self._lt_cur_data[self._cur_data_idx]
        self._cur_data_idx += 1
        end = (self._cur_data_idx == len(self._lt_cur_data))
        return data2return, end

    def reset(self, *args, **kwargs):
        self._lt_cur_data = self._get_next_file_data()
        self._cur_data_idx = 0
        self._pre_action = 0
        return self.step(0)[0],None

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
        cur_data,done = self._get_next_data()
        state = [cur_data[key_i] for key_i in
                 [SPECIAL_TIME_COL, ASK_VIB_COL, BID_VIB_COL, DIFF_AB_COL, HIGH_COL, LOW_COL, CLOSE_COL]]
        state = [action] + state
        if action == 1:
            if self._pre_action == 1:
                # 1.Keep Long Pos
                reward = cur_data[BID_VIB_COL]
            elif self._pre_action == 2:
                # 2.Change From Short To Long Pos
                reward = -cur_data[ASK_VIB_COL] - cur_data[FEE_COL] - cur_data[DIFF_AB_COL] - cur_data[FEE_COL]
            else:
                # 3.Open Long Pos
                reward = -cur_data[DIFF_AB_COL] - cur_data[FEE_COL]
        elif action == 2:
            if self._pre_action == 1:
                # 4.Change From Long To Short Pos
                reward = cur_data[BID_VIB_COL] - cur_data[FEE_COL] - cur_data[DIFF_AB_COL] - cur_data[FEE_COL]
            elif self._pre_action == 2:
                # 5.Keep Short Pos
                reward = -cur_data[ASK_VIB_COL]
            else:
                # 6.Open Short Pos
                reward = -cur_data[DIFF_AB_COL] - cur_data[FEE_COL]
        else:
            if self._pre_action == 1:
                # 7.Close Long Pos
                reward = cur_data[BID_VIB_COL] - cur_data[FEE_COL]
            elif self._pre_action == 2:
                # 8.CLose Short Pos
                reward = -cur_data[ASK_VIB_COL] - cur_data[FEE_COL]
            else:
                # 9.Keep 0 Pos
                reward = 0
        self._pre_action = action
        return state, reward, done, None, None
