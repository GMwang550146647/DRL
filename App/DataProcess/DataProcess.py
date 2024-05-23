import os

import pandas as pd
import numpy as np

from App.Configs.Configs import RAW_DATA_PATH, DATA_PATH
from App.Base.FuturesBase import FuturesBase
from App.Configs.ConfigsFutures import *
from App.Utils.utils import get_all_pattern_files
from App.Utils.parallel_computing import MultiProcess


class DataProcess(FuturesBase):
    def __init__(self, futures, file_pattern=".csv$"):
        super().__init__()
        self._input_dir = os.path.join(RAW_DATA_PATH, futures)
        self._output_dir = os.path.join(DATA_PATH, futures)
        self._input_files = get_all_pattern_files(self._input_dir, file_pattern)
        self._output_files = [file_i.replace(self._input_dir, self._output_dir) for file_i in self._input_files]
        self._input2output = dict(zip(self._input_files, self._output_files))

        self._EXP_TS = 10000

    def run(self):
        MultiProcess.multi_process(self._run, self._input_files)

    def _run(self, file_path):
        df_data = self.load_df_data(file_path)
        df_data = self.process(df_data)
        output_file = self._input2output[file_path]
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        df_data.to_csv(output_file, index=None)

    def process(self, df_data, *args, **kwargs):
        def cal_active_factor(df_data, npl=1, shift_t=1):
            ap_1 = df_data[ASK_PRICE1_COL].shift(shift_t).fillna(method='bfill')
            bp_1 = df_data[BID_PRICE1_COL].shift(shift_t).fillna(method='bfill')
            df_data[VOLUME_COL] = (df_data[ACC_VOLUME_COL] - df_data[ACC_VOLUME_COL].shift(shift_t)).fillna(0)
            df_data[TURN_OVER_COL] = (df_data[ACC_TURN_OVER_COL] - df_data[ACC_TURN_OVER_COL].shift(shift_t)).fillna(0)
            lt_zero = ((df_data[VOLUME_COL] < -1) & (df_data[TURN_OVER_COL] < -1))
            df_data.loc[lt_zero, VOLUME_COL] = df_data.loc[lt_zero, ACC_VOLUME_COL]
            df_data.loc[lt_zero, TURN_OVER_COL] = df_data.loc[lt_zero, ACC_TURN_OVER_COL]
            avg_price = df_data[TURN_OVER_COL] / (df_data[VOLUME_COL] + CLOSE2ZERO) / npl
            diff_ab = (ap_1 - bp_1)
            # avg_ab = ((ap_1 + bp_1) / 2.0)
            gt_ap1 = (avg_price > ap_1)
            lt_bp1 = (avg_price < bp_1)
            avg_price[gt_ap1] = ap_1[gt_ap1]
            avg_price[lt_bp1] = bp_1[lt_bp1]
            diff_ab.fillna(CLOSE2ZERO, inplace=True)
            # avg_ab.fillna(0, inplace=True)
            ask_signal = (avg_price - bp_1) / diff_ab * df_data[VOLUME_COL]
            bid_signal = (ap_1 - avg_price) / diff_ab * df_data[VOLUME_COL]
            return ask_signal, bid_signal

        def get_skipped_data(df_data, n_skipped=20):
            clock_time = pd.to_datetime(df_data[UPDATE_TIME_COL])
            clock_time = clock_time.dt.minute * 60 + clock_time.dt.second + clock_time.dt.microsecond / 1000000
            selected_time = (clock_time % (n_skipped / 2) == 0)
            df_data = df_data.loc[selected_time, :]
            df_data = df_data.reset_index(drop=True)
            return df_data

        def del_row(df_data):
            lt_date = df_data[DATE_COL].to_list()
            lt_time = df_data[UPDATE_TIME_COL].to_list()
            se_date_time = pd.Series(pd.to_datetime([f"{date_i} {time_i}" for date_i, time_i in zip(lt_date, lt_time)]))

            diff_date_time = (se_date_time - se_date_time.shift(1)).fillna(delta_time)
            time2del = (diff_date_time >= delta_time)
            time2del_index = diff_date_time[time2del].index
            for idx_i in time2del_index:
                if not (se_date_time[idx_i].second == 0 and se_date_time[idx_i].microsecond == 0):
                    time2del[idx_i] = False

            df_data = df_data.loc[time2del == False, :].reset_index(drop=True)
            return df_data

        def get_data_delta_time(df_data):
            pd_update_time = pd.to_datetime(df_data[UPDATE_TIME_COL])
            delta_time = pd_update_time - pd_update_time.shift(1).fillna(method='bfill')
            unique_time = np.unique(delta_time)
            data_count = {
                t_i: (delta_time == t_i).sum() for t_i in unique_time
            }
            sorted_lt = sorted(data_count.items(), key=lambda x: x[1])
            return sorted_lt[-1][0]

        def cal_special_time(df_data):
            lt_date = df_data[DATE_COL].to_list()
            lt_time = df_data[UPDATE_TIME_COL].to_list()
            se_date_time = pd.Series(pd.to_datetime([f"{date_i} {time_i}" for date_i, time_i in zip(lt_date, lt_time)]))

            """1.Special Time"""
            diff_date_time = (se_date_time.shift(-1) - se_date_time).fillna(delta_time)
            special_idex = (diff_date_time >= delta_time)
            special_idex = diff_date_time[special_idex].index
            df_data.loc[special_idex, SPECIAL_TIME_COL] = 1

        def cal_env_variables(df_data):
            base_price = df_data.loc[0, LAST_PRICE_COL]
            fee = self.get_fee(ITEM, base_price, 1) / base_price / 2 * self._EXP_TS
            diff_a = (df_data[BUY_PRICE_COL] - df_data[BUY_PRICE_COL].shift(1)).fillna(0) / base_price * self._EXP_TS
            diff_b = (df_data[SELL_PRICE_COL] - df_data[SELL_PRICE_COL].shift(1)).fillna(0) / base_price * self._EXP_TS
            df_data[ASK_VIB_COL] = diff_a.shift(1).fillna(0)
            df_data[BID_VIB_COL] = diff_b.shift(1).fillna(0)
            diff_ab = (df_data[BUY_PRICE_COL] - df_data[SELL_PRICE_COL]).fillna(0) / base_price * self._EXP_TS
            df_data[DIFF_AB_COL] = diff_ab.shift(1).fillna(0)
            df_data[FEE_COL] = fee

            df_data[ASK_VIB_NXT_COL] = diff_a
            df_data[BID_VIB_NXT_COL] = diff_b
            df_data[DIFF_AB_NXT_COL] = diff_ab

            df_data[HIGH_COL] = (df_data[HIGH_PRICE_COL] - df_data[OPEN_PRICE_COL])/df_data[OPEN_PRICE_COL] * self._EXP_TS
            df_data[LOW_COL] = (df_data[LOW_PRICE_COL] - df_data[OPEN_PRICE_COL])/df_data[OPEN_PRICE_COL] * self._EXP_TS
            df_data[CLOSE_COL] = (df_data[CLOSE_PRICE_COL] - df_data[OPEN_PRICE_COL])/df_data[OPEN_PRICE_COL] * self._EXP_TS

        STANDARD_TIME = pd.to_datetime('00:00:10') - pd.to_datetime('00:00:00')
        delta_time = pd.to_datetime('00:10:00') - pd.to_datetime('00:00:00')
        data_delta_time = get_data_delta_time(df_data)
        CLOSE2ZERO = 0.000000001
        WIN_SIZE = round(STANDARD_TIME / data_delta_time)
        ITEM = df_data.loc[0, ITEM_COL]
        NPL = self.get_numperlot(ITEM)
        # all_columns = [
        #     UPDATE_TIME_COL,SPECIAL_TIME_COL, ITEM_COL, DATE_COL, TURN_OVER_COL, VOLUME_COL, ACC_TURN_OVER_COL, ACC_VOLUME_COL,
        #     # 'ask_vol', 'bid_vol',
        #     LAST_PRICE_COL, OPEN_PRICE_COL, CLOSE_PRICE_COL, HIGH_PRICE_COL, LOW_PRICE_COL, AVG_PRICE_COL,
        #     ACTIVE_FACTOR_COL, BUY_PRICE_COL, SELL_PRICE_COL, ASK_VOLUME5_COL, ASK_VOLUME4_COL, ASK_VOLUME3_COL,
        #     ASK_VOLUME2_COL, ASK_VOLUME1_COL, ASK_PRICE5_COL, ASK_PRICE4_COL, ASK_PRICE3_COL, ASK_PRICE2_COL,
        #     ASK_PRICE1_COL, BID_PRICE1_COL, BID_PRICE2_COL, BID_PRICE3_COL, BID_PRICE4_COL, BID_PRICE5_COL,
        #     BID_VOLUME1_COL, BID_VOLUME2_COL, BID_VOLUME3_COL, BID_VOLUME4_COL, BID_VOLUME5_COL,
        #     HIGHEST_PRICE_COL,LOWEST_PRICE_COL
        # ]
        all_columns = [
            UPDATE_TIME_COL, LAST_PRICE_COL,
            SPECIAL_TIME_COL,ASK_VIB_COL,BID_VIB_COL,DIFF_AB_COL,HIGH_COL,LOW_COL,CLOSE_COL,FEE_COL,
            ASK_VIB_NXT_COL,BID_VIB_NXT_COL,DIFF_AB_NXT_COL,
            # OPEN_PRICE_COL, CLOSE_PRICE_COL, HIGH_PRICE_COL,LOW_PRICE_COL, ASK_PRICE1_COL, BID_PRICE1_COL, BUY_PRICE_COL, SELL_PRICE_COL
        ]
        # 1.Del Rows
        df_data = del_row(df_data)
        # 1.Combined Cols (BUY_PRICE_COL/SELL_PRICE_COL)
        df_data[DATE_COL] = df_data[DATE_COL].astype(str)
        df_data[SPECIAL_TIME_COL] = 0
        df_data[BUY_PRICE_COL] = df_data[ASK_PRICE1_COL].rolling(window=WIN_SIZE, min_periods=1).mean().fillna(
            method='bfill').shift(-WIN_SIZE).fillna(method='ffill')
        df_data[SELL_PRICE_COL] = df_data[BID_PRICE1_COL].rolling(window=WIN_SIZE, min_periods=1).mean().fillna(
            method='bfill').shift(-WIN_SIZE).fillna(method='ffill')

        # 2.Splited Cols(OPEN_PRICE_COL/CLOSE_PRICE_COL/HIGH_PRICE_COL/LOW_PRICE_COL/ACTIVE_FACTOR_COL)
        df_data[CLOSE_PRICE_COL] = df_data[LAST_PRICE_COL]
        df_data[OPEN_PRICE_COL] = df_data[LAST_PRICE_COL].shift(WIN_SIZE - 1).fillna(
            method='bfill')
        df_data[HIGH_PRICE_COL] = df_data[LAST_PRICE_COL].rolling(
            window=WIN_SIZE, min_periods=1).max().fillna(0)
        df_data[LOW_PRICE_COL] = df_data[LAST_PRICE_COL].rolling(
            window=WIN_SIZE, min_periods=1).min().fillna(0)
        # df_data["ask_vol"] = ask_factor
        # df_data["ask_vol"] = df_data["ask_vol"].rolling(window=WIN_SIZE, min_periods=1).sum()
        # df_data["bid_vol"] = bid_factor
        # df_data["bid_vol"] = df_data["bid_vol"].rolling(window=WIN_SIZE, min_periods=1).sum()
        df_data[VOLUME_COL] = df_data[VOLUME_COL].rolling(window=WIN_SIZE, min_periods=1).sum()
        df_data[TURN_OVER_COL] = df_data[TURN_OVER_COL].rolling(window=WIN_SIZE, min_periods=1).sum()

        # 3.Skipped Cols
        df_data = get_skipped_data(df_data, WIN_SIZE)
        df_data[AVG_PRICE_COL] = df_data[TURN_OVER_COL] / (df_data[VOLUME_COL] + CLOSE2ZERO) / NPL

        # 4.New Process
        cal_special_time(df_data)
        cal_env_variables(df_data)

        # *.Final Process
        all_columns = [col_i for col_i in all_columns if col_i in df_data.columns]
        df_data = df_data[all_columns].copy(deep=True)
        return df_data
