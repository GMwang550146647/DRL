
import logging
import re
import pandas as pd

from App.Configs.ConfigsFutures import *

class FuturesBase():
    def __init__(self, task_name="Base", *args, **kwargs):
        self.task_name = task_name

    @staticmethod
    def get_numperlot(symbol_name):
        symbol_name = FuturesBase.get_symbol_name(symbol_name)
        try:
            return NUM_PER_LOT[symbol_name]
        except Exception as err:
            logging.error(f"NUMPERLOT NO FOUND : {symbol_name} , details -> {err}")
            exit(1)

    @staticmethod
    def get_fee(symbol_name, price, volume):
        symbol_name = FuturesBase.get_symbol_name(symbol_name)
        try:
            fee = TRAN_FEE_PER_LOT[symbol_name]
            if 'value' in fee:
                return fee['value'] * volume / float(FuturesBase.get_numperlot(symbol_name))
            else:
                return fee['ratio'] * price * volume
        except Exception as err:
            logging.error(f"GET FEE : No Fee Info of {symbol_name} is found !")
            exit(1)

    @staticmethod
    def get_min_diff(symbol_name):
        symbol_name = FuturesBase.get_symbol_name(symbol_name)
        try:
            min_div = MIN_DIV[symbol_name]
        except Exception as err:
            min_div = None
            logging.error(f"GET MIN_DIFF : No MinDiff Info of {symbol_name} is found !")
            exit(1)
        return min_div

    @staticmethod
    def get_symbol_name(symbol):
        pattern = r"(^[a-zA-Z]+)"
        result = re.findall(pattern,symbol)
        if len(result)>0:
            return result[0]
        else:
            return ""

    def load_df_data(self, filename):
        try:
            if re.findall(r'.tsv', filename):
                return pd.read_csv(filename, sep='\t')
            elif re.findall(r'.csv', filename):
                return pd.read_csv(filename)
            else:
                return pd.DataFrame()
        except Exception as err:
            logging.error(f"LOAD DF_DATA: Error Loading {filename} , Reason -> {err}")
            return pd.DataFrame()