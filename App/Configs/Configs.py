import os
import pandas as pd
from App.Configs.AppConfigs import *
import sys

CODE_PATH = os.path.dirname(os.path.dirname(__file__))
PROJECT_PATH = os.path.dirname(CODE_PATH)
RAW_DATA_PATH = os.path.join(PROJECT_PATH, 'InputRaw')
DATA_PATH = os.path.join(PROJECT_PATH, 'Input')
OUTPUT_PATH = os.path.join(PROJECT_PATH, 'Output')
OUTPUT_MODEL_PATH = os.path.join(PROJECT_PATH, 'OutputModel')
PACKAGE_PATH = os.path.join(PROJECT_PATH, "App", 'Derivative')
APP_CONFIGS_DIR = os.path.join(PROJECT_PATH, "AppConfigs")
sys.path.append(CODE_PATH)
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(PACKAGE_PATH, exist_ok=True)
os.makedirs(OUTPUT_MODEL_PATH, exist_ok=True)
os.makedirs(APP_CONFIGS_DIR, exist_ok=True)

APP_CONFIGS_FILE = os.path.join(APP_CONFIGS_DIR, 'AppConfigs.tsv')

df_app_info = pd.read_csv(APP_CONFIGS_FILE, index_col=USER_COL, sep='\t')
CUR_USER, DT_CONFIGS_APP = list(df_app_info.loc[df_app_info[SELECTED_COL], :].T.to_dict().items())[0]
