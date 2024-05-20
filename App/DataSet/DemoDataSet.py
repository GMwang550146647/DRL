import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from App.Configs.Configs import DATA_PATH
from App.Utils.utils import get_all_pattern_files, concat_df_tree

from App.Utils.parallel_computing import MultiProcess


# Prepare the dataset
class DemoDateset(Dataset):
    # 加载数据集
    def __init__(self, filepath_pattern):
        all_data = get_all_pattern_files(DATA_PATH, filepath_pattern)

        xy = self.load_data(all_data)
        self.len = xy.shape[0]  # shape[0]是矩阵的行数,shape[1]是矩阵的列数
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def load_data(self, file_path):
        df_files = MultiProcess.multi_process(pd.read_csv, file_path)
        df_concat = concat_df_tree(df_files)
        return df_concat.values.astype(dtype=np.float32)

    # 获取数据索引
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # 获得数据总量
    def __len__(self):
        return self.len
