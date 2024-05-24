import os
from App.Configs.Configs import RAW_DATA_PATH
from App.Utils.parallel_computing import MultiProcess

def download_apply_func(params):
    futures_i,month_i = params
    source = f"wanghaoheng@192.168.1.107:/data1/yujiateng/futures/output/merge/top1/result/{futures_i}/{month_i}"
    dest = os.path.join(RAW_DATA_PATH,futures_i,month_i)
    os.makedirs(dest,exist_ok=True)
    cmd = f"scp -r {source} {dest}/"
    print(cmd)
    os.system(cmd)

def download(lt_futures,years=None,months=None):
    if years is None:
        years = [2019,2020,2021,2022,2023,2024,2025]
    if months is None:
        months = range(1,13)
    tasks = []
    for futures_i in lt_futures:
        for Year_i in years:
            for Month_i in months:
                month_i = f'{Year_i}{Month_i:02}'
                tasks.append((futures_i, month_i))
    MultiProcess.multi_process(download_apply_func, tasks)

def sync_apply_func(futures):
    source = f"wanghaoheng@192.168.1.107:/data1/yujiateng/futures/output/merge/top1/result/{futures}/"
    dest = os.path.join(RAW_DATA_PATH,futures)
    os.makedirs(dest,exist_ok=True)
    cmd = f"rsync -rvlt --timeout=30 {source} {dest}"
    print(cmd)
    os.system(cmd)

def sync_data(lt_futures):
    MultiProcess.multi_process(sync_apply_func,lt_futures)


if __name__ == '__main__':
    lt_futures = ["IF"]
    """1.download specific file"""
    # Years = ['2024']
    # Months = range(1, 13)
    # download(lt_futures,Years,Months)
    # # download(lt_futures)

    """2.Sync"""
    sync_data(lt_futures)