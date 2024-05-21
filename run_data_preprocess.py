
from App.DataProcess.DataProcess import DataProcess

if __name__ == '__main__':
    lt_futures = ["IF"]
    # pattern = '202301.+?.csv'
    pattern = '.'
    for futures_i in lt_futures:
        DataProcess(futures_i,pattern).run()