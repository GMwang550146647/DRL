from concurrent import futures
from tqdm import tqdm

import pandas as pd

__author__ = "Aaron Wang"


class MultiThread():
    @staticmethod
    def multi_thread(func, data, **kwargs):
        """
        PS:
            n_threads=kwargs['max_workers']
            if key 'max_workers' is not in kwargs, the ThreadPool will judge by itself
        :param data: Dict, tuple, list or nested
        :return:
        """
        all_results = []
        if type(data) == pd.core.frame.DataFrame:
            data = data.reset_index(drop=True)  # This is very important!
            with futures.ThreadPoolExecutor(**kwargs) as executor:
                data = list(data.T.to_dict().values())
                for resp in tqdm(executor.map(func, data), total=len(data)):
                    all_results.append(resp)
            all_results = pd.DataFrame(all_results)
        elif type(data) in [tuple, list] and type(data[0]) in [tuple, list, dict]:
            with futures.ThreadPoolExecutor(**kwargs) as executor:
                for resp in tqdm(executor.map(func, data), total=len(data)):
                    all_results.append(resp)
        elif type(data) in [tuple, list]:
            with futures.ThreadPoolExecutor(**kwargs) as executor:
                for resp in tqdm(executor.map(func, data), total=len(data)):
                    all_results.append(resp)
        else:
            with futures.ThreadPoolExecutor(**kwargs) as executor:
                for resp in tqdm(executor.map(func, data)):
                    all_results.append(resp)
        return all_results

    @staticmethod
    def decorator(**kwargs):
        """
        decorator for multithread
        PS:
            n_threads=kwargs['max_workers']
            if key 'max_workers' is not in kwargs, the ThreadPool will judge by itself
        """

        def multi_thread_wrapper(func):
            def inner_warpper(data):
                """
                :param data: Dict, tuple, list or nested
                :return:
                """
                all_results = []
                if type(data) == pd.core.frame.DataFrame:
                    data = data.reset_index(drop=True)  # This is very important!
                    with futures.ThreadPoolExecutor(**kwargs) as executor:
                        data = list(data.T.to_dict().values())
                        for resp in tqdm(executor.map(func, data), total=len(data)):
                            all_results.append(resp)
                    all_results = pd.DataFrame(all_results)
                elif type(data) in [tuple, list] and type(data[0]) in [tuple, list, dict]:
                    with futures.ThreadPoolExecutor(**kwargs) as executor:
                        for resp in tqdm(executor.map(func, data), total=len(data)):
                            all_results.append(resp)
                elif type(data) in [tuple, list]:
                    with futures.ThreadPoolExecutor(**kwargs) as executor:
                        for resp in tqdm(executor.map(func, data), total=len(data)):
                            all_results.append(resp)
                else:
                    with futures.ThreadPoolExecutor(**kwargs) as executor:
                        for resp in tqdm(executor.map(func, data)):
                            all_results.append(resp)
                return all_results

            return inner_warpper

        return multi_thread_wrapper


class MultiProcess():
    @staticmethod
    def multi_process(func, data, **kwargs):
        """
        PS:
            n_threads=kwargs['max_workers']
            if key 'max_workers' is not in kwargs, the ThreadPool will judge by itself
        :param data: Dict, tuple, list or nested
        :return:
        """
        all_results = []
        if type(data) == pd.core.frame.DataFrame:
            data = data.reset_index(drop=True)  # This is very important!
            with futures.ProcessPoolExecutor(**kwargs) as executor:
                data = list(data.T.to_dict().values())
                for resp in tqdm(executor.map(func, data), total=len(data)):
                    all_results.append(resp)
            all_results = pd.DataFrame(all_results)
        elif type(data) in [tuple, list] and type(data[0]) in [tuple, list, dict]:
            with futures.ProcessPoolExecutor(**kwargs) as executor:
                for resp in tqdm(executor.map(func, data), total=len(data)):
                    all_results.append(resp)
        elif type(data) in [tuple, list]:
            with futures.ProcessPoolExecutor(**kwargs) as executor:
                for resp in tqdm(executor.map(func, data), total=len(data)):
                    all_results.append(resp)
        else:
            with futures.ProcessPoolExecutor(**kwargs) as executor:
                for resp in tqdm(executor.map(func, data)):
                    all_results.append(resp)
        return all_results

    @staticmethod
    def decorator(**kwargs):
        """
        decorator for multiprocess
        PS:
            n_process=kwargs['max_workers']
            if key 'max_workers' is not in kwargs, the ThreadPool will judge by itself
        """

        def multi_process_wrapper(func):
            def inner_warpper(data):
                """
                :param data: Dict, tuple, list or nested
                :return:
                """
                all_results = []
                if type(data) == pd.core.frame.DataFrame:
                    data = data.reset_index(drop=True)  # This is very important!
                    with futures.ProcessPoolExecutor(**kwargs) as executor:
                        data = list(data.T.to_dict().values())
                        for resp in tqdm(executor.map(func, data), total=len(data)):
                            all_results.append(resp)
                    all_results = pd.DataFrame(all_results)
                elif type(data) in [tuple, list] and type(data[0]) in [tuple, list, dict]:
                    with futures.ProcessPoolExecutor(**kwargs) as executor:
                        for resp in tqdm(executor.map(func, data), total=len(data)):
                            all_results.append(resp)
                elif type(data) in [tuple, list]:
                    with futures.ProcessPoolExecutor(**kwargs) as executor:
                        for resp in tqdm(executor.map(func, data), total=len(data)):
                            all_results.append(resp)
                else:
                    with futures.ProcessPoolExecutor(**kwargs) as executor:
                        for resp in tqdm(executor.map(func, data)):
                            all_results.append(resp)
                return all_results

            return inner_warpper

        return multi_process_wrapper
