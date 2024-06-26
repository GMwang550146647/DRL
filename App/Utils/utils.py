import os
import re
import ray
import pandas as pd


def get_all_pattern_files(path, pattern, rel=False):
    target_files = []

    def abs_search(path):
        all_paths = os.listdir(path)
        for pathi in all_paths:
            pathi = os.path.join(path, pathi)
            if os.path.isdir(pathi):
                abs_search(pathi)
            elif re.findall(pattern, pathi):
                target_files.append(pathi)

    def rel_search(path, pre=""):
        all_paths = os.listdir(path)
        for pathi in all_paths:
            abs_path_i = os.path.join(path, pathi)
            rel_path_i = os.path.join(pre, pathi)
            if os.path.isdir(abs_path_i):
                rel_search(abs_path_i, rel_path_i)
            elif re.findall(pattern, rel_path_i):
                target_files.append(rel_path_i)

    if rel:
        rel_path = path.split('/')[-1]
        rel_search(path, rel_path)
    else:
        abs_search(path)
    return target_files


def concat_df_tree(list_df):
    def split_and_concat(start, end):
        if start >= end:
            return list_df[start]
        else:
            mid = int((start + end) / 2)
            left = split_and_concat(start, mid)
            right = split_and_concat(mid + 1, end)
            return pd.concat((left, right), axis=0)

    result = split_and_concat(0, len(list_df) - 1)
    result.reset_index(inplace=True, drop=True)
    return result


def get_all_packages(path, pattern=".py$"):
    files = get_all_pattern_files(path, pattern, rel=True)
    packages = [file_i.split('.')[0].replace('/', '.') for file_i in files]
    return packages


def convert2dc(Obj, *o_args, **o_kwargs):
    @ray.remote
    def inner(*args, **kwargs):
        if type(args[1]) in [list, tuple]:
            return [Obj(*o_args, **o_kwargs).run(args[0], param_i, **kwargs) for param_i in args[1]]
        else:
            return Obj(*o_args, **o_kwargs).run(*args, **kwargs)

    return inner


def convert2dc_sv(Obj, *o_args, **o_kwargs):
    @ray.remote
    def inner(*args, **kwargs):
        return Obj(*o_args, **o_kwargs).run(*args, **kwargs)

    return inner


# def convert2dc(Obj, *o_args, **o_kwargs):
#     @ray.remote
#     def inner(*args, **kwargs):
#         if type(args[1]) in [list, tuple]:
#             result = [Obj(*o_args, **o_kwargs).run(args[0], param_i, **kwargs) for param_i in args[1]]
#         else:
#             result = Obj(*o_args, **o_kwargs).run(*args, **kwargs)
#         result_id = ray.put(result)
#         return result_id
#
#     return inner

def concat_list(lt):
    def split_and_concat(start, end):
        if start >= end:
            return lt[start]
        else:
            mid = int((start + end) / 2)
            left = split_and_concat(start, mid)
            right = split_and_concat(mid + 1, end)
            return left + right

    return split_and_concat(0, len(lt) - 1)


def flatten_list(lt):
    def dfs(tempt_lt):
        if type(tempt_lt) == list:
            for item_i in tempt_lt:
                dfs(item_i)
        else:
            result.append(tempt_lt)

    result = []
    dfs(lt)
    return result


def merge(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = []
    for interval in intervals:
        # 如果列表为空，或者当前区间与上一区间不重合，直接添加
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            if interval[1] is None:
                merged[-1][1] = None
            else:
                # 否则的话，我们就可以与上一区间进行合并
                merged[-1][1] = max(merged[-1][1], interval[1])

    return merged


def concat_df_tree(list_df):
    def split_and_concat(start, end):
        if start >= end:
            return list_df[start]
        else:
            mid = int((start + end) / 2)
            left = split_and_concat(start, mid)
            right = split_and_concat(mid + 1, end)
            return pd.concat((left, right), axis=0)

    result = split_and_concat(0, len(list_df) - 1)
    result.reset_index(inplace=True, drop=True)
    return result
