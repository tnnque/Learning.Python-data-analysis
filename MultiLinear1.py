import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series

path = '/Users/tnnque/PycharmProjects/data-analytics-ex/que/train_small_cleaned.csv'
# input("Enter path:")
col_names = 'Open SMA_50 SMA_20 dayofweek'
# input("Enter dependent/independent variables:")

# def extract(path, col_names):
#     df = pd.read_csv(path)
#     col_names = col_names.split()
#     df = df[col_names]
#     current_row = None
#     old_row = None
#     group_row = pd.DataFrame()
#     new_df =  pd.DataFrame()
#     for index, row in df.iterrows():
#         if index == 0:
#             old_row = row
#             temp = df.ix[index]
#             group_row = group_row.append(temp)
#         else:
#             if row['dayofweek'] == old_row['dayofweek']:
#                 old_row = row
#                 temp = df.ix[index]
#                 group_row = group_row.append(temp)
#             else:
#                 old_row = row
#                 temp_sum = (group_row.sum(axis=0)) / len(group_row)
#                 new_df = new_df.append(temp_sum, ignore_index=True)
#                 group_row = pd.DataFrame()
#
#     Y = new_df.iloc[:, :1]
#     X = new_df.iloc[:, 1:]
#     return X, Y

def extract(path, col_names):
    df = pd.read_csv(path)
    col_names = col_names.split()
    df = df[col_names]
    current_row = None
    old_row = None
    start = 0
    end = 0
    new_df = pd.DataFrame()
    X = pd.DataFrame()
    for index, row in df.iterrows():
        if index == 0:
            old_row = row
            start = 0
            end = 1
        else:
            if row['dayofweek'] == old_row['dayofweek']:
                old_row = row
                end += 1

            else:
                old_row = row
                group = df.iloc[start : end]
                temp = (group.sum(axis= 0)) / len(group)
                new_df = new_df.append(temp, ignore_index= True)
                start = end
                end = end + 1

    X = new_df.iloc[:, 1:]
    Y = new_df.iloc[:, :1]

    temp_array = X.to_numpy(copy=True)
    X = np.insert(temp_array, 0, np.ones(len(X), ), axis= 1)
    Y = Y.to_numpy(copy=True)
    return X, Y

def hypothesis(X, B):
    return X.dot(B)

extract(path, col_names)