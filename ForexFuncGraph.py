paths = input("Enter Path(s):")
col_names = input("Enter Column(s):")

import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
plt.style.use('classic')
import numpy as np


def load(paths):
    paths = paths.split(' ')
    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        dfs.append(df)
    return dfs
# /Users/tnnque/PycharmProjects/data-analytics-ex/que/train_small_cleaned.csv /Users/tnnque/PycharmProjects/data-analytics-ex/que/Test_small_features.csv
def rmv_zero(paths):
    for i in range(0, len(paths)):
        df = paths[i]
        df = df[df["Volume"] != 0]
        paths[i] = df
    return paths

def plot_df(paths, col_names):
    paths = load(paths)
    paths = rmv_zero(paths)
    col_names = col_names.split()
    for df in paths:
        avr_df = []
        for col_name in col_names:
            current_row = None
            old_row = None
            sum_row = 0
            n = 0
            avr_row = []
            for index, row in df.iterrows():
                if index == 0:
                    sum_row = row[col_name]
                    n = 1
                    old_row = row
                    continue
                else:
                    current_row = row
                    if current_row['dayofweek'] == old_row['dayofweek']:
                        sum_row = sum_row + row[col_name]
                        n += 1
                        old_row = row
                    else:
                        avr_row.append(sum_row/n)
                        sum_row = row[col_name]
                        n = 1
                        old_row = row
            avr_row = pd.DataFrame(avr_row)
            plt.plot(avr_row, label=col_name)
        plt.title('Multiple lines graph')
        plt.legend()
        plt.show()

plot_df(paths, col_names)