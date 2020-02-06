import pandas as pd
# import numpy as np
from pandas import DataFrame, Series
# path = '/Users/tnnque/Downloads/super_small_forex.csv'
path = '/Users/tnnque/PycharmProjects/data-analytics-ex/que/train_small_cleaned.csv'
df = pd.read_csv(path)
current_row = None
old_row = None
sum_row = 0
avr_df = []

for index, row in df.iterrows():
    if int(row['Volume']) == 0:
        df.drop(index, inplace=True)

for index, row in df.iterrows():
    if index == 0:
        sum_row = row['High'] - row['Low']
        old_row = row
        n = 1
        continue
    else:
        current_row = row
        if current_row['dayofweek'] == old_row['dayofweek']:
            sum_row = sum_row + row['High'] - row['Low']
            n += 1
        else:
            avr_df.append(sum_row/n)
            old_row = row
            n = 1
            sum_row = 0

avr_df = pd.DataFrame(avr_df)

import matplotlib.pyplot as plt
avr_df.plot(title='Average Per Day')
# plt.scatter([x for x in range(0, len(avr_df))], avr_df)
plt.show()
