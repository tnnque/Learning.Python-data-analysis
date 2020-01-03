## 1.USA.GOV DATA FROM BIT.LY
path = '/Users/tnnque/PycharmProjects/data-analytics-ex/datasets/bitly_usagov/example.txt'
print(open(path).readline())

import json
import matplotlib.pyplot as plt
path = '/Users/tnnque/PycharmProjects/data-analytics-ex/datasets/bitly_usagov/example.txt'
records = [json.loads(line) for line in open(path)]
print(records[0]['tz'])
Counting Time Zones
time_zones = [rec['tz'] for rec in records if 'tz' in rec]
print(time_zones[:10])

def get_counts(sequence):
    counts = dict()
    for c in sequence:
        if c not in counts:
            counts[c] = 1
        else:
            counts[c] += 1
    return counts

from collections import defaultdict
def get_counts2(sequence):
    counts = defaultdict(int)
    for c in sequence:
        counts[c] += 1
    return counts

counts = get_counts2(time_zones)
print(counts['America/New_York'])
print(len(time_zones))

def top10 (count_dict, n = 10):
    pair = [(count, tz) for tz, count in count_dict.items()]
    pair.sort()
    return pair[-n:]
print(top10(counts))

from collections import Counter
counts = Counter(time_zones)
print(counts.most_common(10))

from pandas import DataFrame, Series
import pandas as pd
frame = DataFrame(records)
print(frame)

print(frame['tz'][:10])

tz_counts = frame['tz'].value_counts()
print(tz_counts[:10])

clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz == ''] = 'Unknown'
tz_counts = clean_tz.value_counts()

tz_counts[:10].plot(kind= 'barh', rot = 0)
print(frame['a'][51])

results = Series([x.split()[0] for x in frame.a.dropna()])
print(results.value_counts()[:8])

import numpy as np
cframe = frame[frame.a.notnull()]
operating_system = np.where(cframe['a'].str.contains('Windows'), 'Windows', 'Not Windows')
print(operating_system[:5])

by_tz_os = cframe.groupby(['tz', operating_system])
agg_counts = by_tz_os.size().unstack().fillna(0)

indexer = agg_counts.sum(1).argsort()
print(indexer[:10])

count_subset = agg_counts.take(indexer[-10:])
print(count_subset)

count_subset.plot(kind='barh', stacked=True)
plt.show()
normed_subset = count_subset.div(count_subset.sum(1), axis=0)
normed_subset.plot(kind='barh', stacked=True)
plt.show()

path = '/Users/tnnque/PycharmProjects/data-analytics-ex/que/ch02.py'
open(path).readline()

## MOVIELENS 1M DATA SET
import pandas as pd
unames = ['user_id','gender', 'age', 'occupation', 'zip']
users = pd.read_table('/Users/tnnque/PycharmProjects/data-analytics-ex/datasets/movielens/users.dat', sep='::', header=None, names=unames)
rnames = ['user_id','movie_id', 'rating', 'timestamp']
ratings = pd.read_table('/Users/tnnque/PycharmProjects/data-analytics-ex/datasets/movielens/ratings.dat', sep='::', header=None, names=rnames)
mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('/Users/tnnque/PycharmProjects/data-analytics-ex/datasets/movielens/movies.dat', sep='::', header=None, names=mnames)
# print(users[:5])
# print(movies[:5])
# print(ratings)
data = pd.merge(pd.merge(ratings, users), movies)
# print(data.ix[0])
mean_ratings = data.pivot_table('rating','title',columns='gender',aggfunc='mean')
# print(mean_ratings[:5])
ratings_by_title = data.groupby('title').size()
# print(ratings_by_title[:10])
active_titles =  ratings_by_title.index[ratings_by_title >= 250]
# print(active_titles)
mean_ratings = mean_ratings.ix[active_titles]
# print(mean_ratings)
top_female_ratings = mean_ratings.sort_index(by='F', ascending=False)
# print(top_female_ratings[:10])
mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
sorted_by_diff = mean_ratings.sort_index(by='diff')
# print(sorted_by_diff[:15])
# print(sorted_by_diff[::-1][:15])
rating_std_by_title = data.groupby('title')['rating'].std()
rating_std_by_title = rating_std_by_title.ix[active_titles]
# print(rating_std_by_title.sort_values(ascending=False)[:10])

## US BABY NAMES 1880 - 2010
import pandas as pd

names1880 = pd.read_csv('/Users/tnnque/PycharmProjects/data-analytics-ex/datasets/babynames/yob1880.txt', names=['name', 'sex', 'births'])
# print(names1880)
# print(names1880.groupby('sex').births.sum())
years = range(1880, 2011)
pieces = []
columns = ['name', 'sex', 'births']
for year in years:
    path = '/Users/tnnque/PycharmProjects/data-analytics-ex/datasets/babynames/yob%d.txt' % year
    frame = pd.read_csv(path, names=columns)

    frame['year'] = year
    pieces.append(frame)
names = pd.concat(pieces, ignore_index=True)
total_births = names.pivot_table('births', 'year', columns='sex', aggfunc=sum)
# print(total_births.tail())
import matplotlib.pyplot as plt
total_births.plot(title='Total births by sex and year')
# plt.show()
def add_prop(group):
    births = group.births.astype(float)
    group['prop']= births/births.sum()
    return group
names = names.groupby(['year', 'sex']).apply(add_prop)
# print(names)
import numpy as np
np.allclose(names.groupby(['year', 'sex']).prop.sum(), 1)
#taketop1000 method1
def get_top1000(group):
    return group.sort_index(by='births', ascending=False)[:1000]
grouped = names.groupby(['year', 'sex'])
top1000 = grouped.apply(get_top1000)
#taketop1000 method2
pieces = []
for year, group in names.groupby(['year','sex']):
    pieces.append(group.sort_index(by='births', ascending=False)[:1000])
top1000 = pd.concat(pieces, ignore_index=True)
#Analyzing Naming Trends
boys = top1000[top1000.sex == 'M']
girls = top1000[top1000.sex == 'F']
total_births = top1000.pivot_table('births', 'year', columns='name', aggfunc=sum)
subset = total_births[['John', 'Harry', 'Mary', 'Marilyn']]
subset.plot(subplots=True, figsize=(12, 10), grid=False, title='Number of births per year')

table = top1000.pivot_table('prop', 'year', columns='sex', aggfunc=sum)
table.plot(title='Sum of table1000.prop by year and sex', yticks=np.linspace(0, 1.2, 13), xticks=range(1880, 2020, 10))
# plt.show()
df = boys[boys.year == 2010]
prop_cumsum = df.sort_index(by='prop', ascending=False).prop.cumsum()
# print(prop_cumsum[:10])
print(prop_cumsum.searchsorted(0.5))
df = boys[boys.year == 1900]
in1900 = df.sort_index(by='prop', ascending=False).prop.cumsum()
print(in1900.searchsorted(0.5) + 1)

def get_quantile_count(group, q=0.5):
    group = group.sort_index(by='prop', ascending=False)
    return group.prop.cumsum().searchsorted(q) + 1

diversity = top1000.groupby(['year', 'sex']).apply(get_quantile_count)
diversity = diversity.unstack('sex')
# print(diversity.head())
diversity.plot(title="Number of popular names in top 50%")
# plt.show()
# Extract last letter from name column
get_last_letter = lambda x: x[-1]
last_letters = names.name.map(get_last_letter)
last_letters.name = 'last_letter'
table = names.pivot_table('births', last_letters, columns=['sex','year'], aggfunc=sum)
subtable = table.reindex(columns=[1910, 1960, 2010], level='year')
print(subtable.sum())
letter_prop = subtable/subtable.sum().astype(float)
fig, axes = plt.subplots(2, 1, figsize=(10,8))
letter_prop['M'].plot(kind='bar', rot=0, ax=axes[0], title='Male')
letter_prop['F'].plot(kind='bar', rot=0, ax=axes[1], title='Female')

letter_prop = table/table.sum().astype(float)
dny_ts = letter_prop.ix[['d', 'n', 'y'], 'M'].T
# print(dny_ts.head())

all_names = top1000.name.unique()
mask = np.array(['lesl' in x.lower() for x in all_names])
lesley_like = all_names[mask]
print(lesley_like)
filtered = top1000[top1000.name.isin(lesley_like)]
# print(filtered.groupby('name').births.sum())
table = filtered.pivot_table('births', 'year', columns='sex', aggfunc='sum')
table = table.div(table.sum(1), axis=0)
# print(table.tail())
# print(table.plot(style={'M': 'k-', 'F': 'k--'}))
# plt.show()