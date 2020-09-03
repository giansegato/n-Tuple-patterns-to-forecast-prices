from __future__ import division # if you'll ever need integer division: c = a // b
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
import matplotlib
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter
#import seaborn as sns

def data_import(filename = 'inp.csv'):
    dataframe = pd.read_csv(filename)[['yy', 'value']]
    dataframe['decade'] = dataframe['yy'].apply(lambda yy: yy//10) # genera la decade
    dataframe['up-down'] = 1
    dataframe['up-down'] = dataframe['up-down'] * (dataframe['value'] > dataframe['value'].shift(1)) # definisce l'up-down
    return dataframe

def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)
    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'

def generate_list_of_tuples(df, k):
    # crea un array di arrays, shiftati di valore di volta in volta
    list_of_values = [df['up-down'][i:].tolist() for i in range(k)]
    # poi li zippa sulla base della loro posizione, creando in questo modo
    # una lista di tuple che rappresenta tutte le sequenze di valori che avvengono
    # nella lista di up and down
    list_of_tuples = list(zip(*list_of_values))
    return list_of_tuples

def count_unique_values_sum(df, k):
    list_of_tuples = generate_list_of_tuples(df, k)
    list_of_tuples = map(sum, list_of_tuples)
    # conta i valori univoci negli array
    counts = pd.value_counts(list_of_tuples)
    # ritorna lo zip corrispondente ai risultati
    return zip(counts.index, counts.values)

def count_unique_values_combinations(df, k):
    list_of_tuples = generate_list_of_tuples(df, k)
    # conta i valori univoci di combinazioni
    counts = pd.value_counts(list_of_tuples)
    # ritorna lo zip corrispondente ai risultati
    return zip(counts.index, counts.values)

def columns_to_percent(df):
    for column in df.columns.values:
        df[column] = df[column]/df[column].sum()
    return df

def count_only_up(dataframe):
    results_sum = {}
    decades = dataframe['decade'].unique()

    for k in range(1, 8): # da 1 a 7 giorni di combinazioni possibili
        results_sum[k] = pd.DataFrame(columns = decades, index = range(k))
        for decade in decades: # per ogni decade
            df = dataframe[dataframe['decade'] == decade]
            counts = count_unique_values_sum(df, k)
            for count in counts:
                results_sum[k].ix[count[0], decade] = count[1]
    
    # to percent
    for key, value in results_sum.iteritems():
        results_sum[key] = columns_to_percent(results_sum[key])

    return results_sum

def count_up_down(dataframe):
    combinations = []
    results = {}
    decades = dataframe['decade'].unique()

    for k in range(1, 8): # da 1 a 7 giorni di combinazioni possibili
        r_for_k = [item for item in itertools.product(range(2), repeat=k)]
        combinations.extend(r_for_k)
        results[k] = pd.DataFrame(columns = decades, index = r_for_k)

    df_counts_all = pd.DataFrame(columns = decades, index = combinations)
    df_counts_all = df_counts_all.fillna(0)

    for k in range(1, 8):
        for decade in decades: # per ogni decade
            df = dataframe[dataframe['decade'] == decade]
            counts = count_unique_values_combinations(df, k)
            for count in counts:
                df_counts_all.ix[[count[0]], decade] = count[1]
                results[k].ix[[count[0]], decade] = count[1]

    # Converte i valori da assoluti a percentuali
    for key, value in results.iteritems():
        results[key] = columns_to_percent(results[key])

    results['all'] = df_counts_all
    return results

def forecast_dominant_event(dataframe):

    df = count_up_down(dataframe)['all']
    df = df.sum(axis=1)
    dominant_patterns = {}
    for index, combination in enumerate(df.index.values):
        if (index != 0) and (combination[:-1] == df.index.values[index-1][:-1]):
            first, second = combination, df.index.values[index-1]
            dominant = first if (df.ix[[first]].values[0] > df.ix[[second]].values[0]) else second 
            dominant_patterns[combination[:-1]] = dominant[-1]

    events_for_k = {}
    for k in range(1,8):
        array_k = generate_list_of_tuples(dataframe, k)
        del array_k[-1]
        for i in range(k):
            array_k.insert(i, None)
        events_for_k[k] = array_k
        
    actual_events = pd.DataFrame.from_dict(events_for_k)
    actual_events = actual_events[[key for key in actual_events.keys() if key != 7]][k:]

    actual_events = actual_events.applymap(lambda x: dominant_patterns[x])

    results = {}
    for column in actual_events.columns.values:
        if isinstance(column, (int, long)):
            results[column] = ((actual_events[column] == dataframe['up-down'][k:]) * 1).mean()

    return results

def combination_to_string(comb):
    string = ""
    for item in comb:
        string += "U" if item == 1 else "D"
    return string

def rename_indexes(df):
    dict = {}
    for indexes in df.index.values:
        if not isinstance(indexes, tuple): break
        dict[indexes] = combination_to_string(indexes)
    return df.rename(index=dict)

def rename_columns(df):
    dict = {}
    for column in df.columns.values:
        dict[column] = ("20" if column < 2 else "19") + str(column) + "0s "
    return df.rename(columns=dict)

def plot(df, k, title): 
    # see: http://emptypipes.org/2013/11/09/matplotlib-multicategory-barchart/
    fig = plt.figure()
    ax = fig.add_subplot(111)
    space = 0.3

    df = rename_columns(rename_indexes(df))
    conditions = df.index.values # ie. combinations
    categories = df.columns.values # ie. decades
    n = len(conditions)
    width = (1 - space) / (len(conditions))

    for i, combination in enumerate(conditions):
        indeces = range(1, len(categories)+1)
        vals = df.ix[combination].tolist()
        pos = [j - (1 - space) / 2. + i * width for j in range(1,len(categories)+1)]
        rects = ax.bar(pos, vals, width=width, label=combination, color=cm.Accent(float(i) / n))
        if (k > 3): continue # skip the % labeling with k is too big
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.08*height, '{0:.2f}%'.format(height*100), ha='center', va='bottom', rotation = 90)

    ax.set_ylabel("% of occurrence")
    ax.set_title(title)
    ax.set_xticks(indeces)
    ax.set_xticklabels(categories)
    ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
    ax.set_ylim((ax.get_ylim()[0], ax.get_ylim()[1]*1.25))

    box = ax.get_position() # Shrink current axis's height by 10% on the bottom
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5) # Put a legend below current axis

    plt.show(block=False)