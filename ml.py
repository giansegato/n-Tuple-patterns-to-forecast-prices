from utils import *
from sklearn import tree, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import itertools
from scipy.stats import gmean
from sklearn.feature_selection import *
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn import cross_validation

def feature_extraction(dataframe, past_days = 7, use_updowns = True, use_ups = True, use_value = True):
    """
    Preparares the features, extracting the X matrix and the
    y target vector from the dataframe provided
    """

    ks = range(1, past_days+1) # number of possible (past) days took into account: 1 to 7


    # 1st step: defining all the combination of events from t-1 to t-8 wrt t

    events_for_k = {}
    # n = dataframe.shape[0]
    for k in ks:
        array_k = generate_list_of_tuples(dataframe, k)
        del array_k[-1] # delete the last item: we don't care about today (since we want to make a prediction)
        for i in range(k):
            array_k.insert(i, None) # and we don't care for samples we don't have past data for
        events_for_k[k] = array_k
    # matrix of [n * k] size, with all the events *actually* occurred for each k
    actual_events = pd.DataFrame.from_dict(events_for_k)

    possibile_events = [] # now we generate all the possible events
    for k in ks: # for each day up to t-k+1
        possibile_events.extend(itertools.product(range(2), repeat=k)) # all combinations
    # matrix of [n * features] size, with all the events actually occurred
    features = pd.DataFrame(columns = possibile_events)
    for single_possible_event in possibile_events:
        features[single_possible_event] = (actual_events[len(single_possible_event)] == single_possible_event) * 1

    
    # 2n step: defining the sum of all the UPs from t-1 to t-8

    actual_ups = actual_events.apply(lambda x: map(np.sum, x) or 0) # just a sum of the tuples

    translation = {} # in order not to merge the wrong columns (since they share the name)
    for column in actual_ups.columns.values:
        translation[column] = str(column) + "-ups"
    actual_ups = actual_ups.rename(columns = translation)

    # final training set: dataframe + U/D combinations + U count
    training_set_to_use = [dataframe[(['up-down'] + (['value', 'mean', 'min', 'max'] if use_value else []))]]
    if use_updowns: training_set_to_use.append(features)
    if use_ups: training_set_to_use.append(actual_ups)
    training_set = pd.concat(training_set_to_use, axis=1)[k:]

    # final column renaming
    translation = {'up-down': 'y'}
    for column in training_set.columns.values:
        if not isinstance(column, tuple): continue
        translation[column] = combination_to_string(column)
    training_set = training_set.rename(columns = translation)

    # remove perfectly correlated variables
    # np.corrcoef(training_set['U'].values, training_set['D'].values)
    # np.corrcoef(training_set['1-ups'].values, training_set['U'].values)
    training_set = training_set[[key for key in training_set.keys() if (key != 'U' and key != 'D')]]

    # done!
    X = training_set[[key for key in training_set.keys() if "y" not in key]].values
    y = np.asarray(training_set["y"].values)

    return X, y, training_set

def feature_preparation(X, y, preprocess = True):
    """
    Generates the training and the test sets, and pre-process the features (normalization)
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    if preprocess:
        scaler = StandardScaler()
        # Don't cheat - fit only on training data
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        # apply same transformation to test data
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def experiment_algorithms(X, y, algorithms, kfold = False):
    if (kfold):
        results = pd.DataFrame(columns = ['score'])
        for title, algorithms in models.iteritems():
            results.ix[title, 'score'] = ml.compute_model_score(X, y, model)

        return results.sort_values(by=['score'], ascending=False)
    else:
        results = pd.DataFrame(columns = ['accuracy', 'F1', 'precision', 'recall', 'train_score'])

        X_train, X_test, y_train, y_test = feature_preparation(X, y, preprocess)

        for title, algorithm in algorithms.iteritems():
            algorithm.fit(X_train, y_train)
            test_prediction = algorithm.predict(X_test)
            results.ix[title, 'accuracy'] = metrics.accuracy_score(y_test, test_prediction)
            results.ix[title, 'F1'] = metrics.f1_score(y_test, test_prediction)
            results.ix[title, 'precision'] = metrics.precision_score(y_test, test_prediction)
            results.ix[title, 'recall'] = metrics.recall_score(y_test, test_prediction)
            results.ix[title, 'train_score'] = algorithm.score(X_train, y_train)

        return results

def experiment_dataset(algorithm, datasets, preprocess = True):
    results = pd.DataFrame(columns = ['accuracy', 'F1', 'precision', 'recall', 'train_score'])

    for title, dataset in datasets.iteritems():
        X_train, X_test, y_train, y_test = feature_preparation(dataset[0], dataset[1], preprocess)
        algorithm.fit(X_train, y_train)
        test_prediction = algorithm.predict(X_test)
        results.ix[title, 'accuracy'] = metrics.accuracy_score(y_test, test_prediction)
        results.ix[title, 'F1'] = metrics.f1_score(y_test, test_prediction)
        results.ix[title, 'precision'] = metrics.precision_score(y_test, test_prediction)
        results.ix[title, 'recall'] = metrics.recall_score(y_test, test_prediction)
        results.ix[title, 'train_score'] = algorithm.score(X_train, y_train)

    return results

def run_experiment(datasets, algorithms, kfold = False):
    if kfold:
        results = pd.DataFrame(columns = ['score'])
        for title_dataset, dataset in datasets.iteritems():
            for title_algorithm, algorithm in algorithms.iteritems():
                title = title_algorithm+' - '+title_dataset
                print 'Fitting ' + title + '...'
                results.ix[title, 'score'] = compute_model_score(dataset[0], dataset[1], algorithm)

        return results.sort_values(by=['score'], ascending=False)
    else:
        results = pd.DataFrame(columns = ['accuracy', 'f1', 'mean'])

        for title_dataset, dataset in datasets.iteritems():
            for title_algorithm, algorithm in algorithms.iteritems():
                attempt = 1
                title = title_algorithm+' - '+title_dataset
                print 'Fitting ' + title + '...' + ('[{}]'.format(attempt) if attempt > 1 else '')
                successful = False

                while (not successful) and (attempt < 4):
                    attempt += 1

                    X_train, X_test, y_train, y_test = feature_preparation(dataset[0], dataset[1], True)
                    algorithm.fit(X_train, y_train)
                    test_prediction = algorithm.predict(X_test)

                    accuracy = metrics.accuracy_score(y_test, test_prediction)
                    f1 = metrics.f1_score(y_test, test_prediction)

                    results.ix[title, 'accuracy'] = accuracy
                    results.ix[title, 'f1'] = f1
                    results.ix[title, 'gmean'] = gmean((accuracy, f1))

                    successful = metrics.f1_score(y_test, test_prediction) != 0

        return results.sort(['gmean'], ascending=False).head()

def experiment(data = 'inp.csv'):
    dataframe = data_import(data)
    old_columns = dataframe.columns.values.tolist()
    window = dataframe['value'].expanding()
    dataframe = pd.concat([dataframe, window.min(), window.mean(), window.max()], axis=1)
    dataframe.columns = old_columns + ['min', 'mean', 'max']

    a = {}
    a['LogisticRegression'] = LogisticRegression(verbose=False)
    a['LogisticRegression-1K'] = LogisticRegression(C=1000.0, verbose=False)
    a['DecisionTree'] = tree.DecisionTreeClassifier()
    a['NN_(5,2)-1e-2'] = MLPClassifier(alpha=1e-2, hidden_layer_sizes=(5, 2), verbose=False)
    a['NN_(5,2)-1.0'] = MLPClassifier(alpha=1.0, hidden_layer_sizes=(5, 2), verbose=False)
    a['NN_(25,2)-1e-2'] = MLPClassifier(alpha=1e-2, hidden_layer_sizes=(25, 2), verbose=False)
    a['NN_(25,2)-1.0'] = MLPClassifier(alpha=1.0, hidden_layer_sizes=(25, 2), verbose=False)

    d = {}
    for k in range(1, 8):
        d['all_t_{}'.format(k)] = feature_extraction(dataframe, past_days = k)
        d['no_ups_{}'.format(k)] = feature_extraction(dataframe, use_ups = False, past_days = k)
        d['no_updowns_{}'.format(k)] = feature_extraction(dataframe, use_updowns = False, past_days = k)
        d['no_value_{}'.format(k)] = feature_extraction(dataframe, use_value = False, past_days = k)
        d['only_updowns_{}'.format(k)] = feature_extraction(dataframe, use_ups = False, use_value = False, past_days = k)
        d['only_ups_{}'.format(k)] = feature_extraction(dataframe, use_value = False, use_updowns = False, past_days = k)
        d['only_value'] = feature_extraction(dataframe, use_ups = False, use_updowns = False)

    results = run_experiment(d, a)
    print results
    return results

def compute_model_score(X, y, model):
    num_instances = len(X)
    num_folds = 10
    seed = 7
    kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
    results = cross_validation.cross_val_score(model, X, y, cv = kfold)
    return results.mean(), results.std()

def feature_filtering_performance(X, y, filter = f_classif, model = LogisticRegression(), title = 'Performance varying the percentile of features selected'):
    clf = Pipeline([('filter', SelectPercentile(score_func = filter)), ('model', model)])
    score_means = list()
    score_stds = list()
    percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)
    for percentile in percentiles:
        clf.set_params(filter__percentile=percentile)
        this_scores = cross_val_score(clf, X, y)
        score_means.append(this_scores.mean())
        score_stds.append(this_scores.std())
    plt.errorbar(percentiles, score_means, np.array(score_stds))
    plt.title(title)
    plt.xlabel('Percentile')
    plt.ylabel('Prediction rate')
    plt.axis('tight')
    plt.show(block=False)

def hyperparams_filtering_performance(X, y, model = LogisticRegression()):
    # todo

def feature_analysis(dataframe):
    old_columns = dataframe.columns.values.tolist()
    window = dataframe['value'].expanding()
    dataframe = pd.concat([dataframe, window.min(), window.mean(), window.max()], axis=1)
    dataframe.columns = old_columns + ['min', 'mean', 'max']
    X, y, training_set = ml.feature_extraction(dataframe, past_days = 7)

    #np.corrcoef(training_set['U'].values, training_set['D'].values)
    #np.corrcoef(training_set['mean'].values, y)

    # test = SelectKBest(score_func=chi2, k=4).fit(X, y) # Only for non-negative features
    test = f_classif(X, y)
    test2 = SelectKBest(score_func=mutual_info_classif).fit(X, y) # see https://www.cs.utah.edu/~piyush/teaching/22-9-print.pdf
    test3 = ExtraTreesClassifier().fit(X, y)
    test4 = RFE(SGDClassifier(alpha=1.0)).fit(X, y)

    rank = pd.DataFrame(test.scores_, index = training_set[[key for key in training_set.keys() if "y" not in key]].columns.values, columns=['f_classif'])
    rank['mutual_info_classif'] = test2.scores_
    rank['ExtraTreesClassifier'] = test3.feature_importances_
    rank['RFE'] = test4.ranking_
    rank['RandomForestClassifier'] = RandomForestClassifier().feature_importances_
    rank = rank.sort_values(by='f_classif', ascending=False)
    rank = rank.sort_values(by='mutual_info_classif', ascending=False)
    rank = rank.sort_values(by='ExtraTreesClassifier', ascending=False)
    rank = rank.sort_values(by='RFE')

    selection = pd.DataFrame(index = training_set[[key for key in training_set.keys() if "y" not in key]].columns.values)
    selection['f_classif'] = f_classif(X, y)[0]
    selection['mutual_info_classif'] = mutual_info_classif(X, y)[0]
    selected_features = [x for x in selection.sort_values(by='f_classif', ascending=False).head(selection.shape[0]//5).index.values.tolist()\
        if x in selection.sort_values(by='mutual_info_classif', ascending=False).head(selection.shape[0]//4).index.values.tolist()]

    X_new = training_set[[feature for feature in selected_features]].values

    ml.feature_filtering_performance(X, y, model = LogisticRegression(C=100.0), title = 'Performance of the LogisticRegression-ANOVA varying the percentile of features selected')

    transform = SelectPercentile(feature_selection.f_classif)
    #clf = Pipeline([('anova', transform), ('svc', svm.SVC(C=1.0))])
    clf = Pipeline([('filter', SelectPercentile()), ('svc', svm.SVC(C=1.0))])
    score_means = list()
    score_stds = list()
    percentiles = (1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100)
    for percentile in percentiles:
        clf.set_params(filter__percentile=percentile)
        this_scores = cross_val_score(clf, X, y)
        score_means.append(this_scores.mean())
        score_stds.append(this_scores.std())
    plt.errorbar(percentiles, score_means, np.array(score_stds))
    plt.title('Performance of the SVM-Mutual Information varying the percentile of features selected')
    plt.title('Performance of the LogisticRegression-ANOVA varying the percentile of features selected')
    plt.xlabel('Percentile')
    plt.ylabel('Prediction rate')
    plt.axis('tight')
    plt.show()

    X_new = SelectPercentile(f_classif, percentile=5).fit_transform(X, y)
    scaler = StandardScaler()
    scaler.fit(X_new)
    X_new = scaler.transform(X_new)