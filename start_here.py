from utils import *
import ml
from sklearn import tree, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def data_analysis():
    dataframe = data_import()
    results_up_down = count_up_down(dataframe)
    results_only_up = count_only_up(dataframe)
    results_dominant_event = count_dominant_event(dataframe)


    for column in training_set.columns.values:
        if not isinstance(column, tuple): continue

    yn_save = raw_input("Wanna save the tables?\n")

    for key, df in results_up_down.iteritems():
        if (yn_save == "y"):
            rename_columns(rename_indexes(df)).to_csv("{}-day-movements.csv".format(key), encoding='utf-8')

        if not isinstance(key, (int, long)): continue
        yn = raw_input("Want to proceed with the k = " + str(key) + " U/D plot? [y/n]\t")
        if (yn == "y"): 
            plot(df, key, 'U/D movements in a ' + str(key) + '-day sequence')

    for key, df in results_only_up.iteritems():
        if (yn_save == "y"):
            rename_columns(df).to_csv("{}-day-ups.csv".format(key), encoding='utf-8')
            
        yn = raw_input("Want to proceed with the k = " + str(key) + " U plot? [y/n]\t")
        if (yn == "y"): 
            plot(df, key, 'Count of U movements in a ' + str(key) + '-day sequence')

def experiment():
    dataframe = data_import('sp.csv')

    a = {}
    a['LogisticRegression'] = LogisticRegression(verbose=False)
    a['LogisticRegression - 1K'] = LogisticRegression(C=1000.0, verbose=False)
    a['DecisionTree'] = tree.DecisionTreeClassifier()
    a['NN (5, 2) - 1e-3'] = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, verbose=False)
    a['NN (5, 2) - 1.0'] = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, verbose=False)
    a['NN (25, 2) - 1e-3'] = MLPClassifier(alpha=1e-1, hidden_layer_sizes=(25, 2), verbose=False)
    a['NN (25, 2) - 1.0'] = MLPClassifier(alpha=1.0, hidden_layer_sizes=(25, 2), verbose=False)

    d = {}
    for k in range(1, 8):
        d['all_t_{}'.format(k)] = ml.feature_extraction(dataframe, past_days = k)
        d['no_ups_{}'.format(k)] = ml.feature_extraction(dataframe, use_ups = False, past_days = k)
        d['no_updowns_{}'.format(k)] = ml.feature_extraction(dataframe, use_updowns = False, past_days = k)
        d['no_value_{}'.format(k)] = ml.feature_extraction(dataframe, use_value = False, past_days = k)
        d['only_updowns_{}'.format(k)] = ml.feature_extraction(dataframe, use_ups = False, use_value = False, past_days = k)
        d['only_ups_{}'.format(k)] = ml.feature_extraction(dataframe, use_value = False, use_updowns = False, past_days = k)
        d['only_value'] = ml.feature_extraction(dataframe, use_ups = False, use_updowns = False)

    results = ml.experiment_dataset(MLPClassifier(alpha=0.1, hidden_layer_sizes=(25, 2), verbose=False), d)
    results = results.sort(['accuracy'])

    
    results = ml.experiment_dataset(MLPClassifier(alpha=0.1, hidden_layer_sizes=(25, 2), verbose=False), d)
    results = results.sort(['accuracy'])

    X, y = ml.feature_extraction(dataframe, past_days = 3, use_value = False)
    algorithms = {'Final': MLPClassifier(alpha=0.1, hidden_layer_sizes=(25, 2), verbose=False)}
    results = ml.experiment_algorithms(X, y, algorithms)
    print results

def start():
    if (raw_input("Data analysis [1] or Machine Learning [2]?\t") == "1"):
        data_analysis()
    else:
        experiment()
    

if __name__ == "__main__":
    #setup()
    start()