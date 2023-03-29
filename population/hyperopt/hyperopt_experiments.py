from hpsklearn import HyperoptEstimator, any_classifier, any_preprocessing
from hyperopt import tpe
import numpy as np
import pickle
import os
import pandas as pd
import time
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score

'''
Script used to automatically run HyperOpt experiments for all the datasets
stored in the folder /datasets. They are assumed to have the last column
as the target, and that they contain only numeric attributes.
'''


# User inputs
metric_name = 'Accuracy'  # Choose between: Accuracy, F1, AUC and Precision
user = 'User10'
input_path = r'./datasets/'
output_path = r'./store/'
meta_features = pd.read_csv(r'simple-meta-features.csv') #Used only to get the name of dataset in line 34. Not needed if one wants to store the name as idx

# Custom loss functions

def f1_loss(target, pred):
    return -f1_score(target, pred,average='weighted')

# Scorer helper funciton

def get_score(metric_name,y_pred,y_test):
    if metric_name == 'Accuracy':
        return accuracy_score(y_pred,y_test)
    if metric_name == 'F1':
        return f1_score(y_pred,y_test,average='weighted')
    if metric_name == 'AUC':
        return roc_auc_score(y_pred,y_test,average='weighted')
    if metric_name == 'Precision':
        return precision_score(y_pred,y_test,average='weighted')
        

if __name__ == "__main__":
    datasets = os.listdir(input_path)  
    for current_link in datasets:
        idx = current_link.split('.')[0]
        dataframe = pd.read_csv(input_path+current_link)
        dataframe = dataframe.dropna()
        name = meta_features[meta_features['row']==int(idx)].name
        ds = name.values[0]
        print(ds,idx)
        data = dataframe.values
        X, y = data[:, :-1], data[:, -1]
        y = y.astype('int32')

        test_size = int(0.3 * len(y))
        np.random.seed(13)
        indices = np.random.permutation(len(X))
        X_train = X[indices[:-test_size]]
        y_train = y[indices[:-test_size]]
        X_test = X[indices[-test_size:]]
        y_test = y[indices[-test_size:]]

        estim = HyperoptEstimator(classifier=any_classifier("my_clf"),
                                    preprocessing=any_preprocessing("my_pre"),
                                    algo=tpe.suggest,
                                    loss_fn=f1_loss,
                                    max_evals=15,
                                    trial_timeout=150, verbose= False)

        # Search the hyperparameter space based on the data
        start = time.time()
        estim.fit(X_train, y_train)
        end = time.time()

        y_pred = estim.predict(X_test)

        results = {'metric_name':metric_name,'metric_value':get_score(metric_name,y_pred,y_test),'pipeline':estim.best_model(),
                   'dataset':ds,'user':user,'time':end-start}


        with open(output_path+ds+'.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(estim.score(X_test, y_test))
        print(estim.best_model())











