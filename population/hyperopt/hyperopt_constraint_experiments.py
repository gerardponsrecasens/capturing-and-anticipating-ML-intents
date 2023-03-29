from hpsklearn import HyperoptEstimator, any_classifier, any_preprocessing, svc,random_forest_classifier,k_neighbors_classifier,logistic_regression
from hpsklearn import standard_scaler, min_max_scaler, normalizer
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score
from hyperopt import tpe
import numpy as np
import pickle
import os
import pandas as pd
import random
import time


'''
Script used to automatically run HyperOpt constraint experiments for all the datasets
stored in the folder /datasets. They are assumed to have the last columnas the target, 
and that they contain only numeric attributes. The constraint are related to the
classification algorithms, their hyperparameters and the preprocessing
'''

# User inputs

metric_name = 'F1'  #Change 'metric_value' accordingly
user = 'User10'
input_path = r'./datasets/'
output_path = r'./store/'
meta_features = pd.read_csv(r'simple-meta-features.csv') #Used only to get the name of dataset in line 34. Not needed if one wants to store the name as idx

# Custom loss functions

'''
HyperOpt minimizes the objective funciton, hence if working with metrics for which higher values
indicate better results, they should be negated or inverted.
'''

def f1_loss(target, pred):
    return -f1_score(target, pred,average='weighted')


# Helper functions

def define_classifier(algorithm,args={}):

    '''
    Defines an algorithm and hyperparameter restriction in the HyperOpt syntax.
    '''
    if algorithm == 'SVC':
        return svc('my_svc',**args)
    elif algorithm == 'RandomForestClassifier':
        return random_forest_classifier('my_rf',**args)
    elif algorithm == 'LogisticRegression':
        return logistic_regression('my_lr',**args)
    elif algorithm == 'KNeighborsClassifier':
        return k_neighbors_classifier('my_knc',**args)
    else:
        return any_classifier("my_clf")


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

    # Define Constraint Space (Classifiers (w/ hyperparameters) and Preprocessors)

    different_classifiers = {'SVC':{'C':[0.01,0.1,1,10],'kernel':['linear','poly','rbf']},
                            'RandomForestClassifier':{'n_estimators':[100,150,200],'max_depth':[2,3,4,5,6,7],'criterion':['gini','entropy']},
                            'LogisticRegression':{'penalty':['l1','l2'],'C':[0.01,0.1,1,10]},
                            'KNeighborsClassifier':{'n_neighbors':[3,5,7],'weights':['uniform','distance']}}

    different_preprocessors = {'NoPre':[],'StandardScaler':[standard_scaler('my_pre')],'MinMaxScaler':[min_max_scaler('my_pre')],'Normalizer':[normalizer('my_pre')]} 


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

        # Select Constraints Randomly

        algorithm = random.choice(list(different_classifiers.keys()))
        constraint_args = {}
        algorithm_args = different_classifiers[algorithm]

        for arg in algorithm_args:
            if bool(random.getrandbits(1)):
                l = len(algorithm_args[arg])
                constraint_args[arg]= algorithm_args[arg][random.randint(0, l-1)]

        preprocessor = random.choice(list(different_preprocessors.keys()))



        estim = HyperoptEstimator(classifier=define_classifier(algorithm,constraint_args),
                                    preprocessing=different_preprocessors[preprocessor],
                                    algo=tpe.suggest,
                                    loss_fn=f1_loss,
                                    max_evals=15,
                                    trial_timeout=150, verbose= False)

        # Search the hyperparameter space based on the data
        
        start = time.time()
        estim.fit(X_train, y_train)
        end = time.time()

        y_pred = estim.predict(X_test)

        results = {'metric_name':metric_name,'metric_value':get_score(metric_name,y_pred,y_test),
                   'pipeline':estim.best_model(),'dataset':ds,'user':user,
                   'algorithm_constraint':algorithm,'hyperparam_constraints':constraint_args if len(constraint_args)!=0 else None,
                   'preprocessor_constraint':preprocessor,'time':end-start}


        with open(output_path+ds+'.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(estim.score(X_test, y_test))
        print(estim.best_model())











