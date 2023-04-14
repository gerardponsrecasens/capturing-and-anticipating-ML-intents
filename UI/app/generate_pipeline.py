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
import multiprocessing


'''
Script used to automatically run HyperOpt constraint experiments for all the datasets
stored in the folder /datasets. They are assumed to have the last columnas the target, 
and that they contain only numeric attributes. The constraint are related to the
classification algorithms, their hyperparameters and the preprocessing
'''

 #Used only to get the name of dataset in line 34. Not needed if one wants to store the name as idx

# Custom loss functions

'''
HyperOpt minimizes the objective funciton, hence if working with metrics for which higher values
indicate better results, they should be negated or inverted.
'''

def f1_loss(target, pred):
    return -f1_score(target, pred,average='weighted')
def accuracy_loss(target,pred):
    return -accuracy_score(target,pred)
def auc_loss(target, pred):
    return -roc_auc_score(target, pred,average='weighted')
def precision_loss(target, pred):
    return -precision_score(target, pred,average='weighted')

scorer = {'Accuracy':accuracy_loss,'F1':f1_loss,'AUC':auc_loss,'Precision':precision_loss}


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

def generate(user_input):

    # User inputs
    metric_name = user_input['Metric']
    user = user_input['User']
    max_evals = 5
    max_time = user_input['Time']
    input_path = r'./app/static/datasets/iris.csv'


    # Define Constraint Space (Classifiers (w/ hyperparameters) and Preprocessors)

    different_classifiers = {'SVC':{'C':[0.01,0.1,1,10],'kernel':['linear','poly','rbf']},
                            'RandomForestClassifier':{'n_estimators':[100,150,200],'max_depth':[2,3,4,5,6,7],'criterion':['gini','entropy']},
                            'LogisticRegression':{'penalty':['l1','l2'],'C':[0.01,0.1,1,10]},
                            'KNeighborsClassifier':{'n_neighbors':[3,5,7],'weights':['uniform','distance']}}

    different_preprocessors = {'NoPre':[],'StandardScaler':[standard_scaler('my_pre')],'MinMaxScaler':[min_max_scaler('my_pre')],'Normalizer':[normalizer('my_pre')]} 


   
    dataframe = pd.read_csv(input_path)
    dataframe = dataframe.dropna()
    ds = 'iris'
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

    algorithm = user_input['Algorithm']
    constraint_args = {}
    algorithm_args = different_classifiers[algorithm]

    for arg in algorithm_args:
        if bool(random.getrandbits(1)):
            l = len(algorithm_args[arg])
            constraint_args[arg]= algorithm_args[arg][random.randint(0, l-1)]

    
    if user_input['Preprocessing']:

        preprocessor = random.choice(list(different_preprocessors.keys()))

    else:
        preprocessor = 'NoPre'



    estim = HyperoptEstimator(classifier=define_classifier(algorithm,constraint_args),
                                preprocessing=different_preprocessors[preprocessor],
                                algo=tpe.suggest,
                                loss_fn=scorer[metric_name],
                                max_evals=max_evals,
                                trial_timeout=int(max_time/max_evals), verbose= False)

    # Search the hyperparameter space based on the data
    
    start = time.time()
    estim.fit(X_train, y_train)
    end = time.time()

    y_pred = estim.predict(X_test)

    results = {'metric_name':metric_name,'metric_value':get_score(metric_name,y_pred,y_test),
            'pipeline':estim.best_model(),'dataset':ds,'user':user,
            'algorithm_constraint':algorithm,'hyperparam_constraints':constraint_args if len(constraint_args)!=0 else None,
            'preprocessor_constraint':preprocessor,'max_time':max_time,'time':end-start}


    # with open(output_path+ds+'.pickle', 'wb') as handle:
    #     pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return(np.round(estim.score(X_test, y_test),2))











