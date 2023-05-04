from uci_datasets import Dataset, all_datasets
import pandas as pd
import numpy as np
from hpsklearn import HyperoptEstimator, any_regressor, any_preprocessing,svr, random_forest_regressor, k_neighbors_regressor,sgd_regressor
from hpsklearn import standard_scaler, min_max_scaler, normalizer
from sklearn.metrics import mean_absolute_error
from hyperopt import tpe
import random
import pickle
import time
import uuid


def define_regressor(algorithm,args={}):

    '''
    Defines an algorithm and hyperparameter restriction in the HyperOpt syntax.
    '''
    if algorithm == 'SVR':
        return svr('my_svr',**args)
    elif algorithm == 'RandomForestRegressor':
        return random_forest_regressor('my_rf',**args)
    elif algorithm == 'SGDRegressor':
        return sgd_regressor('my_lr',**args)
    elif algorithm == 'KNeighborsRegressor':
        return k_neighbors_regressor('my_knr',**args)
    else:
        return any_regressor("my_reg")
    
different_regressors = {'SVR':{'C':[0.01,0.1,1,10],'kernel':['linear','poly','rbf']},
                        'RandomForestRegressor':{'n_estimators':[100,150,200],'max_depth':[2,3,4,5,6,7],'criterion':['squared_error','friedman_mse']},
                        'SGDRegressor':{'penalty':['l1','l2'],'learning_rate':['invscaling','optimal']},
                        'KNeighborsRegressor':{'n_neighbors':[3,5,7],'weights':['uniform','distance']}}

different_preprocessors = {'NoPre':[],'StandardScaler':[standard_scaler('my_pre')],'MinMaxScaler':[min_max_scaler('my_pre')],'Normalizer':[normalizer('my_pre')]}

datasets = [name for name, (n_observations, n_dimensions) in all_datasets.items() if n_observations < 50000]
random.shuffle(datasets)


if __name__ == "__main__": 

    for dataset in datasets:

        user = random.choice(['User10','User11','User12','User13'])
        metric_name = 'R2'
        max_time = 2250

        X = Dataset(dataset).x
        y = Dataset(dataset).y.ravel()

        test_size = int(0.3 * len(y))
        np.random.seed(13)
        indices = np.random.permutation(len(X))
        X_train = X[indices[:-test_size]]
        y_train = y[indices[:-test_size]]
        X_test = X[indices[-test_size:]]
        y_test = y[indices[-test_size:]]

        # Select Constraints Randomly

        algorithm = random.choice(list(different_regressors.keys()))
        constraint_args = {}
        algorithm_args = different_regressors[algorithm]

        for arg in algorithm_args:
            if bool(random.getrandbits(1)):
                l = len(algorithm_args[arg])
                constraint_args[arg]= algorithm_args[arg][random.randint(0, l-1)]

        preprocessor = random.choice(list(different_preprocessors.keys()))



        estim = HyperoptEstimator(classifier=define_regressor(algorithm,constraint_args),
                                    preprocessing=different_preprocessors[preprocessor],
                                    algo=tpe.suggest,
                                    loss_fn=mean_absolute_error,
                                    max_evals=15,
                                    trial_timeout=150, verbose= False)

        # Search the hyperparameter space based on the data
        
        start = time.time()
        estim.fit(X_train, y_train)
        end = time.time()

        y_pred = estim.predict(X_test)


        results = {'metric_name':metric_name,'metric_value':estim.score(X_test, y_test),
                   'pipeline':estim.best_model(),'dataset':dataset,'user':user,
                   'algorithm_constraint':algorithm,'hyperparam_constraints':constraint_args if len(constraint_args)!=0 else None,
                   'preprocessor_constraint':preprocessor,'max_time':max_time,'time':end-start}


        with open('./store/'+str(uuid.uuid4())+'.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


        print(estim.score(X_test, y_test))
        print(estim.best_model())