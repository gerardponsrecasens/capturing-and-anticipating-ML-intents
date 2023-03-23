from hpsklearn import HyperoptEstimator, any_classifier, any_preprocessing
from hyperopt import tpe
import numpy as np
import pickle
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

'''
Script used to automatically run HyperOpt experiments for all the datasets
stored in the folder /datasets. They are assumed to have the last column
as the target, and that they contain only numeric attributes.
'''
meta_features = pd.read_csv(r'simple-meta-features.csv')

from sklearn.metrics import f1_score

def f1_loss(target, pred):
    return -f1_score(target, pred,average='weighted')

metric_name = 'Accuracy'
user = 'User10'


if __name__ == "__main__":
    datasets = os.listdir(r'./datasets/')
    for current_link in datasets:
        idx = current_link.split('.')[0]
        dataframe = pd.read_csv(r'./datasets/'+current_link)
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
        estim.fit(X_train, y_train)

        y_pred = estim.predict(X_test)

        results = {'metric_name':metric_name,'metric_value':accuracy_score(y_pred,y_test),'pipeline':estim.best_model(),'dataset':ds,'user':user}


        with open('./store/'+ds+'.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(estim.score(X_test, y_test))
        print(estim.best_model())











