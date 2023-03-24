from hpsklearn import HyperoptEstimator, any_classifier, any_preprocessing
from sklearn.datasets import load_iris,load_digits,load_wine,load_breast_cancer
from hyperopt import tpe
import numpy as np
import pickle


'''
Toy examples to be used when the user does not have available datasets.
'''


output_path = './store/classification/'
user = 'User10'
metric = 'Accuracy'


datasets = [load_iris,load_digits,load_wine,load_breast_cancer]

for dataset in datasets:
    data = dataset()
    ds = str(dataset).split(' ')[1][5:]
    X = data.data
    y = data.target
    test_size = int(0.3 * len(y))
    np.random.seed(13)
    indices = np.random.permutation(len(X))
    X_train = X[indices[:-test_size]]
    y_train = y[indices[:-test_size]]
    X_test = X[indices[-test_size:]]
    y_test = y[indices[-test_size:]]


    if __name__ == "__main__": 
        estim = HyperoptEstimator(classifier=any_classifier("my_clf"),
                                    preprocessing=any_preprocessing("my_pre"),
                                    algo=tpe.suggest,
                                    max_evals=20,
                                    trial_timeout=300, verbose= False)

        # Search the hyperparameter space based on the data
        estim.fit(X_train, y_train)

        results = {'metric_name':metric,'metric_score':estim.score(X_test, y_test),'pipeline':estim.best_model(),'dataset':ds,'user':user}


        with open(output_path+ds+'.pickle', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)