from hpsklearn import HyperoptEstimator, any_classifier, any_preprocessing
from sklearn.datasets import load_iris,load_digits,load_wine,load_breast_cancer
from hyperopt import tpe
import numpy as np
import pickle


'''
Toy examples to be used when the user does not have available datasets.
'''

## DIGITS

# Download the data and split into training and test sets

iris = load_digits()
ds = 'digits'

X = iris.data
y = iris.target

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

    results = {'accuracy':estim.score(X_test, y_test),'pipeline':estim.best_model(),'dataset':ds}


    with open('./store/classification/'+ds+'.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


    print(estim.score(X_test, y_test))
    print(estim.best_model())


## IRIS

iris = load_iris()
ds = 'iris'

X = iris.data
y = iris.target

test_size = int(0.3 * len(y))
np.random.seed(13)
indices = np.random.permutation(len(X))
X_train = X[indices[:-test_size]]
y_train = y[indices[:-test_size]]
X_test = X[indices[-test_size:]]
y_test = y[indices[-test_size:]]




# Instantiate a HyperoptEstimator with the search space and number of evaluations
if __name__ == "__main__":
    estim = HyperoptEstimator(classifier=any_classifier("my_clf"),
                                preprocessing=any_preprocessing("my_pre"),
                                algo=tpe.suggest,
                                max_evals=20,
                                trial_timeout=300, verbose= False)

    # Search the hyperparameter space based on the data
    estim.fit(X_train, y_train)

    results = {'accuracy':estim.score(X_test, y_test),'pipeline':estim.best_model(),'dataset':ds}


    with open('./store/classification/'+ds+'.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


    print(estim.score(X_test, y_test))
    print(estim.best_model())


## BREAST CANCER

iris = load_breast_cancer()
ds = 'breast-cancer'

X = iris.data
y = iris.target

test_size = int(0.3 * len(y))
np.random.seed(13)
indices = np.random.permutation(len(X))
X_train = X[indices[:-test_size]]
y_train = y[indices[:-test_size]]
X_test = X[indices[-test_size:]]
y_test = y[indices[-test_size:]]



# Instantiate a HyperoptEstimator with the search space and number of evaluations
if __name__ == "__main__":
    estim = HyperoptEstimator(classifier=any_classifier("my_clf"),
                                preprocessing=any_preprocessing("my_pre"),
                                algo=tpe.suggest,
                                max_evals=20,
                                trial_timeout=300, verbose= False)

    # Search the hyperparameter space based on the data
    estim.fit(X_train, y_train)

    results = {'accuracy':estim.score(X_test, y_test),'pipeline':estim.best_model(),'dataset':ds}


    with open('./store/classification/'+ds+'.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


    print(estim.score(X_test, y_test))
    print(estim.best_model())


## 

iris = load_wine()
ds = 'wine'

X = iris.data
y = iris.target

test_size = int(0.3 * len(y))
np.random.seed(13)
indices = np.random.permutation(len(X))
X_train = X[indices[:-test_size]]
y_train = y[indices[:-test_size]]
X_test = X[indices[-test_size:]]
y_test = y[indices[-test_size:]]



# Instantiate a HyperoptEstimator with the search space and number of evaluations
if __name__ == "__main__":
    estim = HyperoptEstimator(classifier=any_classifier("my_clf"),
                                preprocessing=any_preprocessing("my_pre"),
                                algo=tpe.suggest,
                                max_evals=20,
                                trial_timeout=300, verbose= False)

    # Search the hyperparameter space based on the data
    estim.fit(X_train, y_train)

    results = {'accuracy':estim.score(X_test, y_test),'pipeline':estim.best_model(),'dataset':ds}


    with open('./store/classification/'+ds+'.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


    print(estim.score(X_test, y_test))
    print(estim.best_model())





