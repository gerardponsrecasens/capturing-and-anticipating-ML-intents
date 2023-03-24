from sklearn.datasets import load_iris,load_digits,load_wine,load_breast_cancer
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
import numpy as np
import pickle


'''
Toy examples to be used when the user does not have available datasets.
'''


output_path = './results/'
user = 'User10'
metric = 'Accuracy'


datasets = [load_wine,load_digits,load_iris,load_breast_cancer]

for dataset in datasets:
    data = dataset()
    ds = str(dataset).split(' ')[1][5:]

    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target,
                                                        train_size=0.75, test_size=0.25)

    pipeline_optimizer = TPOTClassifier(generations=3, population_size=20, cv=5,
                                        random_state=42, verbosity=2)
    pipeline_optimizer.fit(X_train, y_train)


    results = {'metric_name':metric,'metric_score':pipeline_optimizer.score(X_test, y_test),'steps':pipeline_optimizer.fitted_pipeline_.steps,'dataset':ds,'user':user}


    with open(output_path+'tpot_'+ds+'.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)