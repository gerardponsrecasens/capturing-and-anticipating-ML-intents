from tpot import TPOTClassifier
import numpy as np
import pickle
import pandas as pd
import os

'''
Creation of experiments with TPOT: the datasets are assumed to have the tarfet on the last column.
'''


# User inputs
metric_name = 'Accuracy'  #Change 'metric_value' accordingly
user = 'User55'
input_path = r'./datasets/'
output_path = r'./results/'
meta_features = pd.read_csv(r'simple-meta-features.csv') #Used only to get the name of dataset in line 34. Not needed if one wants to store the name as idx

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

    pipeline_optimizer = TPOTClassifier(generations=3, population_size=20, cv=5,
                                        random_state=42, verbosity=2,max_time_mins=5)
    pipeline_optimizer.fit(X_train, y_train)


    results = {'metric_name':metric_name,'metric_value':pipeline_optimizer.score(X_test, y_test),'steps':pipeline_optimizer.fitted_pipeline_.steps,'dataset':ds,'user':user}


    with open(output_path+'tpot_'+ds+'.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)