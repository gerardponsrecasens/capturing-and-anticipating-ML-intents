import rdflib

from rdflib import Graph, URIRef, XSD, Literal
from rdflib.namespace import  RDF,RDFS
from sklearn.utils import all_estimators
from sklearn import preprocessing as pr

'''
Script that generates the RDF triples for all the scikit-learn library classification, regression, clustering and preprocessing algorithms.
Concretely, it links each algorithm implementation to the general algorithm in DMOP (or it creates it if it does not exist) and it creates 
the instances for each of the hyperparameters.
'''

def get_all_algorithms_sklearn(algorithm_type):

    estimators = all_estimators(type_filter=algorithm_type)

    all_regs = []
    for name, RegClass in estimators:
        try:
            reg = RegClass()
            all_regs.append(reg)
        except Exception as e:
            pass
    return all_regs

g = Graph()


## NAME SPACES

uri = "http://localhost/8080/intentOntology#"
ns = rdflib.Namespace(uri)
dmop = "http://www.e-lico.eu/ontologies/dmo/DMOP/DMOP.owl#"
ns_dmop = rdflib.Namespace(dmop)
dolce = "http://www.loa-cnr.it/ontologies/DOLCE-Lite.owl#"
ns_dolce = rdflib.Namespace(dolce)
dmkb = "http://www.e-lico.eu/ontologies/dmo/DMOP/DMKB.owl#"
ns_dmkb = rdflib.Namespace(dmkb)


## CLASSIFIERS

all_classifiers = get_all_algorithms_sklearn(algorithm_type='classifier')

for algorithm in all_classifiers:
    name = str(algorithm).split('(')[0]
    onto_algorithm = URIRef(dmkb+name)

    if name not in ['LogisticRegression','LinearDiscriminantAnalysis']:
        g.add((onto_algorithm,RDF.type,ns_dmop.ClassificationModelingAlgorithm))
    onto_imp = URIRef(uri+'sklearn-'+name)
    g.add((onto_imp,ns.implements,onto_algorithm))
    params = algorithm.get_params()
    for param in params:
        param_instance = URIRef(uri+'sklearn-'+name+'-'+param)
        g.add((onto_imp,ns.hasHyperparameter,param_instance))

    # Additonal SubClasses
    if 'NB' in name:
        g.add((onto_algorithm,RDF.type,ns_dmop.BayesianAlgorithm))
    elif 'SVC' in name:
        g.add((onto_algorithm,RDF.type,URIRef(dmop+'SVC-Algorithm')))
    elif 'Neighbors' in name:
        g.add((onto_algorithm,RDF.type,ns_dmop.KNearestNeighborAlgorithm))
    elif 'Tree' in name:
        g.add((onto_algorithm,RDF.type,ns_dmop.ClassificationTreeInductionAlgorithm))




## REGRESSORS

all_regressors = get_all_algorithms_sklearn(algorithm_type='regressor')

for algorithm in all_regressors:
    name = str(algorithm).split('(')[0]
    onto_algorithm = URIRef(dmkb+name)


    g.add((onto_algorithm,RDF.type,ns_dmop.RegressionModelingAlgorithm))
    onto_imp = URIRef(uri+'sklearn-'+name)
    g.add((onto_imp,ns.implements,onto_algorithm))
    params = algorithm.get_params()
    for param in params:
        param_instance = URIRef(uri+'sklearn-'+name+'-'+param)
        g.add((onto_imp,ns.hasHyperparameter,param_instance))

    # Additonal SubClasses
    if 'Tree' in name:
        g.add((onto_algorithm,RDF.type,ns_dmop.RegressionTreeInductionAlgorithm))

    

## CLUSTERS

all_clusters = get_all_algorithms_sklearn(algorithm_type='cluster')

for algorithm in all_classifiers:
    name = str(algorithm).split('(')[0]
    onto_algorithm = URIRef(dmkb+name)

    if name not in ['KMeans']:
        g.add((onto_algorithm,RDF.type,ns_dmop.ClusteringModelingAlgorithm))
        if 'KMeans' in name:
            g.add((onto_algorithm,RDF.type,URIRef(dmop+'K-MeansAlgorithm')))


    onto_imp = URIRef(uri+'sklearn-'+name)
    g.add((onto_imp,ns.implements,onto_algorithm))
    params = algorithm.get_params()
    for param in params:
        param_instance = URIRef(uri+'sklearn-'+name+'-'+param)
        g.add((onto_imp,ns.hasHyperparameter,param_instance))

## PREPROCESSORS

preprocessers = [pr.Binarizer(),pr.KernelCenterer(),pr.MinMaxScaler(),pr.MaxAbsScaler(),pr.Normalizer(),pr.RobustScaler(),
pr.StandardScaler(),pr.QuantileTransformer(),pr.PowerTransformer(),pr.OneHotEncoder(),pr.OrdinalEncoder(),
pr.LabelBinarizer(),pr.LabelEncoder(),pr.MultiLabelBinarizer(),pr.KBinsDiscretizer(),pr.PolynomialFeatures(),
pr.SplineTransformer()]

for algorithm in preprocessers:
    name = str(algorithm).split('(')[0]
    onto_algorithm = URIRef(dmkb+name)
    g.add((onto_algorithm,RDF.type,ns_dmop.DataProcessingAlgorithm))
    onto_imp = URIRef(uri+'sklearn-'+name)
    g.add((onto_imp,ns.implements,onto_algorithm))
    params = algorithm.get_params()
    for param in params:
        param_instance = URIRef(uri+'sklearn-'+name+'-'+param)
        g.add((onto_imp,ns.hasHyperparameter,param_instance))

    # Additonal SubClasses
    if 'Scaler' in name:
        g.add((onto_algorithm,RDF.type,ns_dmop.FeatureStandardizationAlgorithm))
    elif 'Normalizer' in name:
        g.add((onto_algorithm,RDF.type,ns_dmop.FeatureNormalizationAlgorithm))
    elif 'Transformer' in name:
        g.add((onto_algorithm,RDF.type,ns_dmop.FeatureTransformationAlgorithm))
    elif 'Discretizer' in name:
        g.add((onto_algorithm,RDF.type,ns_dmop.FeatureDiscretizationAlgorithm))


## FEATURE SELECTION

from sklearn import feature_selection as fs

feature_selectors = [fs.GenericUnivariateSelect(),
    fs.SelectFpr(),fs.SelectFwe(),fs.SelectKBest(),fs.SelectPercentile(),fs.VarianceThreshold()]

for algorithm in feature_selectors:
    name = str(algorithm).split('(')[0]
    onto_algorithm = URIRef(dmkb+name)
    g.add((onto_algorithm,RDF.type,ns_dmop.FeatureSelectionAlgorithm))
    onto_imp = URIRef(uri+'sklearn-'+name)
    g.add((onto_imp,ns.implements,onto_algorithm))
    params = algorithm.get_params()
    for param in params:
        param_instance = URIRef(uri+'sklearn-'+name+'-'+param)
        g.add((onto_imp,ns.hasHyperparameter,param_instance))



## EXTRA
from sklearn.decomposition import PCA, FastICA
PCA_imp = URIRef(uri+'sklearn-PCA')
g.add((PCA_imp,ns.implements,ns_dmkb.PrincipalComponentAnalysis))
clf = PCA()
params = clf.get_params()
for param in params:
    param_instance = URIRef(uri+'sklearn-PCA-'+param)
    g.add((PCA_imp,ns.hasHyperparameter,param_instance))


algorithm = FastICA()
name = str(algorithm).split('(')[0]
onto_algorithm = URIRef(dmkb+name)
g.add((onto_algorithm,RDF.type,ns_dmop.ProjectiveFeatureExtractionAlgorithm))
onto_imp = URIRef(uri+'sklearn-'+name)
g.add((onto_imp,ns.implements,onto_algorithm))
params = algorithm.get_params()
for param in params:
    param_instance = URIRef(uri+'sklearn-'+name+'-'+param)
    g.add((onto_imp,ns.hasHyperparameter,param_instance))


import xgboost as xgb

algorithm = xgb.XGBClassifier()
onto_algorithm = URIRef(dmkb+name)
g.add((onto_algorithm,RDF.type,ns_dmop.ClassificationModelingAlgorithm))
onto_imp = URIRef(uri+'sklearn-'+name)
g.add((onto_imp,ns.implements,onto_algorithm))
params = algorithm.get_params()
for param in params:
    param_instance = URIRef(uri+'sklearn-'+name+'-'+param)
    g.add((onto_imp,ns.hasHyperparameter,param_instance))



g.serialize(format="nt",destination="sklearn_helper.nt")