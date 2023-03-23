import rdflib

from rdflib import Graph, URIRef, XSD, Literal
from rdflib.namespace import  RDF,RDFS

'''
Script used to generate all the instances for the algorithm implementations and hyperparameters 
for the algorithms used in HyperOpt.
'''

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



## GENERAL CLASSIFICATION ALGORITHMS

from sklearn.svm import SVC
svmc = URIRef(dmkb+'SVMc')
g.add((svmc,RDF.type,URIRef(dmop+'SVC-Algorithm')))
svmc_imp = URIRef(uri+'sklearn-SVC')
g.add((svmc_imp,ns.implements,svmc))
clf = SVC()
params = clf.get_params()
for param in params:
    param_instance = URIRef(uri+'sklearn-SVC-'+param)
    g.add((svmc_imp,ns.hasHyperparameter,param_instance))


from sklearn.neighbors import KNeighborsClassifier
knnc = URIRef(dmkb+'KNNc')
g.add((knnc,RDF.type,ns_dmop.KNearestNeighborAlgorithm))
knnc_imp = URIRef(uri+'sklearn-KNeighborsClassifier')
g.add((knnc_imp,ns.implements,knnc))
clf = KNeighborsClassifier()
params = clf.get_params()
for param in params:
    param_instance = URIRef(uri+'sklearn-KNeighborsClassifier-'+param)
    g.add((knnc_imp,ns.hasHyperparameter,param_instance))



from sklearn.linear_model import SGDClassifier
sgdc = URIRef(dmkb+'SGDc')
g.add((sgdc,RDF.type,ns_dmop.ClassificationModelingAlgorithm))
sgdc_imp = URIRef(uri+'sklearn-SGDClassifier')
g.add((sgdc_imp,ns.implements,sgdc))
clf = SGDClassifier()
params = clf.get_params()
for param in params:
    param_instance = URIRef(uri+'sklearn-SGDClassifier-'+param)
    g.add((sgdc_imp,ns.hasHyperparameter,param_instance))


from sklearn.ensemble import GradientBoostingClassifier
gbc = URIRef(dmkb+'GradientBoostingC')
g.add((gbc,RDF.type,ns_dmop.ClassificationModelingAlgorithm))
gbc_imp = URIRef(uri+'sklearn-GradientBoostingClassifier')
g.add((gbc_imp,ns.implements,gbc))
clf = GradientBoostingClassifier()
params = clf.get_params()
for param in params:
    param_instance = URIRef(uri+'sklearn-GradientBoostingClassifier-'+param)
    g.add((gbc_imp,ns.hasHyperparameter,param_instance))


from sklearn.ensemble import RandomForestClassifier
rfc = URIRef(dmkb+'RFc')
g.add((rfc,RDF.type,ns_dmop.ClassificationModelingAlgorithm))
rfc_imp = URIRef(uri+'sklearn-RandomForestClassifier')
g.add((rfc_imp,ns.implements,rfc))
clf = RandomForestClassifier()
params = clf.get_params()
for param in params:
    param_instance = URIRef(uri+'sklearn-RandomForestClassifier-'+param)
    g.add((rfc_imp,ns.hasHyperparameter,param_instance))


from sklearn.tree import ExtraTreeClassifier
extraTree = URIRef(dmkb+'ExtraTreeC')
g.add((extraTree,RDF.type,ns_dmop.ClassificationTreeInductionAlgorithm))
extraTree_imp = URIRef(uri+'sklearn-ExtraTree')
g.add((extraTree_imp,ns.implements,extraTree))
clf = ExtraTreeClassifier()
params = clf.get_params()
for param in params:
    param_instance = URIRef(uri+'sklearn-ExtraTree-'+param)
    g.add((extraTree_imp,ns.hasHyperparameter,param_instance))


from sklearn.ensemble import AdaBoostClassifier
adaBoostC = URIRef(dmkb+'AdaBoostC')
g.add((adaBoostC,RDF.type,ns_dmop.ClassificationModelingAlgorithm))
adaBoostC_imp = URIRef(uri+'sklearn-AdaBoostClassifier')
g.add((adaBoostC_imp,ns.implements,adaBoostC))
clf = AdaBoostClassifier()
params = clf.get_params()
for param in params:
    param_instance = URIRef(uri+'sklearn-AdaBoostClassifier-'+param)
    g.add((adaBoostC_imp,ns.hasHyperparameter,param_instance))


## GENERAL REGRESSION ALGORITHMS

from sklearn.svm import SVR
algo = URIRef(dmkb+'SVMr')
g.add((algo,RDF.type,ns_dmop.RegressionModelingAlgorithm))
algo_imp = URIRef(uri+'sklearn-SVR')
g.add((algo_imp,ns.implements,algo))
clf = SVR()
params = clf.get_params()
for param in params:
    param_instance = URIRef(uri+'sklearn-SVR-'+param)
    g.add((algo_imp,ns.hasHyperparameter,param_instance))


from sklearn.neighbors import KNeighborsRegressor
algo = URIRef(dmkb+'KNNr')
g.add((algo,RDF.type,ns_dmop.RegressionModelingAlgorithm))
algo_imp = URIRef(uri+'sklearn-KNeighborsRegressor')
g.add((algo_imp,ns.implements,algo))
clf = KNeighborsRegressor()
params = clf.get_params()
for param in params:
    param_instance = URIRef(uri+'sklearn-KNeighborsRegressor-'+param)
    g.add((algo_imp,ns.hasHyperparameter,param_instance))


from sklearn.ensemble import RandomForestRegressor
algo = URIRef(dmkb+'RFr')
g.add((algo,RDF.type,ns_dmop.RegressionModelingAlgorithm))
algo_imp = URIRef(uri+'sklearn-RandomForestRegressor')
g.add((algo_imp,ns.implements,algo))
clf = RandomForestRegressor()
params = clf.get_params()
for param in params:
    param_instance = URIRef(uri+'sklearn-RandomForestRegressor-'+param)
    g.add((algo_imp,ns.hasHyperparameter,param_instance))

from sklearn.ensemble import AdaBoostRegressor
algo = URIRef(dmkb+'AdaBoostR')
g.add((algo,RDF.type,ns_dmop.RegressionModelingAlgorithm))
algo_imp = URIRef(uri+'sklearn-AdaBoostRegressor')
g.add((algo_imp,ns.implements,algo))
clf = AdaBoostRegressor()
params = clf.get_params()
for param in params:
    param_instance = URIRef(uri+'sklearn-AdaBoostRegressor-'+param)
    g.add((algo_imp,ns.hasHyperparameter,param_instance))

from sklearn.tree import ExtraTreeRegressor
algo = URIRef(dmkb+'ExtraTreeRegressor')
g.add((algo,RDF.type,ns_dmop.RegressionModelingAlgorithm))
algo_imp = URIRef(uri+'sklearn-ExtraTreeRegressor')
g.add((algo_imp,ns.implements,algo))
clf = ExtraTreeRegressor()
params = clf.get_params()
for param in params:
    param_instance = URIRef(uri+'sklearn-ExtraTreeRegressor-'+param)
    g.add((algo_imp,ns.hasHyperparameter,param_instance))


from sklearn.ensemble import GradientBoostingRegressor
algo = URIRef(dmkb+'GradientBoostingR')
g.add((algo,RDF.type,ns_dmop.RegressionModelingAlgorithm))
algo_imp = URIRef(uri+'sklearn-GradientBoostingRegressor')
g.add((algo_imp,ns.implements,algo))
clf = GradientBoostingRegressor()
params = clf.get_params()
for param in params:
    param_instance = URIRef(uri+'sklearn-GradientBoostingRegressor-'+param)
    g.add((algo_imp,ns.hasHyperparameter,param_instance))


from sklearn.linear_model import SGDRegressor
algo = URIRef(dmkb+'SGDr')
g.add((algo,RDF.type,ns_dmop.RegressionModelingAlgorithm))
algo_imp = URIRef(uri+'sklearn-SGDRegressor')
g.add((algo_imp,ns.implements,algo))
clf = SGDRegressor()
params = clf.get_params()
for param in params:
    param_instance = URIRef(uri+'sklearn-SGDRegressor-'+param)
    g.add((algo_imp,ns.hasHyperparameter,param_instance))

## GENERAL PREPROCESSING ALGORITHMS

from sklearn.decomposition import PCA
PCA_imp = URIRef(uri+'sklearn-PCA')
g.add((PCA_imp,ns.implements,ns_dmkb.PrincipalComponentAnalysis))
clf = PCA()
params = clf.get_params()
for param in params:
    param_instance = URIRef(uri+'sklearn-PCA-'+param)
    g.add((PCA_imp,ns.hasHyperparameter,param_instance))


from sklearn.preprocessing import Normalizer as NF
Normalizer = URIRef(dmkb+'Normalizer')
g.add((Normalizer,RDF.type,ns_dmop.FeatureNormalizationAlgorithm))
Normalizer_imp = URIRef(uri+'sklearn-Normalizer')
g.add((Normalizer_imp,ns.implements,Normalizer))
clf = NF()
params = clf.get_params()
for param in params:
    param_instance = URIRef(uri+'sklearn-Normalizer-'+param)
    g.add((Normalizer_imp,ns.hasHyperparameter,param_instance))


from sklearn.preprocessing import MinMaxScaler as MMS
MinMaxScaler = URIRef(dmkb+'MinMaxScaler')
g.add((MinMaxScaler,RDF.type,ns_dmop.FeatureStandardizationAlgorithm))
MinMaxScaler_imp = URIRef(uri+'sklearn-MinMaxScaler')
g.add((MinMaxScaler_imp,ns.implements,MinMaxScaler))
clf = MMS()
params = clf.get_params()
for param in params:
    param_instance = URIRef(uri+'sklearn-MinMaxScaler-'+param)
    g.add((MinMaxScaler_imp,ns.hasHyperparameter,param_instance))


from sklearn.preprocessing import StandardScaler as SS
StandardScaler = URIRef(dmkb+'StandardScaler')
g.add((StandardScaler,RDF.type,ns_dmop.FeatureStandardizationAlgorithm))
StandardScaler_imp = URIRef(uri+'sklearn-StandardScaler')
g.add((StandardScaler_imp,ns.implements,StandardScaler))
clf = SS()
params = clf.get_params()
for param in params:
    param_instance = URIRef(uri+'sklearn-StandardScaler-'+param)
    g.add((StandardScaler_imp,ns.hasHyperparameter,param_instance))


g.serialize(format="nt",destination="rdf_hyperopt.nt")