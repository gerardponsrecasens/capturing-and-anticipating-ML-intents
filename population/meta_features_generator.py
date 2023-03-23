import pandas as pd
import rdflib
from rdflib import Graph, URIRef, XSD, Literal
from rdflib.namespace import  RDF,RDFS
import os
import pandas as pd
import math

'''
This script creates the RDF triples for the characteristics of a dataset. It is assumed that the dataset is in .csv format
and that the last column is the target.
'''

path=r'dataset.csv'
name = path.split('.')[0]
print(name)
data = pd.read_csv(path)
classification = True
regression = False



if classification:
    data[data.columns[-1]] = data[data.columns[-1]].astype('category')

    num_instances = len(data)
    num_features = len(data.columns)-1
    num_classes = int(data.iloc[:,-1:].nunique())
    majority = data.iloc[:,-1:].value_counts().max()
    minority = data.iloc[:,-1:].value_counts().min()

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_features = sum([column in numerics for column in data.dtypes.values])
    categorical_features = num_features - numeric_features

    missing_values = data.isnull().sum().sum()
    instances_with_missings = data.isnull().any(axis=1).sum()
    percentageOfInstancesWithMissingValues = instances_with_missings / num_instances

    categorical = data.iloc[:,:-1].select_dtypes(exclude = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
    maxNominalDistinct = categorical.nunique().max()

    
elif regression: 
    
    num_instances = len(data)
    num_features = len(data.columns)-1

    num_classes = math.nan
    majority = math.nan
    minority = math.nan

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric_features = sum([column in numerics for column in data.dtypes.values])
    categorical_features = num_features - numeric_features

    missing_values = data.isnull().sum().sum()
    instances_with_missings = data.isnull().any(axis=1).sum()
    percentageOfInstancesWithMissingValues = instances_with_missings / num_instances

    categorical = data.iloc[:,:-1].select_dtypes(exclude = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
    maxNominalDistinct = categorical.nunique().max()



qualities = {'MajorityClassSize':majority,'MaxNominalAttDistinctValues':maxNominalDistinct,'MinorityClassSize':minority,
             'NumberOfClasses':num_classes,'NumberOfFeatures':num_features,'NumberOfInstances':num_instances,
             'NumberOfInstancesWithMissingValues':instances_with_missings,'NumberOfMissingValues':missing_values,
             'NumberOfNumericFeatures':numeric_features,'NumberOfSymbolicFeatures':categorical_features,
             'PercentageOfInstancesWithMissingValues':percentageOfInstancesWithMissingValues}

g = Graph()

# NameSpace
uri = "http://localhost/8080/intentOntology#"
ns = rdflib.Namespace(uri)
dmop = "http://www.e-lico.eu/ontologies/dmo/DMOP/DMOP.owl#"
ns_dmop = rdflib.Namespace(dmop)
dolce = "http://www.loa-cnr.it/ontologies/DOLCE-Lite.owl#"
ns_dolce = rdflib.Namespace(dolce)


dataset = URIRef(uri+name)
for quality in qualities:
    if not math.isnan(qualities[quality]):
        characteristic = URIRef(uri+name+quality)
        g.add((characteristic,RDF.type,URIRef(dmop+quality)))
        g.add((dataset,URIRef(dolce+'has-quality'),characteristic))

        if quality!='PercentageOfInstancesWithMissingValues': #If it is an integer
            g.add((characteristic,ns.hasValue,Literal(int(qualities[quality]), datatype=XSD.integer)))
        else:        #It is a float
            g.add((characteristic,ns.hasValue,Literal(qualities[quality], datatype=XSD.float)))

g.serialize(format="nt",destination="meta_features_"+name+".nt")

