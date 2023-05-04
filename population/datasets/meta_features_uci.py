from uci_datasets import Dataset, all_datasets
import pandas as pd
import rdflib
from rdflib import Graph, URIRef, XSD, Literal
from rdflib.namespace import  RDF,RDFS
import math


g = Graph()

# NameSpace
uri = "http://localhost/8080/intentOntology#"
ns = rdflib.Namespace(uri)
dmop = "http://www.e-lico.eu/ontologies/dmo/DMOP/DMOP.owl#"
ns_dmop = rdflib.Namespace(dmop)
dolce = "http://www.loa-cnr.it/ontologies/DOLCE-Lite.owl#"
ns_dolce = rdflib.Namespace(dolce)


datasets = [name for name, (n_observations, n_dimensions) in all_datasets.items() if n_observations < 50000]


for name in datasets:

    data = pd.DataFrame(Dataset(name,print_stats=False).x)

    num_instances = len(data)
    num_features = len(data.columns)

    num_missing = sum(data.isna().sum())
    missing_values = data.isnull().sum().sum()
    instances_with_missings = data.isnull().any(axis=1).sum()
    percentageOfInstancesWithMissingValues = instances_with_missings / num_instances


    qualities = {'NumberOfFeatures':num_features,'NumberOfInstances':num_instances,
             'NumberOfInstancesWithMissingValues':instances_with_missings,'NumberOfMissingValues':missing_values,
             'PercentageOfInstancesWithMissingValues':percentageOfInstancesWithMissingValues}




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

g.serialize(format="nt",destination="meta_features_uci"+".nt")
