import rdflib
import time
import os
import pickle
from random import randrange
from rdflib import Graph, URIRef, XSD, Literal
from rdflib.namespace import  RDF,RDFS


'''
Script to generate RDF triples from the HyperOpt results.
'''

input_path = r'./store/classification/'
output_path = "classification.nt"


g = Graph()


# NameSpace
uri = "http://localhost/8080/intentOntology#"
ns = rdflib.Namespace(uri)
dmop = "http://www.e-lico.eu/ontologies/dmo/DMOP/DMOP.owl#"
ns_dmop = rdflib.Namespace(dmop)
dolce = "http://www.loa-cnr.it/ontologies/DOLCE-Lite.owl#"
ns_dolce = rdflib.Namespace(dolce)
dmkb = "http://www.e-lico.eu/ontologies/dmo/DMOP/DMKB.owl#"
ns_dmkb = rdflib.Namespace(dmkb)


files = os.listdir(input_path)


for file in files:
    with open('./store/classification/'+file, 'rb') as handle:
        data = pickle.load(handle)

    user_name = data['user']
    dataset_name = data['dataset']
    current_time = str(int(time.time()))



    # General Workflow
    user = URIRef(uri+user_name)
    workflow = URIRef(uri+'Worflow'+user_name+dataset_name+'-'+current_time)
    dataset = URIRef(uri+dataset_name)
    task = URIRef(uri+'Task'+user_name+dataset_name+'-'+current_time)

    g.add((user,ns.runs,workflow))
    g.add((workflow,ns.hasFeedback,Literal(randrange(11), datatype=XSD.integer)))
    g.add((dataset,RDF.type,ns_dmop.DataSet))
    g.add((workflow,ns.hasInput,dataset))
    g.add((workflow,ns.achieves,task))

    # Task: Requirements and Constraints

    intent = URIRef(uri+'Predict')
    g.add((task,ns.hasIntent,intent))


    evalRequirement = URIRef(uri+'EvalRequirement'+user_name+dataset_name+'-'+current_time)
    traintestsplit = URIRef(uri+'TrainTestSplit')
    metric = URIRef(uri+data['metric_name'])
    maximize = URIRef(uri+'Maximize')
    g.add((task,ns.hasRequirement,evalRequirement))
    g.add((evalRequirement,ns.withMethod,traintestsplit))
    g.add((evalRequirement,ns.onMetric,metric))
    g.add((maximize,RDF.type,URIRef(uri+'Min-Max-Equal'))) ##############################
    g.add((evalRequirement,ns.howEval,maximize))
    # g.add((evalRequirement,ns.hasValue,Literal(80, datatype=XSD.integer)))
    # g.add((evalRequirement,ns.isSatisfied,Literal('true', datatype=XSD.boolean)))

    modelEval = URIRef(uri+'ModelEval'+user_name+dataset_name+'-'+current_time)
    g.add((workflow,ns.hasOutput,modelEval))
    g.add((modelEval,ns.specifies,metric))
    g.add((modelEval,ns.hasValue,Literal(data['value'], datatype=XSD.float)))


    if data['pipeline']['preprocs'] != None:
        model = URIRef(uri+'Model'+user_name+dataset_name+'-'+current_time)
        preproc = URIRef(uri+'Prepro'+user_name+dataset_name+'-'+current_time)
        g.add((workflow,ns.hasStep,model))
        g.add((workflow,ns.hasStep,preproc))
        g.add((model,ns.order,Literal(2, datatype=XSD.integer)))
        g.add((preproc,ns.order,Literal(1, datatype=XSD.integer)))
        g.add((preproc,ns.followedBy,model))

        prepro = data['pipeline']['preprocs'][0]
        name = str(prepro).split('(')[0]
        g.add((preproc,ns.hasImplementation,URIRef(uri+'sklearn-'+name)))

        params = prepro.get_params()
        for param in params:
            hyperinput = URIRef(uri+user_name+dataset_name+name+param+'-'+current_time)
            g.add((preproc,ns.hasHyperparamInput,hyperinput))
            g.add((URIRef(uri+'sklearn-'+name+'-'+param),ns.specifiedBy,hyperinput))
            value = params[param]
            if type(value)==int:
                g.add((hyperinput,ns.hasValue,Literal(value, datatype=XSD.integer)))
            elif type(value)==float:
                g.add((hyperinput,ns.hasValue,Literal(value, datatype=XSD.float)))
            elif type(value)==str:
                g.add((hyperinput,ns.hasValue,Literal(value, datatype=XSD.string)))
            elif type(value)==bool:
                g.add((hyperinput,ns.hasValue,Literal(value, datatype=XSD.boolean)))


    else:
        model = URIRef(uri+'Model'+user_name+dataset_name+'-'+current_time)
        g.add((workflow,ns.hasStep,model))
        g.add((model,ns.order,Literal(1, datatype=XSD.integer)))

    algo = data['pipeline']['learner']
    name = str(algo).split('(')[0]
    g.add((model,ns.hasImplementation,URIRef(uri+'sklearn-'+name)))

    params = algo.get_params()
    for param in params:
        hyperinput = URIRef(uri+user_name+dataset_name+name+param+'-'+current_time)
        g.add((model,ns.hasHyperparamInput,hyperinput))
        g.add((URIRef(uri+'sklearn-'+name+'-'+param),ns.specifiedBy,hyperinput))
        value = params[param]
        if type(value)==int:
            g.add((hyperinput,ns.hasValue,Literal(value, datatype=XSD.integer)))
        elif type(value)==float:
            g.add((hyperinput,ns.hasValue,Literal(value, datatype=XSD.float)))
        elif type(value)==str:
            g.add((hyperinput,ns.hasValue,Literal(value, datatype=XSD.string)))
        elif type(value)==bool:
            g.add((hyperinput,ns.hasValue,Literal(value, datatype=XSD.boolean)))

g.serialize(format="nt",destination=output_path)
