import rdflib
import time
import os
import pickle
from random import randrange
from rdflib import Graph, URIRef, XSD, Literal, BNode
from rdflib.namespace import  RDF,RDFS


'''
Script to generate RDF triples from the HyperOpt results.
'''

input_path = r'./store/'
output_path = "RDFtriples.nt"
classification = True

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
    with open(input_path+file, 'rb') as handle:
        data = pickle.load(handle)

    user_name = data['user']
    dataset_name = data['dataset']
    current_time = str(int(time.time()))



    # GENERAL WORKFLOW

    user = URIRef(uri+user_name)
    workflow = URIRef(uri+'Worflow'+user_name+dataset_name+'-'+current_time)
    dataset = URIRef(uri+dataset_name)
    task = URIRef(uri+'Task'+user_name+dataset_name+'-'+current_time)

    g.add((user,ns.runs,workflow))
    g.add((workflow,ns.hasFeedback,Literal(randrange(11), datatype=XSD.integer)))
    g.add((dataset,RDF.type,ns_dmop.DataSet))
    g.add((workflow,ns.hasInput,dataset))
    g.add((workflow,ns.achieves,task))

    optimization_time = data.get('time', None)

    if optimization_time:
        opt_time = URIRef(uri+'Time'+user_name+dataset_name+'-'+current_time)
        g.add((workflow,URIRef(dolce+'has-quality'),opt_time))
        g.add((opt_time,RDF.type,ns_dmop.OptimizationTime))
        g.add((opt_time,ns.hasValue,Literal(optimization_time, datatype=XSD.float)))


    ## USER INPUT 

    '''
    In the future, the following two paragraphs should be instantiated previosly, with the creation of the ontology
    '''

    use = URIRef(uri+'Use')
    no_use = URIRef(uri+'NoUse')

    maximize = URIRef(uri+'Max')
    minimize = URIRef(uri+'Min')
    equal = URIRef(uri+'Equal')


    # Task: Requirements 

    if classification:
        intent = URIRef(uri+'Classification')
    else:
        intent = URIRef(uri+'Regression')
        
    g.add((task,ns.hasIntent,intent))


    evalRequirement = URIRef(uri+'EvalReq'+data['metric_name']+'TrainTestSplit')
    traintestsplit = URIRef(uri+'TrainTestSplit')
    metric = URIRef(uri+data['metric_name'])

    g.add((task,ns.hasRequirement,evalRequirement))
    g.add((evalRequirement,ns.withMethod,traintestsplit))
    g.add((evalRequirement,ns.onMetric,metric))
    g.add((evalRequirement,ns.howEval,maximize))

    modelEval = URIRef(uri+'ModelEval'+user_name+dataset_name+'-'+current_time)
    g.add((workflow,ns.hasOutput,modelEval))
    g.add((modelEval,ns.specifies,metric))
    g.add((modelEval,ns.hasValue,Literal(data['metric_value'], datatype=XSD.float)))


    # Task: Constraints

    algorithm_constraint = data.get('algorithm_constraint', None)

    if algorithm_constraint:
        const = URIRef(uri+'Constraint'+'sklearn-'+algorithm_constraint)
        g.add((const,RDF.type,URIRef(uri+'ConstraintAlgorithm')))
        g.add((task,ns.hasConstraint,const))
        g.add((const,ns.isHard,Literal(True, datatype=XSD.boolean)))
        g.add((const,ns.howConstraint,use))
        g.add((const,ns.on,URIRef(uri+'sklearn-'+algorithm_constraint)))


    hyp_constraint = data.get('hyperparam_constraints', None)
    
    if hyp_constraint:
        for i,hycon in enumerate(hyp_constraint):
            const = URIRef(uri+'Constraint'+algorithm_constraint+'-'+hycon)
            g.add((const,RDF.type,URIRef(uri+'ConstraintHyperparameter')))
            g.add((task,ns.hasConstraint,const))
            g.add((const,ns.isHard,Literal(True, datatype=XSD.boolean)))
            g.add((const,ns.howConstraint,equal))


            value = hyp_constraint[hycon]
            bn = BNode()
            g.add((task,ns.hasConstraintValue,bn))
            g.add((bn,ns.onConstraint,const))

            if type(value)==int:
                g.add((bn,ns.hasValue,Literal(value, datatype=XSD.integer)))
            elif type(value)==float:
                g.add((bn,ns.hasValue,Literal(value, datatype=XSD.float)))
            elif type(value)==str:
                g.add((bn,ns.hasValue,Literal(value, datatype=XSD.string)))
            elif type(value)==bool:
                g.add((bn,ns.hasValue,Literal(value, datatype=XSD.boolean)))

            g.add((const,ns.on,URIRef(uri+'sklearn-'+algorithm_constraint+'-'+hycon)))


    preprocessor_constraint = data.get('preprocessor_constraint', None)
    not_use_pre = URIRef(uri+'ConstraintNoPreprocessing')


    if preprocessor_constraint:
        if preprocessor_constraint != 'NoPre':
            const = URIRef(uri+'Constraint'+'sklearn-'+preprocessor_constraint)
            g.add((const,RDF.type,URIRef(uri+'ConstraintPreprocessingAlgorithm')))
            g.add((task,ns.hasConstraint,const))
            g.add((const,ns.howConstraint,use))
            g.add((const,ns.on,URIRef(uri+'sklearn-'+preprocessor_constraint)))
        else: 
            g.add((task,ns.hasConstraint,not_use_pre))
            g.add((const,ns.isHard,Literal(True, datatype=XSD.boolean)))

    max_time = data.get('max_time', None)
    
    if optimization_time and max_time:
        const = URIRef(uri+'TimeConstraint')
        g.add((const,RDF.type,URIRef(uri+'ConstraintWorkflow')))
        g.add((task,ns.hasConstraint,const))
        g.add((const,ns.on,opt_time))
        g.add((const,ns.isHard,Literal(True, datatype=XSD.boolean)))

        bn = BNode()
        g.add((task,ns.hasConstraintValue,bn))
        g.add((bn,ns.onConstraint,const))
        g.add((bn,ns.hasValue,Literal(max_time, datatype=XSD.boolean)))
        

    ## PIPELINE AND STEPS

    if len(data['pipeline']['preprocs']) != 0:
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
