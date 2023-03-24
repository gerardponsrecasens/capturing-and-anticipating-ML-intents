import rdflib
import time
import os
import pickle
from random import randrange
from rdflib import Graph, URIRef, XSD, Literal
from rdflib.namespace import  RDF,RDFS



input_path = r'./results/'
output_path = 'tpot.nt'

current_time = str(int(time.time()))

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
        g.add((modelEval,ns.hasValue,Literal(data['metric_value'], datatype=XSD.float)))


        steps = data['steps']
        num_steps = len(steps)
        old_step = None

        for step in range(num_steps):
        
                new_step = URIRef(uri+'Step'+str(step)+user_name+dataset_name+'-'+current_time)
                g.add((workflow,ns.hasStep,new_step))
                g.add((new_step,ns.order,Literal(step+1, datatype=XSD.integer)))
                
                if old_step!=None:
                        g.add((old_step,ns.followedBy,new_step))
                
                if steps[step][0] == 'stackingestimator':
                        algo = str(steps[step]).split('estimator=')[1].split('(')[0]
                        g.add((new_step,ns.hasImplementation,URIRef(uri+'sklearn-'+algo)))
                        
                        
                        params = steps[step][1].get_params()
                        
                        for param in params:
                                hyperinput = URIRef(uri+user_name+dataset_name+algo+param[11:]+'-'+current_time)
                                value = params[param]
                                
                                if type(value)==int:
                                        g.add((hyperinput,ns.hasValue,Literal(value, datatype=XSD.integer)))
                                        g.add((new_step,ns.hasHyperparamInput,hyperinput))
                                        g.add((URIRef(uri+'sklearn-'+algo+'-'+param[11:]),ns.specifiedBy,hyperinput))
                                elif type(value)==float:
                                        g.add((hyperinput,ns.hasValue,Literal(value, datatype=XSD.float)))
                                        g.add((new_step,ns.hasHyperparamInput,hyperinput))
                                        g.add((URIRef(uri+'sklearn-'+algo+'-'+param[11:]),ns.specifiedBy,hyperinput))
                                elif type(value)==str:
                                        g.add((hyperinput,ns.hasValue,Literal(value, datatype=XSD.string)))
                                        g.add((new_step,ns.hasHyperparamInput,hyperinput))
                                        g.add((URIRef(uri+'sklearn-'+algo+'-'+param[11:]),ns.specifiedBy,hyperinput))
                                elif type(value)==bool:
                                        g.add((hyperinput,ns.hasValue,Literal(value, datatype=XSD.boolean)))
                                        g.add((new_step,ns.hasHyperparamInput,hyperinput))
                                        g.add((URIRef(uri+'sklearn-'+algo+'-'+param[11:]),ns.specifiedBy,hyperinput))
                                
                                
                        
                else:
                        algo = str(steps[step]).split(', ')[1].split('(')[0]
                        g.add((new_step,ns.hasImplementation,URIRef(uri+'sklearn-'+algo)))
                        
                        params = steps[step][1].get_params()
                        for param in params:
                                hyperinput = URIRef(uri+user_name+dataset_name+algo+param+'-'+current_time)
                                value = params[param]
                        
                                if type(value)==int:
                                        g.add((hyperinput,ns.hasValue,Literal(value, datatype=XSD.integer)))
                                        g.add((new_step,ns.hasHyperparamInput,hyperinput))
                                        g.add((URIRef(uri+'sklearn-'+algo+'-'+param),ns.specifiedBy,hyperinput))
                                elif type(value)==float:
                                        g.add((hyperinput,ns.hasValue,Literal(value, datatype=XSD.float)))
                                        g.add((new_step,ns.hasHyperparamInput,hyperinput))
                                        g.add((URIRef(uri+'sklearn-'+algo+'-'+param),ns.specifiedBy,hyperinput))
                                elif type(value)==str:
                                        g.add((hyperinput,ns.hasValue,Literal(value, datatype=XSD.string)))
                                        g.add((new_step,ns.hasHyperparamInput,hyperinput))
                                        g.add((URIRef(uri+'sklearn-'+algo+'-'+param),ns.specifiedBy,hyperinput))
                                elif type(value)==bool:
                                        g.add((hyperinput,ns.hasValue,Literal(value, datatype=XSD.boolean)))
                                        g.add((new_step,ns.hasHyperparamInput,hyperinput))
                                        g.add((URIRef(uri+'sklearn-'+algo+'-'+param),ns.specifiedBy,hyperinput))
                        
                                
                old_step = new_step


g.serialize(format="nt",destination=output_path)
