import rdflib

from rdflib import Graph, URIRef, XSD
from rdflib.namespace import  RDF,RDFS

'''
This script creates the Intent Ontology, linked to DMOP and Person, as of 02/05/2023
'''

g = Graph()

# NameSpace
uri = "http://localhost/8080/intentOntology#"
ns = rdflib.Namespace(uri)
dmop = "http://www.e-lico.eu/ontologies/dmo/DMOP/DMOP.owl#"
ns_dmop = rdflib.Namespace(dmop)
dolce = "http://www.loa-cnr.it/ontologies/DOLCE-Lite.owl#"
ns_dolce = rdflib.Namespace(dolce)
person = URIRef("https://schema.org/Person")


###### CLASSES ######
# Workflow
workflow = URIRef(dmop+'DM-Workflow')
#g.add((workflow,RDF.type,RDFS.Class))

# Step
step = URIRef(uri+'Step')
g.add((step,RDFS.subClassOf,URIRef(dmop+'DM-Operation')))
#g.add((step,RDF.type,RDFS.Class))

# User
user = URIRef(uri+'User')
g.add((user,RDF.type,RDFS.Class))
g.add((user,RDF.type,person))
g.add((user,RDFS.subClassOf,URIRef(dolce+'non-physical-endurant')))

# # Feedback
# feedback = URIRef(uri+'Feedback')
# g.add((feedback,RDF.type,RDFS.Class))

# Task
task = URIRef(dmop+'DM-Task')
#g.add((task,RDF.type,RDFS.Class))

# Intent
intent = URIRef(uri+'Intent')
g.add((intent,RDF.type,RDFS.Class))
g.add((intent,RDFS.subClassOf,URIRef(dolce+'non-physical-endurant')))

# Dataset
dataset = ns_dmop.DataSet

# General Dataset Characteristic
genericDatasetCharacteristic = ns_dmop.DataSetCharacteristic

# Hyperparameter
hyperparameter = URIRef(uri+'Hyperparameter')
g.add((hyperparameter,RDF.type,RDFS.Class))
g.add((hyperparameter,RDFS.subClassOf,URIRef(dolce+'non-physical-endurant')))

# Hyperparameter Specification
hyperparamspec = URIRef(uri+'HyperparamSpec')
g.add((hyperparamspec,RDF.type,RDFS.Class))
g.add((hyperparamspec,RDFS.subClassOf,URIRef(dolce+'non-physical-endurant')))

# Implementation
implementation = URIRef(uri+'Implementation')
g.add((implementation,RDFS.subClassOf,URIRef(dmop+'DM-Operator')))
#g.add((implementation,RDF.type,RDFS.Class))

# Algorithm/Preprocess
algoprepro = URIRef(dmop+'DM-Algorithm')

# Model Evaluation
modelEval = URIRef(uri+'ModelEvaluation')
g.add((modelEval,RDF.type,RDFS.Class))
g.add((modelEval,RDFS.subClassOf,URIRef(dolce+'non-physical-endurant')))

# Evaluation Requirement
evalReq = URIRef(uri+'EvaluationRequirement')
g.add((evalReq,RDF.type,RDFS.Class))
g.add((evalReq,RDFS.subClassOf,URIRef(dolce+'non-physical-endurant')))

# Method
method = URIRef(uri+'Method')
g.add((method,RDF.type, RDFS.Class))
g.add((method,RDFS.subClassOf,URIRef(dolce+'non-physical-endurant')))

# Metric
metric = URIRef(uri+'Metric')
g.add((metric,RDF.type,RDFS.Class))
g.add((metric,RDFS.subClassOf,URIRef(dolce+'non-physical-endurant')))

# Constraint
Constraint = URIRef(uri+'Constraint')
g.add((Constraint,RDF.type,RDFS.Class))
g.add((Constraint,RDFS.subClassOf,URIRef(dolce+'non-physical-endurant')))

# ConstraintHyperparameter
HypConstraint = URIRef(uri+'ConstraintHyperparameter')
g.add((HypConstraint,RDF.type,RDFS.Class))
g.add((HypConstraint,RDFS.subClassOf,Constraint))


# ConstraintWorkfloCharacteristic
WorkflowConstraint = URIRef(uri+'ConstraintWorkflow')
g.add((WorkflowConstraint,RDF.type,RDFS.Class))
g.add((WorkflowConstraint,RDFS.subClassOf,Constraint))

# ConstraintAlgorithm
AlgoConstraint = URIRef(uri+'ConstraintAlgorithm')
g.add((AlgoConstraint,RDF.type,RDFS.Class))
g.add((AlgoConstraint,RDFS.subClassOf,Constraint))

# ConstraintPreprocessingAlgorithm
AlgoConstraint = URIRef(uri+'ConstraintPreprocessingAlgorithm')
g.add((AlgoConstraint,RDF.type,RDFS.Class))
g.add((AlgoConstraint,RDFS.subClassOf,Constraint))

#Min-Max-Equal
MinMaxEqual = URIRef(uri+'Min-Max-Equal')
g.add((MinMaxEqual,RDF.type,RDFS.Class))
g.add((MinMaxEqual,RDFS.subClassOf,URIRef(dmop+'DecisionStrategy'))) 

# Use-NoUSe
UseNoUse = URIRef(uri+'Use-NoUse')
g.add((UseNoUse,RDF.type,RDFS.Class))
g.add((UseNoUse,RDFS.subClassOf,URIRef(dmop+'DecisionStrategy')))

# Workflow Characteristic  (we add the particular workflo char. as a sublass of them: memory, speed...)
genericWorkflowCharacteristic = ns_dmop.WorkflowCharacteristic


###### PROPERTIES ######

# # Defined on: A task is defined over a dataset
# definedOn = URIRef(uri+'definedOn')
# g.add((definedOn,RDF.type,RDF.Property))
# g.add((definedOn,RDFS.domain,task))
# g.add((definedOn,RDFS.range,dataset))

# Runs: a user runs a workflow
runs = URIRef(uri+'runs')
g.add((runs,RDF.type,RDF.Property))
g.add((runs,RDFS.domain,user))
g.add((runs,RDFS.range,workflow))

# Has Feedback: a user has feedback about a workflow (It can be more than a simple feedback, e.g., notes + rating)
hasFeedback = URIRef(uri+'hasFeedback')
g.add((hasFeedback,RDF.type,RDF.Property))
g.add((hasFeedback,RDFS.domain,workflow))
# g.add((hasFeedback,RDFS.range,feedback))

# Achieves: a worflow is run to achieve a task
achieves = URIRef(uri+'achieves')
g.add((achieves,RDF.type,RDF.Property))
g.add((achieves,RDFS.domain,workflow))
g.add((achieves,RDFS.range,task))

# hasRequirement: a task can define an evaluation requirement (i.e., what needs to be achieved in the task)
hasRequirement = URIRef(uri+'hasRequirement')
g.add((hasRequirement,RDF.type,RDF.Property))
g.add((hasRequirement,RDFS.domain,task))
g.add((hasRequirement,RDFS.range,evalReq))

# How Eval: how a specification is evaluated (e.g., minimum accuracy must be mmm, maximum entropy must be nnn,...)
howEval = URIRef(uri+'howEval')
g.add((howEval,RDF.type,RDF.Property))
g.add((howEval,RDFS.domain,evalReq))
g.add((howEval,RDFS.range,MinMaxEqual))

# With Method: with which method requierement is evaluated (e.g., 5-foldCV, train test split...)
withMethod = URIRef(uri+'withMethod')
g.add((withMethod,RDF.type,RDF.Property))
g.add((withMethod,RDFS.domain,evalReq))
g.add((withMethod,RDFS.range,method))

#On Metric: to which metric the evaluation is specified (e.g., accuracy)
onMetric = URIRef(uri+'onMetric')
g.add((onMetric,RDF.type,RDF.Property))
g.add((onMetric,RDFS.domain,evalReq))
g.add((onMetric,RDFS.range,metric))

# Is Satisfied: if a evaluation specification or a constraint is satisfied or not (boolean)
isSatisfied = URIRef(uri+'isSatisfied')
g.add((isSatisfied,RDF.type,RDF.Property))
g.add((isSatisfied,RDFS.range,XSD.boolean))

# Has Constraint: a task can have a language constraint 
hasConstraint = URIRef(uri+'hasConstraint')
g.add((hasConstraint,RDF.type,RDF.Property))
g.add((hasConstraint,RDFS.domain,task))
g.add((hasConstraint,RDFS.range,Constraint))

# Has Constraint Value: a task can have a value constraint
hasConstraintValue = URIRef(uri+'hasConstraintValue')
g.add((hasConstraintValue,RDF.type,RDF.Property))
g.add((hasConstraintValue,RDFS.domain,task))

# On constraint: the blank node of a Constraint Value links to the Constraint
onC = URIRef(uri+'onConstraint')
g.add((onC,RDF.type,RDF.Property))
g.add((onC,RDFS.range,Constraint))

# On: on what the constraint is imposed (e.g., hyperparameter, implementation, workflow characteristic)
on = URIRef(uri+'on')
g.add((on,RDF.type,RDF.Property))
g.add((on,RDFS.domain,Constraint))

# How Constraint: how a constaint is specified (e.g., number of trees, not using SVM...). The range is Use-NoUse and Min-Max-Equal
howConstraint = URIRef(uri+'howConstraint')
g.add((howConstraint,RDF.type,RDF.Property))
g.add((howConstraint,RDFS.domain,Constraint))

# Is Hard: if a constraint needs to be satisfied or if it is only a user's preference
isHard = URIRef(uri+'isHard')
g.add((isHard,RDF.type,RDF.Property))
g.add((isHard,RDFS.domain,Constraint))
g.add((isHard,RDFS.range,XSD.boolean))

# Has Intent: a task has an intent
hasIntent = URIRef(uri+'hasIntent')
g.add((hasIntent,RDF.type,RDF.Property))
g.add((hasIntent,RDFS.domain,task))
g.add((hasIntent,RDFS.range,intent))

# Has Input: a workflow has one (or more) datasets as input
hasInput = URIRef(uri+'hasInput')
g.add((hasInput,RDF.type,RDF.Property))
g.add((hasInput,RDFS.domain,workflow))
g.add((hasInput,RDFS.range,dataset))

# Has Output: a worflow has some outputs
hasOutput = URIRef(uri+'hasOutput')
g.add((hasOutput,RDF.type,RDF.Property))
g.add((hasOutput,RDFS.domain,workflow))
g.add((hasOutput,RDFS.range,modelEval))

# Has Step: a workflow is compossed of steps
hasStep = URIRef(uri+'hasStep')
g.add((hasStep,RDF.type,RDF.Property))
g.add((hasStep,RDFS.domain,workflow))
g.add((hasStep,RDFS.range,step))

# # Realizes: the high-level view parts of a workflow (e.g., Standardize, Impute,Classify...)
# realizes = URIRef(uri+'realizes')
# g.add((realizes,RDF.type,RDF.Property))
# g.add((realizes,RDFS.domain,workflow))
# g.add((realizes,RDFS.range,algoprepro))

# Followed By: A step can be followed by another
followedBy = URIRef(uri+'followedBy')
g.add((followedBy,RDF.type,RDF.Property))
g.add((followedBy,RDFS.domain,step))
g.add((followedBy,RDFS.range,step))

# Order: captures the order in which a step takes part in a workflow
order = URIRef(uri+'order')
g.add((order,RDF.type,RDF.Property))
g.add((order,RDFS.domain,step))
g.add((order,RDFS.range,XSD.integer))

# Has Hyperparam Input: a step can have different hyperparameters associated with it
hasHyperparamInput = URIRef(uri+'hasHyperparamInput')
g.add((hasHyperparamInput,RDF.type,RDF.Property))
g.add((hasHyperparamInput,RDFS.domain,step))
g.add((hasHyperparamInput,RDFS.range,hyperparamspec))

# Has Implementation: a step has some implementation (i.e., code from a library)
hasImplementation = URIRef(uri+'hasImplementation')
g.add((hasImplementation,RDF.type,RDF.Property))
g.add((hasImplementation,RDFS.domain,step))
g.add((hasImplementation,RDFS.range,implementation))

# Implements: an implementaton implements an algorithm or a process
implements = URIRef(uri+'implements')
g.add((implements,RDF.type,RDF.Property))
g.add((implements,RDFS.domain,implementation))
g.add((implements,RDFS.range,algoprepro))

# Has Hyperparameter: an implementation one or more hyperparameters to be specified
hasHyperparam = URIRef(uri+'hasHyperparameter')
g.add((hasHyperparam,RDF.type,RDF.Property))
g.add((hasHyperparam,RDFS.domain,implementation))
g.add((hasHyperparam,RDFS.range,hyperparameter))

# Specified By: an hyperparameter is specified by an hyperparameter specification
specifiedBy = URIRef(uri+'specifiedBy')
g.add((specifiedBy,RDF.type,RDF.Property))
g.add((specifiedBy,RDFS.domain,hyperparameter))
g.add((specifiedBy,RDFS.range,hyperparamspec))

# Specifies: a model evaluation specifies a metric
specifies = URIRef(uri+'specifies')
g.add((specifies,RDF.type,RDF.Property))
g.add((specifies,RDFS.domain,modelEval))
g.add((specifies,RDFS.range,metric))

# Has Value
hasValue = URIRef(uri+'hasValue')
g.add((hasValue,RDF.type,RDF.Property))

# Satisfies: A particular Algorithm Satisfies an Intent (e.g., SVM -> Predict)
satisfies = URIRef(uri+'satisfies')
g.add((satisfies,RDF.type,RDF.Property))
g.add((satisfies,RDFS.domain,algoprepro))
g.add((satisfies,RDFS.range,intent))




### GENERAL INSTANTIATON

use = URIRef(uri+'Use')
no_use = URIRef(uri+'NoUse')
g.add((use,RDF.type,UseNoUse))
g.add((no_use,RDF.type,UseNoUse))

maximize = URIRef(uri+'Max')
minimize = URIRef(uri+'Min')
equal = URIRef(uri+'Equal')
g.add((maximize,RDF.type,MinMaxEqual))
g.add((minimize,RDF.type,MinMaxEqual))
g.add((equal,RDF.type,MinMaxEqual))


precision = URIRef(uri+'Precision')
accuracy = URIRef(uri + 'Accuracy')
f1 = URIRef(uri+'F1')
AUC = URIRef(uri+'AUC')
R2 = URIRef(uri+'R2')
recall = URIRef(uri+'Recall')
g.add((precision,RDF.type,metric))
g.add((accuracy,RDF.type,metric))
g.add((f1,RDF.type,metric))
g.add((AUC,RDF.type,metric))
g.add((R2,RDF.type,metric))
g.add((recall,RDF.type,metric))


trainTestSplit = URIRef(uri+'TrainTestSplit')
CV = URIRef(uri+'CV')
foldCV5 = URIRef(uri+'5foldCV')
foldCV10 = URIRef(uri+'10foldCV')
g.add((recall,RDF.type,method))
g.add((CV,RDF.type,method))
g.add((foldCV5,RDF.type,method))
g.add((foldCV10,RDF.type,method))


predict = URIRef(uri+'Predict')
describe = URIRef(uri+'Describe')
explain = URIRef(uri+'Explain')
classification = URIRef(uri+'Classification')
regression = URIRef(uri+'Regression')
clustering = URIRef(uri+'Clustering')
g.add((predict,RDF.type,intent))
g.add((describe,RDF.type,intent))
g.add((explain,RDF.type,intent))
g.add((classification,RDF.type,intent))
g.add((regression,RDF.type,intent))
g.add((clustering,RDF.type,intent))


not_use_pre = URIRef(uri+'ConstraintNoPreprocessing')
g.add((not_use_pre,RDF.type,Constraint))



#### SERIALIZE 

g.serialize(format="nt",destination="intent_ontology.nt")



