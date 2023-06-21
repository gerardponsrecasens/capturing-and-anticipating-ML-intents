from rdflib import URIRef, Literal, Graph
from rdflib.namespace import OWL
import torch
import rdflib
from torch import cuda
from torch.optim import Adam
import pandas as pd
from torchkge.models import TransEModel
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader
from torchkge.data_structures import KnowledgeGraph
from numpy import linalg as LA
import numpy as np
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

'''
Recommendation Engine for Link Prediction.
'''

def recommendation(stage, task,
                   n_epochs = 500, lr = 0.0001): #Stage can be 1 (recommend Intent) or 2 (recommend const)

    model_path = './app/static/model/model_new_16.pt'

    # Load stored model
    e,n,r,entity_name_to_idx,relation_name_to_idx,state= torch.load(model_path)
    model = TransEModel(e,n,r)
    model.load_state_dict(state)


    # Load the new data to be included
    g_n = Graph()

    if stage == 1:
        g_n.parse('./app/static/triples/user_dataset.nt', format="nt")
    else:
        g_n.parse('./app/static/triples/intent.nt', format="nt")

    new_triples = []

    for s, p, o in g_n:
        if p not in (OWL.sameAs, OWL.inverseOf):
            new_triples.append((str(s), str(p), str(o)))
            
    new_data = pd.DataFrame(new_triples)
    new_data.columns = ['from','rel','to']

    # Add the new entities and relations to the dictionaries
    for h, r, t in new_triples:
        if h not in entity_name_to_idx:
            entity_name_to_idx[h] = len(entity_name_to_idx)
        if r not in relation_name_to_idx:
            relation_name_to_idx[r] = len(relation_name_to_idx)
        if t not in entity_name_to_idx:
            entity_name_to_idx[t] = len(entity_name_to_idx)

    # Create the KnowledgeGraph object
    kg_fine = KnowledgeGraph(df=new_data,ent2ix=entity_name_to_idx,rel2ix=relation_name_to_idx)
    

    # Update the entity and relation embeddings in the model. For the new entities intialize the 
    # weights at random.
    num_new_entities = len(entity_name_to_idx) - model.n_ent
    print('#######################')
    print(num_new_entities)
    print(model.n_ent)


    if num_new_entities > 0:
        new_ent_emb = torch.randn(num_new_entities, model.emb_dim)
        updated_ent_emb = torch.nn.Parameter(torch.cat([model.ent_emb.weight, new_ent_emb], dim=0))
        model.n_ent += num_new_entities

    # Create a new model with the complete weights and the appropriate size
    state_dict = model.state_dict()

    if num_new_entities > 0:
        state_dict['ent_emb.weight'] = updated_ent_emb

    model = TransEModel(model.emb_dim,model.n_ent, model.n_rel)
    model.load_state_dict(state_dict)

    # Define the loss function
    criterion = MarginLoss(margin=0.5)

    # Set up the optimizer with a smaller learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create a DataLoader for the training set
    new_dataloader = DataLoader(kg_fine, batch_size=32, use_cuda=False)
    sampler = BernoulliNegativeSampler(kg_fine)

    # Fine-tune the model with a limited number of training epochs

    iterator = tqdm(range(n_epochs), unit='epoch')


    for epoch in iterator:
        running_loss = 0.0
        for i, batch in enumerate(new_dataloader):
            h, t, r = batch[0], batch[1], batch[2]
            n_h, n_t = sampler.corrupt_batch(h, t, r)

            optimizer.zero_grad()

            # forward + backward + optimize
            pos, neg = model(h, t, r, n_h, n_t)
            loss = criterion(pos, neg)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        iterator.set_description(
            'Epoch {} | mean loss: {:.5f}'.format(epoch + 1,
                                                running_loss / len(new_dataloader)))
    
    model.normalize_parameters()
    emb = model.get_embeddings()

    # Save again for future use

    torch.save([model.emb_dim,model.n_ent,model.n_rel,entity_name_to_idx,relation_name_to_idx,model.state_dict()],model_path)



    if stage == 1: # Recommend Intent

        head_idx = entity_name_to_idx[str(task)]

        relation_idx = relation_name_to_idx['http://localhost/8080/intentOntology#hasIntent']

        tail_idx = entity_name_to_idx['http://localhost/8080/intentOntology#Classification']

        predict_score = model.scoring_function(torch.tensor([head_idx]),torch.tensor([tail_idx]),torch.tensor([relation_idx])) 

        print(predict_score)

        return 'Classification'
    
    elif stage == 2: #Recommend Eval Requirement (Metric) and Algorithm Constraint


        # RECOMEND ALGORITHM:
        algorithms = ['SVC','KNeighborsClassifier','RandomForestClassifier','LogisticRegression']
        algorithm_scores = []

        head_idx = entity_name_to_idx[str(task)]
        relation_idx = relation_name_to_idx['http://localhost/8080/intentOntology#hasConstraint']

        for algorithm in algorithms:
            tail_idx = entity_name_to_idx['http://localhost/8080/intentOntology#Constraintsklearn-'+algorithm]
            algorithm_scores.append(model.scoring_function(torch.tensor([head_idx]),torch.tensor([tail_idx]),torch.tensor([relation_idx])))


        algorithm_constraint = algorithms[algorithm_scores.index(max(algorithm_scores))]

        # RECOMMEND PREPROCESSOR

        algorithms = ['NoPre','StandardScaler','MinMaxScaler','Normalizer']
        algorithm_scores = []

        tail_idx = entity_name_to_idx['http://localhost/8080/intentOntology#ConstraintNoPreprocessing']
        algorithm_scores.append(model.scoring_function(torch.tensor([head_idx]),torch.tensor([tail_idx]),torch.tensor([relation_idx])))  

        tail_idx = entity_name_to_idx['http://localhost/8080/intentOntology#Constraintsklearn-StandardScaler']
        algorithm_scores.append(model.scoring_function(torch.tensor([head_idx]),torch.tensor([tail_idx]),torch.tensor([relation_idx])))  

        tail_idx = entity_name_to_idx['http://localhost/8080/intentOntology#Constraintsklearn-MinMaxScaler']
        algorithm_scores.append(model.scoring_function(torch.tensor([head_idx]),torch.tensor([tail_idx]),torch.tensor([relation_idx])))  

        tail_idx = entity_name_to_idx['http://localhost/8080/intentOntology#Constraintsklearn-Normalizer']
        algorithm_scores.append(model.scoring_function(torch.tensor([head_idx]),torch.tensor([tail_idx]),torch.tensor([relation_idx])))

        prepro_constraint = algorithms[algorithm_scores.index(max(algorithm_scores))]


        # RECOMMEND METRIC

        metrics = ['F1','AUC','Precision','Accuracy']
        metrics_scores = []

        head_idx = entity_name_to_idx[str(task)]
        relation_idx = relation_name_to_idx['http://localhost/8080/intentOntology#hasRequirement']

        for metric in metrics:
            tail_idx = entity_name_to_idx['http://localhost/8080/intentOntology#EvalReq'+metric+'TrainTestSplit']
            metrics_scores.append(model.scoring_function(torch.tensor([head_idx]),torch.tensor([tail_idx]),torch.tensor([relation_idx])))

        metric = metrics[metrics_scores.index(max(metrics_scores))]


        return algorithm_constraint, prepro_constraint ,metric


'''
TO DO: once more dataset are given to the user, the Intent must be filtered. Right now, the filter has been done manually, as only classification methods
can be used.
'''