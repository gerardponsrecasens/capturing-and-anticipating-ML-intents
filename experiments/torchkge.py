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

train = False
fine_tune = False
plot = False
predict = True

# Load the already stored graph. If we are predicting/fine-tuning, we only need the 
# entity_name_to_idx and relation_name_to_idx. To be stored in triples in the future.

g = Graph()
g.parse("constraintRDF.nt", format="nt")
triples = []
for s, p, o in g:
    if p not in (OWL.sameAs, OWL.inverseOf):
        triples.append((str(s), str(p), str(o)))
        
data = pd.DataFrame(triples)
data.columns = ['from','rel','to']
data = data[['from','to','rel']]
kg_train = KnowledgeGraph(df=data)
entity_name_to_idx = kg_train.ent2ix.copy()
relation_name_to_idx = kg_train.rel2ix.copy()

if train:

    # Define some hyper-parameters for training
    emb_dim = 2
    lr = 0.0004
    n_epochs = 10000
    b_size = 32768
    margin = 0.5

    # Define the model and criterion
    model = TransEModel(emb_dim, kg_train.n_ent, kg_train.n_rel, dissimilarity_type='L2')
    criterion = MarginLoss(margin)

    # Move everything to CUDA if available
    if cuda.is_available():
        cuda.empty_cache()
        model.cuda()
        criterion.cuda()

    # Define the torch optimizer to be used
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Define how negative samples are going to be genereated
    sampler = BernoulliNegativeSampler(kg_train)

    dataloader = DataLoader(kg_train, batch_size=b_size, use_cuda='all')

    iterator = tqdm(range(n_epochs), unit='epoch')
    for epoch in iterator:
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
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
                                                running_loss / len(dataloader)))

    model.normalize_parameters()

    # Save the model state dict and configuration parameters
    torch.save([model.emb_dim,model.n_ent,model.n_rel, model.state_dict()],'model_all.pt')


else:
    # Load stored model
    e,n,r,state= torch.load('model_all.pt')
    model = TransEModel(e,n,r)
    model.load_state_dict(state)




if fine_tune:

    # Load the new data to be included
    g_n = Graph()
    g_n.parse("additional.nt", format="nt")

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
    num_new_relations = len(relation_name_to_idx) - model.n_rel

    if num_new_entities > 0:
        new_ent_emb = torch.randn(num_new_entities, model.emb_dim)
        updated_ent_emb = torch.nn.Parameter(torch.cat([model.ent_emb.weight, new_ent_emb], dim=0))
        model.n_ent += num_new_entities

    # if num_new_relations > 0:
    #     new_rel_emb = torch.randn(num_new_relations, model.emb_dim)
    #     updated_rel_emb = torch.nn.Parameter(torch.cat([model.rel_emb.weight, new_rel_emb], dim=0))
    #     model.n_rel += num_new_relations

    
    # Create a new model with the complete weights and the appropriate size
    state_dict = model.state_dict()
    state_dict['ent_emb.weight'] = updated_ent_emb
    model = TransEModel(model.emb_dim,model.n_ent, model.n_rel)
    model.load_state_dict(state_dict)

    emb_1 = model.get_embeddings()

    # Define the loss function
    criterion = MarginLoss(margin=0.5)

    # Set up the optimizer with a smaller learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Create a DataLoader for the training set
    new_dataloader = DataLoader(kg_fine, batch_size=32, use_cuda=False)
    sampler = BernoulliNegativeSampler(kg_fine)

    # Fine-tune the model with a limited number of training epochs
    n_epochs = 100
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
        

    results = [torch.equal(a,b) for a,b in zip(emb[0],emb_1[0])]

    # Save the model state dict and configuration parameters
    torch.save([model.emb_dim,model.n_ent,model.n_rel, model.state_dict()],'model_fine.pt')


if plot:
    emb = model.get_embeddings()

    diabetes = emb[0][kg_train.ent2ix['http://localhost/8080/intentOntology#diabetes']]
    lymph = emb[0][kg_train.ent2ix['http://localhost/8080/intentOntology#lymph']]
    credit = emb[0][kg_train.ent2ix['http://localhost/8080/intentOntology#credit-approval']]
    RF = emb[0][kg_train.ent2ix['http://localhost/8080/intentOntology#sklearn-RandomForestClassifier']]
    SVC = emb[0][kg_train.ent2ix['http://localhost/8080/intentOntology#sklearn-SVC']]
    LR = emb[0][kg_train.ent2ix['http://localhost/8080/intentOntology#sklearn-LogisticRegression']]
    Norm = emb[0][kg_train.ent2ix['http://localhost/8080/intentOntology#sklearn-Normalizer']]
    MMS = emb[0][kg_train.ent2ix['http://localhost/8080/intentOntology#sklearn-MinMaxScaler']]
    user = emb[0][kg_train.ent2ix['http://localhost/8080/intentOntology#User10']]


    fig, ax = plt.subplots(figsize=(12, 12))

    # Extract 2D coordinates and node names from the embeddings dictionary
    coords = [diabetes,lymph,credit,RF,SVC,LR,Norm,MMS,user]
    node_names = ['Diabates','Lymph','Credit','RF','SVC','LR','Normalizer','MMS','User10']

    # Plot the 2D points
    for i in range(len(coords)):
        ax.scatter(coords[i][0], coords[i][1])

    # Annotate each point with the node name
    for i, node_name in enumerate(node_names):
        ax.annotate(node_name, (coords[i][0], coords[i][1]), fontsize=8)

    ax.grid()
    ax.set_title("2D Node Embeddings")
    plt.show()


if predict:

    head_idx = kg_train.ent2ix['http://localhost/8080/intentOntology#User10adaAlgorithmConstraint-1681731162']
    relation_idx = kg_train.rel2ix['http://localhost/8080/intentOntology#on']

    tail_idx = kg_train.ent2ix['http://localhost/8080/intentOntology#sklearn-SVC']
    score = model.scoring_function(torch.tensor([head_idx]),torch.tensor([tail_idx]),torch.tensor([relation_idx]))    
    print(f"Score for SVC: {score.item()}")


    tail_idx = kg_train.ent2ix['http://localhost/8080/intentOntology#sklearn-RandomForestClassifier']
    score = model.scoring_function(torch.tensor([head_idx]),torch.tensor([tail_idx]),torch.tensor([relation_idx]))    
    print(f"Score for RF: {score.item()}")



    tail_idx = kg_train.ent2ix['http://localhost/8080/intentOntology#sklearn-LogisticRegression']
    score = model.scoring_function(torch.tensor([head_idx]),torch.tensor([tail_idx]),torch.tensor([relation_idx]))
    print(f"Score for LR: {score.item()}")

    tail_idx = kg_train.ent2ix['http://localhost/8080/intentOntology#sklearn-KNeighborsClassifier']
    score = model.scoring_function(torch.tensor([head_idx]),torch.tensor([tail_idx]),torch.tensor([relation_idx]))
    print(f"Score for KNN: {score.item()}")




