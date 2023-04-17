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


def predict_triple_score(emb, head_idx, relation_idx, tail_idx):

    head_emb = emb[0][head_idx]
    relation_emb = emb[0][relation_idx]
    tail_emb = emb[0][tail_idx]
    score = -LA.norm(head_emb + relation_emb - tail_emb) #The closer to 0 the better
    
    return score

train = False
plot = False
predict = True

g = Graph()
g.parse("constraintRDF.nt", format="nt")


triples = []

for s, p, o in g:
    if p not in (OWL.sameAs, OWL.inverseOf):
        triples.append((str(s), str(p), str(o)))
        
data = pd.DataFrame(triples)
data.columns = ['from','rel','to']
torchKGData = KnowledgeGraph(df=data)
# Load dataset
kg_train = torchKGData

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

    print(emb_dim, kg_train.n_ent, kg_train.n_rel)
    torch.save(model.state_dict(), 'model.pt')

else:
    model = TransEModel(2,2652,25, dissimilarity_type='L2')
    model.load_state_dict(torch.load('model.pt'))


emb = model.get_embeddings()


# print(emb[0][kg_train.ent2ix['http://localhost/8080/intentOntology#diabetes']])
# print(emb[0][kg_train.ent2ix['http://localhost/8080/intentOntology#sklearn-RandomForestClassifier']])
# print(emb[0][kg_train.ent2ix['http://localhost/8080/intentOntology#sklearn-SVC']])
# print(emb[0][kg_train.ent2ix['http://localhost/8080/intentOntology#sklearn-LogisticRegression']])


if plot:
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
    score = predict_triple_score(emb, head_idx, relation_idx, tail_idx)
    print(f"Score for SVC: {score}")

    tail_idx = kg_train.ent2ix['http://localhost/8080/intentOntology#sklearn-RandomForestClassifier']
    score = predict_triple_score(emb, head_idx, relation_idx, tail_idx)
    print(f"Score for RF: {score}")

    tail_idx = kg_train.ent2ix['http://localhost/8080/intentOntology#sklearn-LogisticRegression']
    score = predict_triple_score(emb, head_idx, relation_idx, tail_idx)
    print(f"Score for LR: {score}")

    tail_idx = kg_train.ent2ix['http://localhost/8080/intentOntology#sklearn-KNeighborsClassifier']
    score = predict_triple_score(emb, head_idx, relation_idx, tail_idx)
    print(f"Score for KNN: {score}")


