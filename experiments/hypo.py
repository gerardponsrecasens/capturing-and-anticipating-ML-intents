from pykeen.hpo import hpo_pipeline
from pykeen.triples import TriplesFactory
import torch
from pykeen.metrics.ranking import HitsAtK

# Load dataset
tf = TriplesFactory.from_path(r'all_clean.tsv')
training, testing, validation = tf.split([.8, .1, .1], random_state = 1998)



hpo_pipeline_result = hpo_pipeline(
    n_trials=100,
    training = training,
    testing = testing,
    validation = validation,
    training_loop = 'slcwa',
    model='TransH',
    sampler = 'tpe',
    model_kwargs={"random_seed": 1998},
    model_kwargs_ranges=dict(embedding_dim=dict(type=int,low=16,high=64,q=2)),
    optimizer_kwargs_ranges=dict(lr=dict(type=float,low=0.0005,high=0.005)),
    loss='marginranking',
    training_kwargs=dict(num_epochs=200, batch_size=32),
    negative_sampler='bernoulli',
    negative_sampler_kwargs_ranges=dict(num_negs_per_pos=dict(type=int,low=20,high=80,q=10)),
    evaluation_kwargs=dict(batch_size=128),
    metric = "tail.realistic.hits_at_3",
    stopper='early',
    device='gpu'
    )

print(hpo_pipeline_result.objective)

hpo_pipeline_result.save_to_directory(r'./TransH_HPO')