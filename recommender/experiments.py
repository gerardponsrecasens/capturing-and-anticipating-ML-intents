import pykeen
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import torch


# Indicate if in evaluation the relationships should be filtered
filtered_evaluation = False
evaluation_relation_whitelist = {'<http://localhost/8080/intentOntology#hasConstraint>', 
                                 '<http://localhost/8080/intentOntology#hasIntent>'}

# Load dataset
tf = TriplesFactory.from_path(r'data.tsv')
training, testing, validation = tf.split([.8, .1, .1], random_state = 1998)


epochs = 300
models = ['TransE','TransH','DistMult','ComplEx','RotatE','TransR','RGCN']
seeds = [11,22,33,44,55]
emb_dimensions = [16,32,64,128]
learning_rates = [0.0001,0.001,0.01]
num_negatives = [1,5,10,20]




for seed in seeds:
    for model in models:    
        for emb_dimension in emb_dimensions:        
            for learning_rate in learning_rates:            
                for num_negative in num_negatives:
                
                    # Define the model configuration
                    model_kwargs = {'embedding_dim': emb_dimension}

                    # Define the training configuration
                    training_kwargs = {'batch_size':32,'num_epochs': epochs, 'use_tqdm_batch':False}

                    # Define the optimizer configuration
                    optimizer_kwargs = {'lr':learning_rate}

                    # Define the evaluator configuration
                    evaluation_kwargs= {'batch_size':32,'use_tqdm':False}

                    # Define negative sampler evaluation
                    negative_sampler_kwargs = {'num_negs_per_pos':num_negative}


                    # Define the pipeline
                    pipe = pipeline(
                        model = model,
                        training = training,
                        testing = testing,
                        validation = validation,
                        training_loop = 'slcwa',
                        model_kwargs = model_kwargs,
                        training_kwargs = training_kwargs,
                        optimizer_kwargs = optimizer_kwargs,
                        evaluation_kwargs = evaluation_kwargs,
                        negative_sampler_kwargs = negative_sampler_kwargs,
                        evaluation_relation_whitelist = evaluation_relation_whitelist if filtered_evaluation else None,
                        metric = 'hits@3',
                        negative_sampler = 'bernoulli',
                        stopper = 'early',
                        device = 'gpu',
                        random_seed=seed
                    )


                    parameters = {'model':model,'emb_dim':emb_dimension,'learning_rate':learning_rate,
                                'negative_per_pos':num_negative,'n_epochs':epochs, 'filtered_eval':filtered_evaluation
                                }
                    test_results = pipe.metric_results.to_df()


                    name = model + '_'+str(emb_dimension) + '_' + str(learning_rate) + '_' + str(num_negative)+'_'+ str(epochs) + '_' + str(filtered_evaluation)
                    torch.save([parameters,test_results,pipe.losses],'./results/'+name+'.pkl')




training_inv = TriplesFactory.from_labeled_triples(training.triples, create_inverse_triples=True)
validation_inv = TriplesFactory.from_labeled_triples(validation.triples, create_inverse_triples=True)

models = ['CompGCN']
for seed in seeds:
    for model in models:    
        for emb_dimension in emb_dimensions:        
            for learning_rate in learning_rates:            
                for num_negative in num_negatives:
                
                    # Define the model configuration
                    model_kwargs = {'embedding_dim': emb_dimension}

                    # Define the training configuration
                    training_kwargs = {'batch_size':32,'num_epochs': epochs, 'use_tqdm_batch':False}

                    # Define the optimizer configuration
                    optimizer_kwargs = {'lr':learning_rate}

                    # Define the evaluator configuration
                    evaluation_kwargs= {'batch_size':32,'use_tqdm':False}

                    # Define negative sampler evaluation
                    negative_sampler_kwargs = {'num_negs_per_pos':num_negative}


                    # Define the pipeline
                    pipe = pipeline(
                        model = model,
                        training = training_inv,
                        testing = testing,
                        validation = validation_inv,
                        training_loop = 'slcwa',
                        model_kwargs = model_kwargs,
                        training_kwargs = training_kwargs,
                        optimizer_kwargs = optimizer_kwargs,
                        evaluation_kwargs = evaluation_kwargs,
                        negative_sampler_kwargs = negative_sampler_kwargs,
                        evaluation_relation_whitelist = evaluation_relation_whitelist if filtered_evaluation else None,
                        metric = 'hits@3',
                        negative_sampler = 'bernoulli',
                        stopper = 'early',
                        device = 'gpu',
                        random_seed=seed
                    )


                    parameters = {'model':model,'emb_dim':emb_dimension,'learning_rate':learning_rate,
                                'negative_per_pos':num_negative,'n_epochs':epochs, 'filtered_eval':filtered_evaluation
                                }
                    test_results = pipe.metric_results.to_df()


                    name = model + '_'+str(emb_dimension) + '_' + str(learning_rate) + '_' + str(num_negative)+'_'+ str(epochs) + '_' + str(filtered_evaluation)
                    torch.save([parameters,test_results,pipe.losses],'./results/'+name+'.pkl')


