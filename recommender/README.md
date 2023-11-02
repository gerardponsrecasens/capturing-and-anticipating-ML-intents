# EMBEDDINGS (Work in Progress)

This folder contains main scripts for the **embedding** generation. The main library used is *PyKEEN*, which is has a wide range of embedding models already implemented. In the *experiments.py* file, the script to run the different experiments presented in the Thesis can be found. The *hpo.py* file contains the script to generate the final Hyperparameter Tuning Optimization step.

Additionally, the library *torchKGE* has also been used to implement the fine tuning of models, as it has to be done ad-hoc. This will be used in the UI to generate the recommendations as the user fills in the input.
