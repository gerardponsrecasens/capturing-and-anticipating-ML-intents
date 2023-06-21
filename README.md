# Capturing and anticipating user intents in complex analytics workflows

This repository contains the main scripts for the **Capturing and Anticipating User Intents in Complex Analytics Workflows** for my *Master's Thesis*, which correspond the Task 4.3 of the **ExtremeXP** project, focusing on two key functionalities of the ExtremeXP framework: 
- Capturing of user intents, user preferences, and constraints in a newly designed Knowledge Graph.
- Suggesting the end-users the different input parameters, enabling them to interact in a more efficient and intuitive manner with the system.


To achieve it, different steps have been followed: 
1. A Knowledge Graph has been **designed** to capture all the requiered concepts and relationships related to the data mining domain and the user interaction with the system, so that the generated knowledge graph can be used in a future step to derive valid workflows.
2. The Knowledge Graph has been **populated** with ML workflows from various sources.
3. Inference algorithms have been applied to the populated knowledge graph, so that it can be used to find **relevant intents, constraint and preferences** (e.g., serve as a reccomender system).



![image](https://github.com/gerardponsrecasens/capturing-and-anticipating-ML-intents/assets/95172600/95dda1c3-cd15-4ff8-9afa-6354e2b50e16)



The repository is organized as follows: in the *Knowledge Graph* folder, the ontology .nt files can be found, alongside with the scripts used to generate them. Then, in the ontology *Knowledge Graph* folder the scripts for creating the instances from the different sources can be found. Then, the *Recommender* folder contains the scripts used to experiment with the anticipation system. Finally, in the *UI* folder, the script to execute a prototype of the whole system can be found. A video demo of the prototype can be found [here](https://youtu.be/MMHCoE6yonw).

