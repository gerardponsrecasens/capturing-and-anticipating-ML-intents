# Capturing and anticipating user intents in complex analytics workflows

This repository contains the main scripts for the Task 4.3 of the **ExtremeXP** project, which focuses on two key functionalities of the ExtremeXP framework: 
- Capturing of user intents, and also user preferences, and constraints. 
- Mapping the intents to possible complex analytics workflows. 


To achieve it, different steps have been followed: 
1. An ontology has been **designed** to capture all the requiered concepts and relationships related to the user intents and also to capture the machine learning workflow associated to the intent, so that the generated knowledge graph can be used in a future step to derived valid workflows.
2. The ontology has been **populated** with ML workflows from various sources.
3. Inference algorithms have been applied to the populated knowledge graph, so that it can be used to find **relevant workflows** (e.g., serve as a reccomender system).



![workflow_22_03](https://user-images.githubusercontent.com/95172600/227177301-b1ebdca0-13df-4411-9576-739da2e3bef8.png)


The repository is organized as follows: in the *Ontology* folder, the ontology .owl/.nt files can be found, alongside with the scripts used to generate them. Then, in the ontology *Population* folder the scripts for creating the instances from the different sources can be found. Finally, the *Recommender* folder contains the scripts and demos for the recommender system.
