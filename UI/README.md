# USER INTERFACT

This folder contains the User Interface created for the Task 4.3 of the **ExtremeXP** project. Its purpose is to mock the user experience and visualize the recommnendation/knowledge graph task, hence it is a simple version that should be completely changed to incorporate the different parts of the project created in other Tasks. The usage is as follows:

1. When the user connects, it is asked his/her name and the **input data**. The recommender should help the user with that, hence once he/she starts filling in some information, the other slots should display automatically some recommendations.
2. Once the user has filled in the input, the system generates a **Workflow** for that. For now, the system uses the library *HyperOpt*, but it should be replaced by the pipleine creation system designed in the second part of the Task 4.3. 
3. The system displays the **result** (e.g., numerical value, visualization...), and allows the user to give Feedback about the output (i.e., comment and rating.)

4. The input, the workflow created and the feedback are translated into **RDF triples**, so that they can be used for enhancing the recommendations.
