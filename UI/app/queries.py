from rdflib import Graph


############# Define the flow for anticipation queries. #####################

# INTENT ANTICIPATION

# First of all we will check if the user has previously used the dataset.


def get_intent(user,dataset):

    graph = Graph()
    graph.parse("./app/static/triples/KnowledgeBase.nt", format="nt") 

    found = False

    query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX ml: <http://localhost/8080/intentOntology#>

    SELECT ?intent (COUNT(?intent) AS ?count)
    WHERE {{
        ml:{user} ml:runs ?workflow.
        ?workflow ml:hasInput ml:{dataset}.
        ?workflow ml:achieves ?task.
        ?task ml:hasIntent ?intent 
    }}
    GROUP BY ?intent
    ORDER BY DESC(?count)
    LIMIT 1
    """

    # Execute the SPARQL query
    results = graph.query(query)

    # Iterate over the query results and print them
    for row in results:
        intent = row["intent"]
        found = True
    
    if not found:

        # If the user has not used the dataset, let's look for other user's:

        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX ml: <http://localhost/8080/intentOntology#>

        SELECT ?intent (COUNT(?intent) AS ?count)
        WHERE {{
            ?workflow ml:hasInput ml:{dataset}.
            ?workflow ml:achieves ?task.
            ?task ml:hasIntent ?intent 
        }}
        GROUP BY ?intent
        ORDER BY DESC(?count)
        LIMIT 1
        """

        # Execute the SPARQL query
        results = graph.query(query)

        # Iterate over the query results and print them
        for row in results:
            intent = row["intent"]
            found = True
    
    if not found:

        # If the dataset has not been used before, let's look for the most used intents by the user

        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX ml: <http://localhost/8080/intentOntology#>

        SELECT ?intent (COUNT(?intent) AS ?count)
        WHERE {{
            ml:{user} ml:runs ?workflow.
            ?workflow ml:achieves ?task.
            ?task ml:hasIntent ?intent 
        }}
        GROUP BY ?intent
        ORDER BY DESC(?count)
        LIMIT 1
        """

        # Execute the SPARQL query
        results = graph.query(query)

        # Iterate over the query results and print them
        for row in results:
            intent = row["intent"]
            found = True
    
    if not found:

        # If the user has never used the tool, give the most used intent

        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX ml: <http://localhost/8080/intentOntology#>

        SELECT ?intent (COUNT(?intent) AS ?count)
        WHERE {{
            ?task ml:hasIntent ?intent 
        }}
        GROUP BY ?intent
        ORDER BY DESC(?count)
        LIMIT 1
        """

        # Execute the SPARQL query
        results = graph.query(query)

        # Iterate over the query results and print them
        for row in results:
            intent = row["intent"]
            found = True

    return intent



def get_metric(user,dataset,intent):

    graph = Graph()
    graph.parse("./app/static/triples/KnowledgeBase.nt", format="nt") 

    found = False

    query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX ml: <http://localhost/8080/intentOntology#>

    SELECT ?metric (COUNT(?metric) AS ?count)
    WHERE {{
        ml:{user} ml:runs ?workflow.
        ?workflow ml:hasInput ml:{dataset}.
        ?workflow ml:achieves ?task.
        ?task ml:hasRequirement ?eval.
        ?eval ml:onMetric ?metric 
    }}
    GROUP BY ?metric
    ORDER BY DESC(?count)
    LIMIT 1
    """

    # Execute the SPARQL query
    results = graph.query(query)

    # Iterate over the query results and print them
    for row in results:
        metric = row["metric"]
        found = True
    
    if not found:

        # If the user has not used the dataset, let's look for other user's:

        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX ml: <http://localhost/8080/intentOntology#>

        SELECT ?metric (COUNT(?metric) AS ?count)
        WHERE {{
            ?workflow ml:hasInput ml:{dataset}.
            ?workflow ml:achieves ?task.
            ?task ml:hasRequirement ?eval.
            ?eval ml:onMetric ?metric 
        }}
        GROUP BY ?metric
        ORDER BY DESC(?count)
        LIMIT 1
        """

        # Execute the SPARQL query
        results = graph.query(query)

        # Iterate over the query results and print them
        for row in results:
            metric = row["metric"]
            found = True
    
    if not found:

        # If the dataset has not been used before, let's look for the most used metric for by the user for the intent

        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX ml: <http://localhost/8080/intentOntology#>

        SELECT ?metric (COUNT(?metric) AS ?count)
        WHERE {{
            ml:{user} ml:runs ?workflow.
            ?workflow ml:achieves ?task.
            ?task ml:hasIntent ml:{intent}.
            ?task ml:hasRequirement ?eval.
            ?eval ml:onMetric ?metric 
        }}
        GROUP BY ?metric
        ORDER BY DESC(?count)
        LIMIT 1
        """

        # Execute the SPARQL query
        results = graph.query(query)

        # Iterate over the query results and print them
        for row in results:
            metric = row["metric"]
            found = True
    
    if not found:

        # If the user has never used the tool, give the most used metric for the intent

        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX ml: <http://localhost/8080/intentOntology#>

        SELECT ?metric (COUNT(?metric) AS ?count)
        WHERE {{
            ?task ml:hasIntent ml:{intent}.
            ?task ml:hasRequirement ?eval.
            ?eval ml:onMetric ?metric 
        }}
        GROUP BY ?metric
        ORDER BY DESC(?count)
        LIMIT 1
        """

        # Execute the SPARQL query
        results = graph.query(query)

        # Iterate over the query results and print them
        for row in results:
            metric = row["metric"]
            found = True

    return metric


def get_preprocessing(user,dataset,intent):

    graph = Graph()
    graph.parse("./app/static/triples/KnowledgeBase.nt", format="nt") 

    found = False

    query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX ml: <http://localhost/8080/intentOntology#>

    SELECT (COUNT(DISTINCT ?task) AS ?constraintTaskCount)
    WHERE {{
        ml:{user} ml:runs ?workflow.
        ?workflow ml:hasInput ml:{dataset}.
        ?workflow ml:achieves ?task.
        ?task ml:hasConstraint ml:ConstraintNoPreprocessing
    }}
    """
    query_aux = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX ml: <http://localhost/8080/intentOntology#>

    SELECT (COUNT(DISTINCT ?task) AS ?taskCount)
    WHERE {{
        ml:{user} ml:runs ?workflow.
        ?workflow ml:hasInput ml:{dataset}.
        ?workflow ml:achieves ?task.
    }}
    """

    results = graph.query(query)
    results_aux = graph.query(query_aux)

    for row in results_aux:
        total_tasks = row['taskCount']
        found = int(row['taskCount']) != 0
    for row in results:
        constraint_task = row['constraintTaskCount']
    
    if found:
        if int(constraint_task)/int(total_tasks)>0.5:
            preprocessing = True
        else:
            preprocessing = False
    
    if not found:
        # Use all users with the dataset:
        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX ml: <http://localhost/8080/intentOntology#>

        SELECT (COUNT(DISTINCT ?task) AS ?constraintTaskCount)
        WHERE {{
            ?workflow ml:hasInput ml:{dataset}.
            ?workflow ml:achieves ?task.
            ?task ml:hasConstraint ml:ConstraintNoPreprocessing
        }}
        """
        query_aux = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX ml: <http://localhost/8080/intentOntology#>

        SELECT (COUNT(DISTINCT ?task) AS ?taskCount)
        WHERE {{
            ?workflow ml:hasInput ml:{dataset}.
            ?workflow ml:achieves ?task.
        }}
        """

        results = graph.query(query)
        results_aux = graph.query(query_aux)

        for row in results_aux:
            total_tasks = row['taskCount']
            found = int(row['taskCount']) != 0

        for row in results:
            constraint_task = row['constraintTaskCount']
        
        if found:
            if int(constraint_task)/int(total_tasks)>0.5:
                preprocessing = True
            else:
                preprocessing = False

    if not found:
        # User with the Intent
        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX ml: <http://localhost/8080/intentOntology#>

        SELECT (COUNT(DISTINCT ?task) AS ?constraintTaskCount)
        WHERE {{
            ml:{user} ml:runs ?workflow.
            ?workflow ml:achieves ?task.
            ?task ml:hasIntent ml:{intent}
            ?task ml:hasConstraint ml:ConstraintNoPreprocessing
        }}
        """
        query_aux = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX ml: <http://localhost/8080/intentOntology#>

        SELECT (COUNT(DISTINCT ?task) AS ?taskCount)
        WHERE {{
            ml:{user} ml:runs ?workflow.
            ?workflow ml:achieves ?task.
            ?task ml:hasIntent ml:{intent}
        }}
        """

        results = graph.query(query)
        results_aux = graph.query(query_aux)

        for row in results_aux:
            total_tasks = row['taskCount']
            found = int(row['taskCount']) != 0

        for row in results:
            constraint_task = row['constraintTaskCount']
        
        if found:
            if int(constraint_task)/int(total_tasks)>0.5:
                preprocessing = True
            else:
                preprocessing = False
    
    if not found:
        # All users with intent
        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX ml: <http://localhost/8080/intentOntology#>

        SELECT (COUNT(DISTINCT ?task) AS ?constraintTaskCount)
        WHERE {{
            ?task ml:hasIntent ml:{intent}
            ?task ml:hasConstraint ml:ConstraintNoPreprocessing
        }}
        """
        query_aux = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX ml: <http://localhost/8080/intentOntology#>

        SELECT (COUNT(DISTINCT ?task) AS ?taskCount)
        WHERE {{
            ?task ml:hasIntent ml:{intent}
        }}
        """

        results = graph.query(query)
        results_aux = graph.query(query_aux)

        for row in results_aux:
            total_tasks = row['taskCount']
            found = int(row['taskCount']) != 0

        for row in results:
            constraint_task = row['constraintTaskCount']
        
        if found:
            if int(constraint_task)/int(total_tasks)>0.5:
                preprocessing = True
            else:
                preprocessing = False

    return preprocessing

def get_algorithm(user,dataset,intent):
    
    graph = Graph()
    graph.parse("./app/static/triples/KnowledgeBase.nt", format="nt") 

    found = False

    query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX ml: <http://localhost/8080/intentOntology#>

    SELECT ?algorithm (COUNT(?algorithm) AS ?count)
    WHERE {{
        ml:{user} ml:runs ?workflow.
        ?workflow ml:hasInput ml:{dataset}.
        ?workflow ml:achieves ?task.
        ?task ml:hasConstraint ?constraint.
        ?constraint rdf:type ml:ConstraintAlgorithm.
        ?constraint ml:on ?algorithm 
    }}
    GROUP BY ?algorithm
    ORDER BY DESC(?count)
    LIMIT 1
    """

    # Execute the SPARQL query
    results = graph.query(query)

    # Iterate over the query results and print them
    for row in results:
        algorithm = row["algorithm"]
        found = True
    
    if not found:
        # Look for other users usage of the Datset
        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX ml: <http://localhost/8080/intentOntology#>

        SELECT ?algorithm (COUNT(?algorithm) AS ?count)
        WHERE {{
            ?workflow ml:hasInput ml:{dataset}.
            ?workflow ml:achieves ?task.
            ?task ml:hasConstraint ?constraint.
            ?constraint rdf:type ml:ConstraintAlgorithm.
            ?constraint ml:on ?algorithm 
        }}
        GROUP BY ?algorithm
        ORDER BY DESC(?count)
        LIMIT 1
        """

        # Execute the SPARQL query
        results = graph.query(query)

        # Iterate over the query results and print them
        for row in results:
            algorithm = row["algorithm"]
            found = True
        
    if not found:
        # Look for the user's usages of the same intent
        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX ml: <http://localhost/8080/intentOntology#>

        SELECT ?algorithm (COUNT(?algorithm) AS ?count)
        WHERE {{
            ml:{user} ml:runs ?workflow.
            ?workflow ml:achieves ?task.
            ?task ml:hasIntent ml:{intent}.
            ?task ml:hasConstraint ?constraint.
            ?constraint rdf:type ml:ConstraintAlgorithm.
            ?constraint ml:on ?algorithm 
        }}
        GROUP BY ?algorithm
        ORDER BY DESC(?count)
        LIMIT 1
        """

        # Execute the SPARQL query
        results = graph.query(query)

        # Iterate over the query results and print them
        for row in results:
            algorithm = row["algorithm"]
            found = True
    
    if not found:
        # All user usages of the Intent
        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX ml: <http://localhost/8080/intentOntology#>

        SELECT ?algorithm (COUNT(?algorithm) AS ?count)
        WHERE {{

            ?task ml:hasIntent ml:{intent}.
            ?task ml:hasConstraint ?constraint.
            ?constraint rdf:type ml:ConstraintAlgorithm.
            ?constraint ml:on ?algorithm 
        }}
        GROUP BY ?algorithm
        ORDER BY DESC(?count)
        LIMIT 1
        """

        # Execute the SPARQL query
        results = graph.query(query)

        # Iterate over the query results and print them
        for row in results:
            algorithm = row["algorithm"]
            found = True
    
    return algorithm

def get_preprocessing_algorithm(user,dataset,intent):

    graph = Graph()
    graph.parse("./app/static/triples/KnowledgeBase.nt", format="nt") 

    found = False

    query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX ml: <http://localhost/8080/intentOntology#>

    SELECT ?algorithm (COUNT(?algorithm) AS ?count)
    WHERE {{
        ml:{user} ml:runs ?workflow.
        ?workflow ml:hasInput ml:{dataset}.
        ?workflow ml:achieves ?task.
        ?task ml:hasConstraint ?constraint.
        ?constraint rdf:type ml:ConstraintPreprocessingAlgorithm.
        ?constraint ml:on ?algorithm 
    }}
    GROUP BY ?algorithm
    ORDER BY DESC(?count)
    LIMIT 1
    """

    # Execute the SPARQL query
    results = graph.query(query)

    # Iterate over the query results and print them
    for row in results:
        algorithm = row["algorithm"]
        found = True
    
    if not found:
        # Look for other users usage of the Datset
        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX ml: <http://localhost/8080/intentOntology#>

        SELECT ?algorithm (COUNT(?algorithm) AS ?count)
        WHERE {{
            ?workflow ml:hasInput ml:{dataset}.
            ?workflow ml:achieves ?task.
            ?task ml:hasConstraint ?constraint.
            ?constraint rdf:type ml:ConstraintPreprocessingAlgorithm.
            ?constraint ml:on ?algorithm 
        }}
        GROUP BY ?algorithm
        ORDER BY DESC(?count)
        LIMIT 1
        """

        # Execute the SPARQL query
        results = graph.query(query)

        # Iterate over the query results and print them
        for row in results:
            algorithm = row["algorithm"]
            found = True
        
    if not found:
        # Look for the user's usages of the same intent
        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX ml: <http://localhost/8080/intentOntology#>

        SELECT ?algorithm (COUNT(?algorithm) AS ?count)
        WHERE {{
            ml:{user} ml:runs ?workflow.
            ?workflow ml:achieves ?task.
            ?task ml:hasIntent ml:{intent}.
            ?task ml:hasConstraint ?constraint.
            ?constraint rdf:type ml:ConstraintPreprocessingAlgorithm.
            ?constraint ml:on ?algorithm 
        }}
        GROUP BY ?algorithm
        ORDER BY DESC(?count)
        LIMIT 1
        """

        # Execute the SPARQL query
        results = graph.query(query)

        # Iterate over the query results and print them
        for row in results:
            algorithm = row["algorithm"]
            found = True
    
    if not found:
        # All user usages of the Intent
        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX ml: <http://localhost/8080/intentOntology#>

        SELECT ?algorithm (COUNT(?algorithm) AS ?count)
        WHERE {{

            ?task ml:hasIntent ml:{intent}.
            ?task ml:hasConstraint ?constraint.
            ?constraint rdf:type ml:ConstraintPreprocessingAlgorithm.
            ?constraint ml:on ?algorithm 
        }}
        GROUP BY ?algorithm
        ORDER BY DESC(?count)
        LIMIT 1
        """

        # Execute the SPARQL query
        results = graph.query(query)

        # Iterate over the query results and print them
        for row in results:
            algorithm = row["algorithm"]
            found = True
    
    return algorithm