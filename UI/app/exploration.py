from rdflib import Graph


'''
Script to apply the user defined query to the KG and return the results.
'''
def get_result(query):

    graph = Graph()
    graph.parse("./app/static/triples/KnowledgeBase.nt", format="nt") 

    prefix = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX exp: <http://localhost/8080/intentOntology#>
        """

    query = prefix + query

    results = graph.query(query)

    return results