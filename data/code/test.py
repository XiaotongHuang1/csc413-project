import openai
from pyopenie import OpenIE5
from pykeen.models import TransE
import numpy as np
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import json
import requests
import re

def query_yago_sparql(query):
    sparql = SPARQLWrapper("https://yago-knowledge.org/sparql/query")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    if query.strip().upper().startswith("ASK"):
        return results['boolean']
    else:
        return results['results']['bindings']


def load_kg_and_news():
    # Load news data from CSV
    news_data = pd.read_csv('train.csv')[1:100]

    # Convert headline data to list of JSON objects
    headlines_json = []
    for _, row in news_data.iterrows():
        headline_json = {
            'headline': row['headline'],
            'url': row['url'],
            'publisher': row['publisher'],
            'date': row['date'],
            'stock': row['stock']
        }
        headlines_json.append(headline_json)

    return headlines_json

# Assuming you have a function to load your news data
news_data = load_kg_and_news()

# OpenIE for event extraction
extractor = OpenIE5("http://localhost:8000")

def extract_event_tuples(text):
    if not isinstance(text, str):
        return []  # Return an empty list if the input is not a string
    response = requests.post('http://localhost:8000/getExtraction', data=text.encode('utf-8'))
    if response.status_code == 200 and response.text:
        # print(response.text)
        # Parse the JSON response
        extractions = json.loads(response.text)
        tuples = []
        for extraction in extractions:
            arg1 = extraction['extraction']['arg1']['text']
            rel = extraction['extraction']['rel']['text']
            for arg2 in extraction['extraction']['arg2s']:
                tuples.append((arg1, rel, arg2['text']))
        return tuples
    else:
        print(f"Error: Server responded with status code {response.status_code}")
        return []



# Entity linking and extension (simplified for illustration)
def link_and_extend_event_tuples(event_tuples):
    linked_tuples = []
    for s, p, o in event_tuples:
        # print('s:', s, 'p:', p, 'o:', o)
        # Format the subject and object for the SPARQL query
        formatted_s = s.replace(" ", "_").replace("(", "%28").replace(")", "%29")
        formatted_o = o.replace(" ", "_").replace("(", "%28").replace(")", "%29")

        # print('formatted_s:', formatted_s, 'formatted_o:', formatted_o)

        # Perform SPARQL query to check if subject exists in YAGO
        subject_query = f"""
        ASK WHERE {{
            <http://yago-knowledge.org/resource/{formatted_s}> ?p ?o .
        }}
        """
        subject_exists = query_yago_sparql(subject_query)

        # Perform SPARQL query to check if object exists in YAGO
        object_query = f"""
        ASK WHERE {{
            ?s ?p <http://yago-knowledge.org/resource/{formatted_o}> .
        }}
        """
        object_exists = query_yago_sparql(object_query)

        if subject_exists and object_exists:
            linked_tuples.append((s, p, o))
    return linked_tuples



# Knowledge-driven multi-channel concatenation
def get_event_embedding(event_tuple, kg, openai_model):
    s, p, o = event_tuple
    # KG embeddings (using TransE)
    kg_embedding = TransE(kg)
    ves_l = kg_embedding.get_embedding(s) if s in kg else np.zeros(kg_embedding.embedding_dim)
    veo_l = kg_embedding.get_embedding(o) if o in kg else np.zeros(kg_embedding.embedding_dim)
    vr_p_l = kg_embedding.get_embedding(p) if p in kg else np.zeros(kg_embedding.embedding_dim)
    
    # Word vectors (using OpenAI embeddings)
    vesw = openai.Embedding.create(input=s, model=openai_model)['data'][0]['embedding']
    veow = openai.Embedding.create(input=o, model=openai_model)['data'][0]['embedding']
    vrpw = openai.Embedding.create(input=p, model=openai_model)['data'][0]['embedding']
    
    # Concatenate
    event_embedding = np.concatenate([ves_l, vr_p_l, veo_l, vesw, vrpw, veow])
    return event_embedding

# Example usage
for news in news_data:
    if 'headline' in news and isinstance(news['headline'], str):
        event_tuples = extract_event_tuples(news['headline'])
        # print(event_tuples)
        linked_tuples = link_and_extend_event_tuples(event_tuples)
        print(linked_tuples)
        for event_tuple in linked_tuples:
            pass
            # embedding = get_event_embedding(event_tuple, kg, 'text-embedding-ada-002')
            # Do something with the embedding
    else:
        print("Invalid or missing headline in news data.")