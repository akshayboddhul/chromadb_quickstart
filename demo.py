import csv
import chromadb
from chromadb.utils import embedding_functions

from fastapi import FastAPI

app = FastAPI()

collection_1 = []

def reading_input_file():
    with open('food_items.csv', 'r') as fp:
        rows = csv.reader(fp)

        documents = []
        metadata = []
        ids = []

        id = 1

        for i, row in enumerate(rows):
            if i == 0:
                continue

            documents.append(row[1])
            metadata.append({"item_id": row[0]})
            ids.append(str(id))
            id += 1
        return documents, metadata, ids

def chroma_initialize():
    client = chromadb.Client()
    embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="multi-qa-MiniLM-L6-cos-v1")

    collection = client.get_or_create_collection(name="food_items_collection", metadata = {"hnsw:space": "cosine"}, embedding_function=embedding_model)

    collection.add(
        documents = reading_input_file()[0],
        metadatas = reading_input_file()[1],
        ids = reading_input_file()[2]
    )
    collection_1[:] = collection
    return collection

def semantic_search(search_query):
    global collection_1
    if not collection_1:
        collection_1 = chroma_initialize()
    print(collection_1)
    results = collection_1.query(
        query_texts = [search_query],
        n_results = 5,
        include = ['documents','distances']
    )

    # return results
    final_results = {}
    for i in range(len(results.get('distances')[0])):
        final_results[results.get('documents')[0][i]] = results.get('distances')[0][i]
    
    final_results_by_score = sorted(final_results.items(), key=lambda x:x[1])
    return {key[0]: key[1] for key in final_results_by_score}
    # return final_results

# print(semantic_search('rice'))


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/search/{query}")
def search_func(query):
    return semantic_search(query)