import numpy as np

from typing import List
from langchain_community.vectorstores.faiss import FAISS

def score_normalizer(val: float) -> float:
    # return 1 - 1 / (1 + np.exp(val))
    # return (1 + val) / 2
    return 1 / (1 + val)


def doc_vector_embedding(documents:List, embedding_model):
    faiss = FAISS.from_documents(documents, embedding_model).as_retriever()

    return faiss

def get_query_embedding(query, embedding_model):
    ## query 임베딩
    query_embedding = embedding_model.embed_query(query)
    
    return np.array(query_embedding)

def calculate_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def calculate_l2_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)