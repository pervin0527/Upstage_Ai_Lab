import os
import pickle
import numpy as np

from tqdm import tqdm
from datetime import datetime
from langchain.schema import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_teddynote.retrievers import EnsembleRetriever, EnsembleMethod

from retriever.sparse import load_sparse_retriever
from retriever.dense import load_dense_retriever

def setup_dense_retriever(cfg, dataset):
    embedder = load_dense_retriever(method=cfg['retriever']['dense_retriever']['method'], 
                                    model_name=cfg['retriever']['dense_retriever']['model'], 
                                    model_kwargs=cfg['retriever']['dense_retriever']['model_kwargs'], 
                                    encode_kwargs=cfg['retriever']['dense_retriever']['encode_kwargs'])

    if cfg['retriever']['dense_retriever']['persist_dir'] is None or not os.path.exists(cfg['retriever']['dense_retriever']['persist_dir']):
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        persist_dir = f"./indexes/{cfg['retriever']['dense_retriever']['method']}/{timestamp}"
        os.makedirs(persist_dir, exist_ok=True)

        documents = []
        for key in dataset.keys():
            if key in ["full_documents", "en_queries", "en_documents"]:
                continue
            documents.extend(dataset[key])
            
        vector_store = FAISS.from_documents(documents, embedder)
        vector_store.save_local(persist_dir)
        print(f"  Vector Store Saved {persist_dir}")
        retriever = vector_store

    else:
        retriever = FAISS.load_local(cfg['retriever']['dense_retriever']['persist_dir'], embedder, allow_dangerous_deserialization=True)

    print(f"  Total Documents in VectorStore: {retriever.index.ntotal}")
    return retriever


def load_retriever(cfg, dataset):
    print("=" * 70)
    print(f"{cfg['retriever']['method']} Retriever Loading")

    if cfg['retriever']['method'] == "sparse":
        retriever = load_sparse_retriever(dataset, cfg['dataset']['language'])
        retriever.k = cfg['retriever']['top_k']

    elif cfg['retriever']['method'] == "dense":
        retriever = setup_dense_retriever(cfg, dataset)

    elif cfg['retriever']['method'] == "ensemble":
        print("  Sparse retriever Loading.")
        sparse = load_sparse_retriever(dataset, cfg['dataset']['language'])
        sparse.k = cfg['retriever']['top_k']
        
        print("  Dense retriever Loading.")
        dense = setup_dense_retriever(cfg, dataset).as_retriever(search_type="mmr", search_kwargs={"k": cfg['retriever']['top_k']})

        if cfg['retriever']['ensemble_retriever']['method'] == "rrf":
            retriever = EnsembleRetriever(retrievers=[sparse, dense], method=EnsembleMethod.RRF)

        elif cfg['retriever']['ensemble_retriever']['method'] == "cc":
            retriever = EnsembleRetriever(retrievers=[sparse, dense], method=EnsembleMethod.CC)

    return retriever


def load_query_ensemblers(cfg):
    ensemble_embedders = []
    for info in cfg['query_ensemble']['models']:
        method, model_name = info['type'], info['name']
        embedder = load_dense_retriever(method=method, 
                                        model_name=model_name, 
                                        model_kwargs=cfg['retriever']['dense_retriever']['model_kwargs'], 
                                        encode_kwargs=cfg['retriever']['dense_retriever']['encode_kwargs'])
        ensemble_embedders.append(embedder)

    return ensemble_embedders