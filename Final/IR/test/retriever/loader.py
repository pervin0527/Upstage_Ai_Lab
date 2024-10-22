import os

from datetime import datetime
from langchain_chroma import Chroma
from langchain.storage import InMemoryByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever, SearchType

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
        persist_dir = f"./indexes/{timestamp}"
        os.makedirs(persist_dir, exist_ok=True)
        vector_store = Chroma(collection_name=cfg['dataset']['collection_name'], 
                              embedding_function=embedder, 
                              persist_directory=persist_dir)

        for key in dataset.keys():
            if key == "full_documents":
                continue
            vector_store.add_documents(dataset[key], persist_directory=persist_dir)

    else:
        vector_store = Chroma(embedding_function=embedder, persist_directory=cfg['retriever']['dense_retriever']['persist_dir'])

    # retriever = vector_store
    id_key = "docid"
    store = InMemoryByteStore()
    retriever = MultiVectorRetriever(vectorstore=vector_store, byte_store=store, id_key=id_key)
    retriever.search_type = SearchType.mmr ## mmr, similarity, similarity_score_threshold

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
        # dense = setup_dense_retriever(cfg, dataset).as_retriever(search_kwargs={"k": cfg['retriever']['top_k']})
        dense = setup_dense_retriever(cfg, dataset)

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