## /home/pervinco/miniconda3/envs/ir-project/lib/python3.10>site_packages/langchain_community/retrievers/konlpy_bm25.py
from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers.konlpy_bm25 import OktBM25Retriever

def load_sparse_retriever(dataset, language="ko"):
    documents = []
    for key in dataset.keys():
        if key in ["full_documents", "en_queries", "en_documents"]:
            continue
        documents.extend(dataset[key])

    if language == "ko":
        from konlpy.tag import Okt
        okt = Okt()
        
        def tokenize(text):
            tokens = okt.morphs(text)
            return tokens
        
        retriever = OktBM25Retriever.from_documents(documents)
        
    elif language == "en":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('voyageai/voyage-3')

        def tokenize(text):
            tokens = tokenizer.tokenize(text)
            return tokens

        retriever = BM25Retriever.from_documents(documents, tokenizer=tokenize)
    
    return retriever