import numpy as np

from typing import List
from Final.IR.src.sparse_retriever.kiwi_bm25 import KiwiBM25Retriever

def doc_kiwi_bm25(documents):
    bm25 = KiwiBM25Retriever.from_documents(documents)
    
    return bm25