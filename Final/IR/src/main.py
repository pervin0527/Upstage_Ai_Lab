import os
import warnings
import huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning)

import faiss

from openai import OpenAI

from langchain_community.vectorstores.faiss import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from data.data import load_document, chunk_documents

from dense_retriever.model import load_hf_encoder
from dense_retriever.doc_processor import score_normalizer

from sparse_retriever.kiwi_bm25 import KiwiBM25Retriever

from dotenv import load_dotenv
load_dotenv("../keys.env")

openai_api_key = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = openai_api_key

hf_token = os.getenv("HF_TOKEN")
huggingface_hub.login(hf_token)

def debug(query, retrieval):
    if isinstance(retrieval, FAISS):
        # results = retrieval.similarity_search_with_score(query, k=3)
        results = retrieval.similarity_search_with_relevance_scores(query, k=3)

        print(f"\nQuery: {query}\n{'='*50}")
        for i, (res, score) in enumerate(results, 1):
            # L2 거리를 유사도로 변환
            print(f"\nResult {i}:")
            print(f"Score: {score}")
            print(f"DocID: {res.metadata.get('docid', 'N/A')}")
            print(f"Content: {res.page_content}")  # 첫 200자를 출력
            print(f"Metadata: {res.metadata}")
            print(f"{'-'*50}")
    
    elif isinstance(retrieval, KiwiBM25Retriever):
        results = retrieval.search_with_score(query, k=3)

        print(f"\nQuery: {query}\n{'='*50}")
        for i, result in enumerate(results, 1):
            doc = result  # result 자체가 Document 객체
            score = doc.metadata.get('score', 'N/A')  # score는 metadata에 포함됨
            print(f"\nResult {i}:")
            print(f"Score: {score}")  # metadata에서 score를 출력
            print(f"DocID: {doc.metadata.get('docid', 'N/A')}")
            print(f"Content: {doc.page_content}")  # 첫 200자를 출력
            print(f"Metadata: {doc.metadata}")
            print(f"{'-'*50}")


class Args:
    doc_method = "dense"

    chunk_size=100
    chunk_overlap=5

    ## sparse
    tokenizer = "kiwi"

    ## dense
    model_name = "intfloat/multilingual-e5-large-instruct" ## "jhgan/ko-sroberta-multitask"
    model_kwargs = {"device": "cuda:0"}
    encode_kwargs = {"normalize_embeddings": False,
                     "clean_up_tokenization_spaces": True}

def main(args:Args):
    documents = load_document(path="../dataset/processed_documents.jsonl")
    # documents = chunk_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    if args.doc_method == "dense":
        encoder = load_hf_encoder(args.model_name, args.model_kwargs, args.encode_kwargs)
        index = faiss.IndexFlatL2(len(encoder.embed_query("파이썬")))
        # index = faiss.IndexFlatIP(len(encoder.embed_query("파이썬")))

        vector_store = FAISS(
            embedding_function=encoder,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            relevance_score_fn=score_normalizer
        )
        vector_store.add_documents(documents=documents)
        retrieval = vector_store

    elif args.doc_method == "sparse":
        retrieval = KiwiBM25Retriever.from_documents(documents)

    query = "평균속도는 어떻게 계산하나요?"
    debug(query, retrieval)

    client = OpenAI()
    model = "gpt-4o"
    
if __name__ == "__main__":
    main(Args)