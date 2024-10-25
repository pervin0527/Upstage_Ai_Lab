from langchain_community.vectorstores.faiss import FAISS
from sparse_retriever.kiwi_bm25 import KiwiBM25Retriever

def retrieval_debug(query, retrieval):
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