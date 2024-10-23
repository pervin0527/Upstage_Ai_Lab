import os
import json
import yaml
import voyageai
import numpy as np

from datetime import datetime
from scipy.spatial.distance import cosine

from langchain_openai import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores.faiss import FAISS
from langchain_teddynote.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever

from search.rankgpt.ranker import reranking
from search.rankgpt.custom import custom_reranker
from retriever.loader import load_query_ensemblers

def search_with_scores(retriever, query, k=10):
    processed_query = retriever.preprocess_func(query)
    doc_scores = retriever.vectorizer.get_scores(processed_query)
    
    top_n_indices = np.argsort(doc_scores)[::-1][:k]
    results = [(retriever.docs[i], doc_scores[i]) for i in top_n_indices]

    return results


def retrieve_documents(query, retriever, k=10):
    if isinstance(retriever, FAISS):
        return retriever.similarity_search_with_relevance_scores(query, k=k)
    
    elif isinstance(retriever, BM25Retriever):
        return [(doc, score) for doc, score in search_with_scores(retriever, query, k=k)]
    
    elif isinstance(retriever, (EnsembleRetriever, MultiQueryRetriever, ContextualCompressionRetriever)):
        results = retriever.invoke(query)
        return [(result, result.metadata['score']) for result in sorted(results, key=lambda x: x.metadata['score'], reverse=True)[:k]]
    
    else:
        raise ValueError("Unknown retriever type")
    

def filter_search_results(search_result, score_threshold):
    # score_threshold 이상인 결과만 필터링
    filtered_by_score = [
        (doc, text) for doc, text in search_result 
        if doc.metadata['score'] >= score_threshold
    ]
    
    # docid별로 최고 점수를 가진 결과를 찾기 위한 임시 딕셔너리
    docid_to_best_result = {}
    for doc, text in filtered_by_score:
        docid = doc.metadata['docid']
        score = doc.metadata['score']
        
        if docid not in docid_to_best_result or score > docid_to_best_result[docid][0].metadata['score']:
            docid_to_best_result[docid] = (doc, text)
    
    # 최종 결과를 점수 순으로 정렬
    final_results = sorted(
        docid_to_best_result.values(), 
        key=lambda x: x[0].metadata['score'], 
        reverse=True
    )
    
    return final_results
    

def query_ensembling(cfg, search_results, ensemble_encoders=None, query_embeddings=None):
    docid_scores = {}
    
    for doc, _ in search_results:
        if query_embeddings:
            combined_similarity = sum(
                weight * (1 - cosine(query_embedding, doc.metadata.get(f"embedding_{cfg['query_ensemble']['models'][idx]['name']}", ensemble_encoders[idx].embed_query(doc.page_content))))
                for idx, (query_embedding, weight) in enumerate(query_embeddings)
            )

        # 문서 필터링 및 저장
        if combined_similarity >= cfg['retriever']['score_threshold']:
            docid = doc.metadata.get('docid')
            docid_scores[docid] = {'doc': doc, 'score': combined_similarity}

    return sorted(docid_scores.values(), key=lambda x: x['score'], reverse=True)


def rag(cfg, standalone_query, retriever, dataset, full_documents, ensemble_encoders=None):
    response = {"standalone_query": standalone_query or "", "topk": [], "references": [], "answer": ""}
    
    if standalone_query is None:
        print("  답변을 생성할 수 없습니다.\n")
        response["answer"] = "질문이 과학 상식에 해당하지 않습니다."
        return response
    
    search_result = retrieve_documents(standalone_query, retriever, cfg['retriever']['top_k'])
    filtered_search_result = filter_search_results(search_result, cfg['retriever']['score_threshold'])

    candidates1 = [(doc.metadata['docid'], doc.metadata['score']) for doc, _ in search_result]
    candidates2 = [(doc.metadata['docid'], doc.metadata['score']) for doc, _ in filtered_search_result]
    print(f"candidates1 : {candidates1}")
    print(f"candidates2 : {candidates2}")
    
    if cfg['query_ensemble']['apply']:
        print("  query ensembling...")
        if len(cfg['query_ensemble']['models']) != len(cfg['query_ensemble']['weights']):
            raise ValueError("ensemble_models와 ensemble_weights의 길이가 동일해야 합니다.")

        query_embeddings = [(encoder.embed_query(standalone_query), cfg['query_ensemble']['weights'][idx]) for idx, encoder in enumerate(ensemble_encoders)]
        final_results = query_ensembling(cfg, filtered_search_result, ensemble_encoders, query_embeddings)

        candidates3 = [(result['doc'].metadata['docid'], result['score']) for result in final_results]
        print(f"candidates3 : {candidates3}")
    else:
        final_results = [
            {'doc': doc, 'score': score} 
            for doc, score in filtered_search_result
        ]
    
    retrieved_context = []
    if cfg['reranking']:
        print("  reranking...")
        top_docs = [result['doc'].page_content for result in final_results]

        vo = voyageai.Client()
        voyage_reranking = vo.rerank(standalone_query, top_docs, model="rerank-2", top_k=3)

        reranked_results = []
        for r in voyage_reranking.results:
            doc = next((result for result in final_results if result['doc'].page_content == r.document), None)
            if doc:
                reranked_results.append({
                    "doc": doc['doc'],
                    "score": r.relevance_score
                })

        for result in reranked_results:
            doc = result['doc']
            retrieved_context.append(doc.page_content)
            response["topk"].append(doc.metadata.get('docid'))
            response["references"].append({
                "docid": doc.metadata.get('docid'),
                "score": result['score'],
                "content": doc.page_content
            })

        ## top_docs를 [1]: 문서1\n, [2]: 문서2\n 형태로 변환
        # top_docs = [result['doc'].metadata['docid'] for result in final_results]
        # docs = [{'content': result['doc'].page_content, 'metadata': result['doc'].metadata} for result in final_results]
        # initial_scores = [result['score'] for result in final_results]

        # reranked_doc_indices, reranked_scores = reranking(standalone_query, top_docs, docs, top_k=3, initial_scores=initial_scores)
        # reranked_results = [
        #     {"doc": final_results[int(idx)]['doc'], "score": reranked_scores[i]}
        #     for i, idx in enumerate(reranked_doc_indices)
        # ]
        # final_results = reranked_results[:3]  # 상위 3개 문서만 선택

    if cfg['custom_reranking']:
        final_results = custom_reranker(standalone_query, final_results)
        
    for result in final_results[:3]:
        docid = result['doc'].metadata['docid']
        result['doc'].page_content = full_documents[docid].page_content

        doc = result['doc']
        retrieved_context.append(doc.page_content)
        response["topk"].append(doc.metadata.get('docid'))
        response["references"].append({
            "docid": doc.metadata.get('docid'),
            "score": result['score'],
            "content": doc.page_content
        })
    
    print("\n  검색결과")
    for idx, ref in enumerate(response["references"], start=1):
        print(f"  Rank{idx}")
        print("  ","-" * 70)
        print(f"    - DocID: {ref['docid']}")
        print(f"    - Score: {ref['score']:.4f}")
        print(f"    - Content: {ref['content']}\n")
    
    return response


def start_rag(cfg, retriever, dataset):
    if cfg['query_ensemble']['apply']:
        ensemble_models = load_query_ensemblers(cfg)
    else:
        ensemble_models = None

    if cfg['multiple_query']:
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)

    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_path = f"{cfg['output']['path']}/{timestamp}"
    os.makedirs(output_path, exist_ok=True)

    cfg_output_path = f"{output_path}/config.yaml"
    with open(cfg_output_path, 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False, allow_unicode=True)

    full_doc_map = {doc.metadata['docid']: doc for doc in dataset['full_documents']}
    with open(cfg['dataset']['eval_file']) as f, open(f"{output_path}/{cfg['output']['name']}", "w") as of:
        idx = 0
        for line in f:
            data = json.loads(line)

            id = data['eval_id']
            query = data['query']
            print("=" * 70)
            print(f'Test {idx:>04}')
            print("=" * 70)
            print(f"  Question: {query}")
            print(f"  Standalone_Query : {query}")

            if id in [276, 261, 283, 32, 94, 90, 220,  245, 229, 247, 67, 57, 2, 227, 301, 222, 83, 64, 103, 218]:
                query = None

            response = rag(cfg, query, retriever, dataset, full_doc_map, ensemble_models)

            output = {"eval_id": data["eval_id"], "standalone_query": response["standalone_query"], "topk": response["topk"], "answer": response["answer"], "references": response["references"]}
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            idx += 1
            print()            
