import json
import numpy as np

from openai import OpenAI
from scipy.spatial.distance import cosine

from langchain.retrievers import EnsembleRetriever
from langchain_community.chat_models import ChatOllama
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores.faiss import FAISS

from search.query_processor import query_refinement, query_expansion
from search.ollama_utils import ollama_standalone_query, ollama_domain_check, ollama_translate_query, ollama_query_expansion
from dense_retriever.model import load_hf_encoder, load_upstage_encoder, load_hf_reranker, load_voyage_encoder, load_llm_reranker


def search_with_scores(retriever, query, k=10):
    processed_query = retriever.preprocess_func(query)
    doc_scores = retriever.vectorizer.get_scores(processed_query)
    
    top_n_indices = np.argsort(doc_scores)[::-1][:k]
    results = [(retriever.docs[i], doc_scores[i]) for i in top_n_indices]

    return results


def get_chat_history(eval_data):
    str_chat_history = []
    for msg in eval_data['msg']:
        str_chat_history.append(f"{msg['role']} : {msg['content']}")

    return '\n'.join(str_chat_history)


def ollama_answer_question(args, standalone_query, retriever, compression_retriever=None, ensemble_encoders=None):
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    if standalone_query is not None:
        response['standalone_query'] = standalone_query

        ## Query Ensemble
        if args.query_ensemble:
            print("=" * 30)
            print("query ensembling...")
            if len(args.ensemble_models) != len(args.ensemble_weights):
                raise ValueError("ensemble_models와 ensemble_weights의 길이가 동일해야 합니다.")
    
            query_embeddings = []
            for idx, encoder in enumerate(ensemble_encoders):
                model_type = args.ensemble_models[idx]['type']
                query_embedding = encoder.embed_query(standalone_query)
                query_embeddings.append((query_embedding, args.ensemble_weights[idx]))
    
            combined_scores = []

            # 검색 결과 가져오기
            if isinstance(retriever, FAISS):
                search_result = retriever.similarity_search_with_relevance_scores(standalone_query, k=5)

            elif isinstance(retriever, BM25Retriever):
                results = search_with_scores(retriever, standalone_query, k=5)
                search_result = [(doc, score) for doc, score in results]

            elif isinstance(retriever, EnsembleRetriever):
                results = retriever.invoke(standalone_query)
                sorted_results = sorted(results, key=lambda x: x.metadata['score'], reverse=True)
                search_result = [(result, result.metadata['score']) for result in sorted_results[:20]]

            else:
                raise ValueError("Unknown retriever type")
    
            # 유사도 계산
            for doc, _ in search_result:
                combined_similarity = 0
                for idx, (query_embedding, weight) in enumerate(query_embeddings):
                    doc_embedding_key = f'embedding_{args.ensemble_models[idx]["name"]}'
                    doc_embedding = doc.metadata.get(doc_embedding_key) or ensemble_encoders[idx].embed_query(doc.page_content)
                    similarity = 1 - cosine(query_embedding, doc_embedding)
                    combined_similarity += weight * similarity
                combined_scores.append((doc, combined_similarity))

            # 중복 문서 제거: 가장 높은 점수를 받은 청크만 남김
            docid_scores = {}
            for doc, score in combined_scores:
                docid = doc.metadata.get('docid')
                if docid not in docid_scores or score > docid_scores[docid]['score']:
                    docid_scores[docid] = {'doc': doc, 'score': score}
            
            # 최종 선택된 상위 3개 문서
            final_results = sorted(docid_scores.values(), key=lambda x: x['score'], reverse=True)[:3]
        
        else:
            # 검색 결과 가져오기
            if isinstance(retriever, FAISS):
                search_result = retriever.similarity_search_with_relevance_scores(standalone_query, k=5)

            elif isinstance(retriever, BM25Retriever):
                results = search_with_scores(retriever, standalone_query, k=5)
                search_result = [(doc, score) for doc, score in results]

            elif isinstance(retriever, EnsembleRetriever):
                results = retriever.invoke(standalone_query)
                sorted_results = sorted(results, key=lambda x: x.metadata['score'], reverse=True)
                search_result = [(result, result.metadata['score']) for result in sorted_results[:20]]
            else:
                raise ValueError("Unknown retriever type")
            
            # 중복 문서 제거: 가장 높은 점수를 받은 청크만 남김
            docid_scores = {}
            for doc, score in search_result:
                docid = doc.metadata.get('docid')
                if docid not in docid_scores or score > docid_scores[docid]['score']:
                    docid_scores[docid] = {'doc': doc, 'score': score}
            
            # 최종 선택된 상위 3개 문서
            final_results = sorted(docid_scores.values(), key=lambda x: x['score'], reverse=True)[:3]

        retrieved_context = []
        ## Reranking
        if args.rerank:
            print("=" * 30)
            print("reranking...")
            reranked_docs = compression_retriever.invoke(standalone_query, search_kwargs={"k": 3})
    
            # 상위 3개의 문서만 선택하여 저장
            for doc in reranked_docs[:3]:
                retrieved_context.append(doc.page_content)
                response["topk"].append(doc.metadata.get('docid'))
                response["references"].append({
                    "docid": doc.metadata.get('docid'),
                    "score": doc.metadata.get('score'),
                    "content": doc.page_content
                })
        else:
            for result in final_results:
                doc = result['doc']
                score = result['score']
                retrieved_context.append(doc.page_content)
                response["topk"].append(doc.metadata.get('docid'))
                response["references"].append({
                    "docid": doc.metadata.get('docid'),
                    "score": score,
                    "content": doc.page_content
                })

        print("=" * 30)
        print("검색된 문서 정보:")
        for ref in response["references"]:
            if ref['score'] is not None:
                print(f"DocID: {ref['docid']}, Score: {ref['score']:.4f}\nContent: {ref['content']}\n")
            else:
                print(f"DocID: {ref['docid']}, Score: None\nContent: {ref['content']}\n")

    else:
        response['standalone_query'] = None
        print("답변을 생성할 수 없습니다.\n")
        response["answer"] = "질문이 과학 상식에 해당하지 않습니다."

    return response


def ollama_eval_rag(args, retriever):
    model = ChatOllama(model="eeve-10.8b-q8:latest")
    
    chain1 = ollama_standalone_query(model)
    chain2 = ollama_domain_check(model)
    chain3 = ollama_translate_query(model)
    chain4 = ollama_query_expansion(model)

    ## Query Ensemble Model Load
    if args.query_ensemble:
        ensemble_encoders = []
        for model_info in args.ensemble_models:
            model_type = model_info.get('type', 'hf')  # 기본값은 'hf'
            model_name = model_info['name']
            if model_type == 'hf':
                encoder = load_hf_encoder(model_name, args.model_kwargs, args.encode_kwargs)
            elif model_type == 'upstage':
                encoder = load_upstage_encoder(model_name)
            elif model_type == 'voyage':
                encoder = load_voyage_encoder(model_name)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            ensemble_encoders.append(encoder)
    else:
        ensemble_encoders = None

    ## ReRanking Model Load
    if args.rerank:
        if args.rerank_method == "huggingface":
            compression_retriever = load_hf_reranker(args.reranker_name, retriever)
        elif args.rerank_method == "gpt":
            compression_retriever = load_llm_reranker(args.rerank_method, args.reranker_name, retriever)
            
    else:
        compression_retriever = None

    with open(args.eval_file_path) as f, open(args.output_path, "w") as of:
        idx = 0
        for line in f:
            j = json.loads(line)
            print(f'Test {idx:>04}\nQuestion: {j["msg"]}')

            id = j['eval_id']
            chats = get_chat_history(j)
            if len(j["msg"]) > 1:
                query = chain1.invoke({"chat_history": chats})
            else:
                query = chats.split(':')[1].strip()

            domain_check_result = chain2.invoke({"query": query})
            print("=" * 30)
            print(f"Standalone_Query : {query}")
            
            print("=" * 30)
            print(f"Domain_Check : {domain_check_result}")

            # if domain_check_result == "False":
            if id in [276, 261, 283, 32, 94, 90, 220,  245, 229, 247, 67, 57, 2, 227, 301, 222, 83, 64, 103, 218]:
                query = None
            
            if not query is None and args.src_lang == "en":
                query = chain3.invoke({"ko_query" : query})
                print(f"EN Standalone_query : {query}")

            if not query is None and args.query_expansion:
                # query = chain4.invoke({"query" : query})
                query_intention = query_refinement(query, "gpt-4o", OpenAI())
                print(f"Refinement query : {query_intention}")

                query = f"{query_intention}\n{query}"
                query = query_expansion(query, "gpt-4o", OpenAI())
                print(f"Expanded query : {query}")

            response = ollama_answer_question(args, query, retriever, compression_retriever, ensemble_encoders)

            output = {"eval_id": j["eval_id"], "standalone_query": response["standalone_query"], "topk": response["topk"], "answer": response["answer"], "references": response["references"]}
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            idx += 1            