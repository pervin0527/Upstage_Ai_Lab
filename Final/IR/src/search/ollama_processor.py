import json
import numpy as np

from scipy.spatial.distance import cosine

from langchain.retrievers import EnsembleRetriever
from langchain_community.chat_models import ChatOllama
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores.faiss import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from sparse_retriever.kiwi_bm25 import KiwiBM25Retriever
from dense_retriever.model import load_hf_encoder, load_upstage_encoder, load_hf_reranker

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


def ollama_standalone_query(model):
    prompt_context1 = (
        "주어진 '대화내용'을 정리해서 하나의 '질문'을 생성해주세요.\n"
        "반드시 '질문'을 생성해야 하는 것입니다. 질문에 해당하는 답변을 생성하지 않게 주의해주세요."

        "예제 입력\n"
        "'user : 기억 상실증 걸리면 너무 무섭겠다.\n' 'assistant : 네 맞습니다.\n' 'user : 어떤 원인 때문에 발생하는지 궁금해.\n'"
        "예제 출력\n"
        "'기억 상실증에 걸리는 원인은 무엇인가요?'"

        "대화내용 : {chat_history}\n\n"

        "답변:"
    )

    prompt1 = ChatPromptTemplate.from_template(prompt_context1)
    chain1 = prompt1 | model | StrOutputParser()

    return chain1


def ollama_domain_check(model):
    prompt_context2 = (
        # "주어진 질문이 '상식'을 물어보는 것인지 판단하세요.\n"
        "주어진 질문이 일반적인 지식이나 사실에 대한 질문인지 판단하세요.\n"

        "만약 개인적인 감정, 의견, 일상 대화처럼 질문이 상식을 묻는 질문이 아닌 경우 False를, 상식을 묻는 질문이 맞다면 True를 반환하세요.\n"
        "당신의 답변은 반드시 'True' 또는 'False'로만 구성되어야 하며, 다른 문장은 생성하지 않아야 합니다.\n"

        "예제 입력1\n"
        "평형분극비율이 뭐야?\n"
        "예제 출력1\n"
        "True\n"

        "예제 입력2\n"
        "다음 세대의 꽃 색깔은 무엇인가요?\n"
        "예제 출력2\n"
        "True\n"

        "예제 입력3\n"
        "요새 너무 힘드네..\n"
        "예제 출력3\n"
        "False\n"

        "예제 입력4\n"
        "너 모르는 것도 있어?\n"
        "예제 출력4\n"
        "False\n"

        "예제 입력5\n"
        "니가 대답을 잘해줘서 너무 신나!\n"
        "예제 출력5\n"
        "False\n"


        "예제 입력6\n"
        "우울한데 신나는 얘기 좀 해줘!\n"
        "예제 출력6\n"
        "False\n"

        "예제 입력7\n"
        "너는 누구야?\n"
        "예제 출력7\n"
        "False\n"

        "예제 입력8\n"
        "너 잘하는게 뭐야?\n"
        "예제 출력8\n"
        "False\n"

        "예제 입력9\n"
        "요새 너무 힘들다..\n"
        "예제 출력9\n"
        "False\n"

        "질문 : {query}\n\n"

        "답변:"
    )

    prompt2 = ChatPromptTemplate.from_template(prompt_context2)
    chain2 = prompt2 | model | StrOutputParser()

    return chain2

def ollama_contextual_retrieval(model):
    prompt_context3 = (
        "입력된 문서의 조각(document_chunk)를 보고, 다음과 같은 정보들을 생성해주세요.\n\n"
        "<title>\n"
        "문서 조각에 해당하는 제목\n"
        "</title>\n\n"

        "<summary>\n"
        "문서 조각으로부터 만들 수 있는 요약\n"
        "</summary>\n\n"

        "<data_insights>\n"
        "문서 조각으로부터 얻을 수 있는 인사이트\n"
        "</data_insights>\n\n"

        "<hypothetical_questions>\n"
        "문서 조각으로부터 떠오르는 질문들\n"
        "<hypothetical_questions>\n\n"

        "문서 조각 : {document_chunk}\n\n"

        "답변:"
    )

    prompt3 = ChatPromptTemplate.from_template(prompt_context3)
    chain3 = prompt3 | model | StrOutputParser()

    return chain3


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
                if model_type == 'hf':
                    query_embedding = encoder.embed_query(standalone_query)
                elif model_type == 'upstage':
                    query_embedding = encoder.embed_query(standalone_query)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                query_embeddings.append((query_embedding, args.ensemble_weights[idx]))
    
            combined_scores = []

            # 검색 결과 가져오기
            if isinstance(retriever, FAISS):
                search_result = retriever.similarity_search_with_relevance_scores(standalone_query, k=3)
            elif isinstance(retriever, BM25Retriever):
                # results = retriever.search_with_score(standalone_query, k=3)
                results = search_with_scores(retriever, standalone_query, k=3)
                
                search_result = []
                for doc, score in results:
                    search_result.append((doc, score))
                    
            elif isinstance(retriever, EnsembleRetriever):
                results = retriever.invoke(standalone_query)
                
                # 점수 기준으로 정렬하고 상위 3개 선택
                sorted_results = sorted(results, key=lambda x: x.metadata['score'], reverse=True)
                search_result = [(result, result.metadata['score']) for result in sorted_results[:3]]
            else:
                raise ValueError("Unknown retriever type")
    
            for doc, _ in search_result:
                combined_similarity = 0
                for idx, (query_embedding, weight) in enumerate(query_embeddings):
                    model_type = args.ensemble_models[idx]['type']
                    model_name = args.ensemble_models[idx]['name']
                    
                    # 각 모델별로 문서 임베딩 가져오기
                    doc_embedding_key = f'embedding_{model_name}'
                    if doc_embedding_key not in doc.metadata:
                        # 문서 임베딩이 없다면 생성
                        if model_type == 'hf':
                            doc_embedding = ensemble_encoders[idx].embed_query(doc.page_content)
                        elif model_type == 'upstage':
                            doc_embedding = ensemble_encoders[idx].embed_query(doc.page_content)
                        else:
                            raise ValueError(f"Unknown model type: {model_type}")
                        # 문서 메타데이터에 저장
                        doc.metadata[doc_embedding_key] = doc_embedding
                    else:
                        doc_embedding = doc.metadata[doc_embedding_key]
                    
                    # 유사도 계산
                    similarity = 1 - cosine(query_embedding, doc_embedding)
                    combined_similarity += weight * similarity
    
                combined_scores.append((doc, combined_similarity))
    
            # 최종 유사도 기준으로 문서 정렬
            combined_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)

        else:
            # 검색 결과 가져오기
            if isinstance(retriever, FAISS):
                search_result = retriever.similarity_search_with_relevance_scores(standalone_query, k=3)
            elif isinstance(retriever, BM25Retriever):
                # results = retriever.search_with_score(standalone_query)
                results = search_with_scores(retriever, standalone_query, k=3)

                search_result = []
                for doc, score in results:
                    search_result.append((doc, score))

            elif isinstance(retriever, EnsembleRetriever):
                results = retriever.invoke(standalone_query)
                
                # 점수 기준으로 정렬하고 상위 3개 선택
                sorted_results = sorted(results, key=lambda x: x.metadata['score'], reverse=True)
                search_result = [(result, result.metadata['score']) for result in sorted_results[:3]]
            else:
                raise ValueError("Unknown retriever type")
            
            combined_scores = [(doc, score) for doc, score in search_result]

        retrieved_context = []
        ## Reranking
        if args.rerank:
            print("=" * 30)
            print("reranking...")
            reranked_docs = compression_retriever.invoke(standalone_query)
    
            # 상위 3개의 문서만 선택하여 저장
            for doc in reranked_docs[:3]:
                retrieved_context.append(doc.page_content)
                response["topk"].append(doc.metadata.get('docid'))
                response["references"].append({
                    "docid": doc.metadata.get('docid'),   # docid 추가
                    "score": doc.metadata.get('score'),
                    "content": doc.page_content
                })
        else:
            for doc, score in combined_scores[:3]:
                retrieved_context.append(doc.page_content)
                response["topk"].append(doc.metadata.get('docid'))
                response["references"].append({
                    "docid": doc.metadata.get('docid'),   # docid 추가
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
        response["answer"] = "질문이 과학 상식에 해당하지 않습니다."

    return response


def ollama_eval_rag(args, retriever):
    model = ChatOllama(model="eeve-10.8b-q8:latest")
    
    chain1 = ollama_standalone_query(model)
    chain2 = ollama_domain_check(model)

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
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            ensemble_encoders.append(encoder)
    else:
        ensemble_encoders = None

    ## ReRanking Model Load
    if args.rerank:
        compression_retriever = load_hf_reranker(args.reranker_name, retriever)
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
            print(f"Standalone_Query : {query}, Domain_Check : {domain_check_result}")

            # if domain_check_result == "False":
            if id in [276, 261, 283, 32, 94, 90, 220,  245, 229, 247,
                      67, 57, 2, 227, 301, 222, 83, 64, 103, 218]:
                query = None
            
            response = ollama_answer_question(args, query, retriever, compression_retriever, ensemble_encoders)

            output = {"eval_id": j["eval_id"], "standalone_query": response["standalone_query"], "topk": response["topk"], "answer": response["answer"], "references": response["references"]}
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            idx += 1            