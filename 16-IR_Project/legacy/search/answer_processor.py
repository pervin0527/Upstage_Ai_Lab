import json
import traceback

from scipy.spatial.distance import cosine

from search.query_processor import create_standalone_query, domain_check, translate_query, query_expansion
from dense_retriever.model import load_hf_reranker, load_hf_encoder, load_openai_encoder, load_upstage_encoder

# RAG 구현에 필요한 Question Answering을 위한 LLM  프롬프트
persona_qa = """
## Role: 과학 상식 전문가

## Instructions
- 사용자의 이전 메시지 정보 및 주어진 Reference 정보를 활용하여 간결하게 답변을 생성한다.
- 주어진 검색 결과 정보로 대답할 수 없는 경우는 정보가 부족해서 답을 할 수 없다고 대답한다.
- 한국어로 답변을 생성한다.
"""

# RAG 구현에 필요한 질의 분석 및 검색 이외의 일반 질의 대응을 위한 LLM 프롬프트
persona_function_calling = """
## Role: 과학 상식 전문가

## Instruction
- 사용자가 대화를 통해 과학 지식에 관한 주제로 질문하면 search api를 호출할 수 있어야 한다.
- 과학 상식과 관련되지 않은 나머지 대화 메시지에는 적절한 대답을 생성한다.
"""

# Function calling에 사용할 함수 정의
tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "search relevant documents",
            "parameters": {
                "properties": {
                    "standalone_query": {
                        "type": "string",
                        "description": "Final query suitable for use in search from the user messages history."
                    }
                },
                "required": ["standalone_query"],
                "type": "object"
            }
        }
    },
]


def answer_question(messages, retriever, client, args, compression_retriever=None, ensemble_encoders=None):
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}
    
    # 쿼리 검사1: standalone query 생성
    phase1_query = create_standalone_query(messages, args.llm_model, client, "ko")
    print("=" * 30)
    print("멀티턴 검사 결과")
    print(phase1_query)
    
    # 쿼리 검사2: 과학 상식 관련 쿼리 확인
    print("=" * 30)
    print("도메인 검사 결과")
    phase2_query = domain_check(phase1_query['query'], args.llm_model, client, "ko")
    print(phase2_query)
    
    if not phase2_query.get('out_of_domain', True):
        if args.src_lang == "en":
            phase2_query = translate_query(phase2_query['query'], args.llm_model, client)
            print("=" * 30)
            print("번역 결과")
            print(phase2_query)

        if args.query_expansion:
            print("=" * 30)
            print("쿼리 확장 결과")
            phase2_query = query_expansion(phase2_query['query'], args.llm_model, client)
            print(phase2_query)
    
        query = phase2_query['query']
        response['standalone_query'] = query
    
        if args.query_ensemble:
            print("=" * 30)
            print("query ensembling...")
            if len(args.ensemble_models) != len(args.ensemble_weights):
                raise ValueError("ensemble_models와 ensemble_weights의 길이가 동일해야 합니다.")
    
            query_embeddings = []
            for idx, encoder in enumerate(ensemble_encoders):
                model_type = args.ensemble_models[idx]['type']
                if model_type == 'hf':
                    query_embedding = encoder.embed_query(query)
                elif model_type == 'upstage':
                    query_embedding = encoder.embed_query(query)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                query_embeddings.append((query_embedding, args.ensemble_weights[idx]))
    
            combined_scores = []
            search_result = retriever.similarity_search_with_relevance_scores(query, k=3)
            for doc, score in search_result:
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
            # 쿼리 앙상블을 하지 않는 경우
            search_result = retriever.similarity_search_with_relevance_scores(query, k=3)
            combined_scores = [(doc, score) for doc, score in search_result]
    
        retrieved_context = []
        if args.rerank:
            print("=" * 30)
            print("reranking...")
            reranked_docs = compression_retriever.invoke(query)
    
            # 상위 3개의 문서만 선택하여 저장
            for doc in reranked_docs:
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
        
        # 문서 정보 출력 (docid, score, content)
        print("검색된 문서 정보:")
        for ref in response["references"]:
            if ref['score'] is not None:
                print(f"DocID: {ref['docid']}, Score: {ref['score']:.4f}\nContent: {ref['content']}\n")
            else:
                print(f"DocID: {ref['docid']}, Score: None\nContent: {ref['content']}\n")

    
        content = "\n".join(retrieved_context)
        messages.append({"role": "assistant", "content": content})
        
        # 최종 답변 생성
        msg = [{"role": "system", "content": persona_qa}] + messages
        try:
            qaresult = client.chat.completions.create(
                model=args.llm_model,
                messages=msg,
                temperature=0,
                seed=1,
                timeout=30
            )
            response["answer"] = qaresult.choices[0].message.content
    
        except Exception as e:
            traceback.print_exc()
            response["answer"] = "답변 생성 중 오류가 발생했습니다."
    
    else:
        response["answer"] = "질문이 과학 상식에 해당하지 않습니다."
    
    return response



def eval_rag(args, retriever, client):
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

    if args.rerank:
        if args.rerank_method == "huggingface":
            compression_retriever = load_hf_reranker(args.reranker_name, retriever)

    else:
        compression_retriever = None

    with open(args.eval_file_path) as f, open(args.output_path, "w") as of:
        idx = 0
        for line in f:
            j = json.loads(line)
            print(f'Test {idx:>04}\nQuestion: {j["msg"]}')

            response = answer_question(j["msg"], retriever, client, args, compression_retriever, ensemble_encoders)
            print(f'Answer: {response["answer"]}\n')

            # 대회 score 계산은 topk 정보를 사용, answer 정보는 LLM을 통한 자동평가시 활용
            output = {"eval_id": j["eval_id"], "standalone_query": response["standalone_query"], "topk": response["topk"], "answer": response["answer"], "references": response["references"]}
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            idx += 1