import json
import traceback

from search.query_processor import create_standalone_query, domain_check

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

def answer_question(messages, retriever, client, model):
    # 함수의 출력값 초기화
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}
    
    # 쿼리 검사1: 멀티턴 대화인 경우 standalone query를 생성한다.
    result1 = create_standalone_query(messages, model, client)
    print(result1)
    
    # 쿼리 검사2: 쿼리가 과학 상식과 관련된 것인지 검사한다.
    result2 = domain_check(result1['query'], model, client)
    print(result2)
    
    if not result2.get('out_of_domain', True):
        query = result2['query']
        response['standalone_query'] = query
        
        # Retriever를 통해 관련 문서 검색 및 점수 반환
        search_result = retriever.similarity_search_with_relevance_scores(query, k=3)
        
        if not search_result:
            response["answer"] = "관련된 문서를 찾을 수 없습니다."
            return response
        
        # 검색된 문서에서 상위 3개만 선택하여 저장
        retrieved_context = []
        for doc, score in search_result:
            retrieved_context.append(doc.page_content)
            response["topk"].append(doc.metadata.get('docid'))
            response["references"].append({
                "score": score,  # 검색에서 반환된 유사도 점수
                "content": doc.page_content
            })
        
        # 검색된 문서들을 assistant 메시지에 추가
        content = "\n".join(retrieved_context)
        messages.append({"role": "assistant", "content": content})
        
        # 검색된 문서들을 바탕으로 최종 답변 생성
        msg = [{"role": "system", "content": persona_qa}] + messages
        try:
            qaresult = client.chat.completions.create(
                model=model,
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


def eval_rag(eval_filename, output_filename, retriever, client, model):
    with open(eval_filename) as f, open(output_filename, "w") as of:
        idx = 0
        for line in f:
            print(f"{idx:>04}")
            j = json.loads(line)
            print(f'Test {idx:>04}\nQuestion: {j["msg"]}')
            response = answer_question(j["msg"], retriever, client, model)
            print(f'Answer: {response["answer"]}\n')

            # 대회 score 계산은 topk 정보를 사용, answer 정보는 LLM을 통한 자동평가시 활용
            output = {"eval_id": j["eval_id"], "standalone_query": response["standalone_query"], "topk": response["topk"], "answer": response["answer"], "references": response["references"]}
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            idx += 1