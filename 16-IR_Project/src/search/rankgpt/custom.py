import re
import openai

client = openai
model = "gpt-4o"

def custom_reranker(query, final_results):
    relevant_results = []
    
    for result in final_results:
        doc_content = result['doc'].page_content
        messages = [
            {"role": "user", "content": f"쿼리: {query}\n문서: {doc_content}\n해당 문서의 내용이 쿼리와 연관성이 있는지 'True' 또는 'False'로만 답하세요."}
        ]
        
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        relevance = completion.choices[0].message.content.strip()

        # 관련성이 높다고 판단되면 문서를 유지
        if relevance == "True":
            relevant_results.append(result)

    return relevant_results  # 관련성 높은 문서들만 반환