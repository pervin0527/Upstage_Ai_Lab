import re
import json

def clean_json_response(response):
    # 코드 블록(예: ```json, ```) 제거
    cleaned_response = re.sub(r'```(?:json)?', '', response).strip()
    return cleaned_response

def contextual_retrieval(document_chunk, model, client):
    context = (
    "입력된 문서 조각(document_chunk)의 내용을 기반으로, 책에서 읽을 수 있는 것처럼 자연스럽고 매끄러운 글을 작성하세요. 다음 항목을 하나의 연결된 글처럼 포함시켜 주세요:\n\n"
    "- 제목\n"
    "- 요약\n"
    "- 인사이트\n"
    "- 관련 질문들\n\n"

    "문서의 내용이 자연스럽게 연결되어야 하며, 정보가 매끄럽게 이어져야 합니다.\n\n"

    "문서 조각 : {document_chunk}\n\n"

    "반드시 JSON 형식으로 응답하며, 키와 값은 모두 이중 따옴표로 감쌉니다."
    "{\" gen_doc : \" : \" 생성한 문서 \"}"
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": document_chunk}
        ],
    )
    
    response = completion.choices[0].message.content
    cleaned_response = clean_json_response(response)
    
    try:
        json_response = json.loads(cleaned_response)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response", "response": response}
    
    return json_response