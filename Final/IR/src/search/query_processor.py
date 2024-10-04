import re
import json

def clean_json_response(response):
    # 코드 블록(예: ```json, ```) 제거
    cleaned_response = re.sub(r'```(?:json)?', '', response).strip()
    return cleaned_response


def create_standalone_query(query, model, client):
    standalone_content = ("입력된 내용이 한 줄의 문장인지 여러 줄의 대화 내용인지 분류하세요. 반드시 JSON 형식으로 응답하며, 키와 값은 모두 이중 따옴표로 감쌉니다."
                        "- 단일 문장인 경우 : {\"multi_turn\": false, \"query\": \"입력 문장\"} "
                        "- 여러 줄의 대화 내용인 경우 : {\"multi_turn\": true, \"query\": \"대화 내용을 종합하여 만든 새로운 질문\"}")

    # query가 중첩된 리스트 형태일 때 텍스트를 추출
    if isinstance(query, list) and 'content' in query[0]:
        query = query[0]['content'][0]['content']
    
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": standalone_content},
            {"role": "user", "content": query}
        ],
    )
    
    response = completion.choices[0].message.content
    cleaned_response = clean_json_response(response)
    
    try:
        json_response = json.loads(cleaned_response)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response", "response": response}
    
    return json_response


def domain_check(query, model, client):
    domain_content = (
        "입력된 텍스트가 상식에 대한 질문인지 아니면 단순한 일상적인 대화인지 구분하세요."
        " - 상식에 대한 질문이란, 사용자가 지식이나 정보를 얻기 위해 하는 질문입니다. 예: '나무의 분류에 대해 조사하는 방법은?', 'Dmitri Ivanovsky가 누구야?', '남녀 관계에서 정서적인 행동이 왜 중요해?'"
        " - 일상적인 대화란, 주로 감정이나 의견을 표현하거나 대화의 흐름을 유지하기 위한 내용입니다. 예: '요새 너무 힘들다.', '니가 대답을 잘해줘서 너무 신나!', '이제 그만 얘기해!', '오늘 너무 즐거웠다!', ''너는 누구니?', '너는 어떤 능력을 가지고 있니?"
        " 반드시 JSON 형식으로 응답하세요. 키와 값은 모두 이중 따옴표로 감싸야 합니다."
        " - 상식에 대한 질문인 경우: {\"out_of_domain\": false, \"query\": \"입력된 쿼리를 그대로 반환\"}"
        " - 일상적인 대화인 경우: {\"out_of_domain\": true, \"query\": \"적절한 응답을 할 수 없습니다.\"}"
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": domain_content},
            {"role": "user", "content": query}
        ],
    )
    
    response = completion.choices[0].message.content
    cleaned_response = clean_json_response(response)
    
    try:
        json_response = json.loads(cleaned_response)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response", "response": response}
    
    return json_response