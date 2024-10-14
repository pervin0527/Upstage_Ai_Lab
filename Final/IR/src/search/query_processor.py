import re
import json

def clean_json_response(response):
    # 코드 블록(예: ```json, ```) 제거
    cleaned_response = re.sub(r'```(?:json)?', '', response).strip()
    return cleaned_response


def create_standalone_query(query, model, client, src_lang):
    # 대화 내용이 multi-turn인 경우, 각 대화 턴을 문자열로 변환
    dialogue_content = []
    for turn in query:
        dialogue_content.append(f'"{turn["role"]}" : "{turn["content"]}"')
    dialogue_content = "\n".join(dialogue_content)

    if src_lang == "en":
        standalone_content = (
            "Classify whether the input is a single sentence or multiple turns of dialogue. "
            "Respond strictly in JSON format, with all keys and values enclosed in double quotes."
            "- If it is a single sentence: {\"multi_turn\": false, \"query\": \"input sentence\"} "
            "- If it is multiple turns of dialogue: {\"multi_turn\": true, \"query\": \"new question synthesized from the dialogue\"}"
        )

    else:
        standalone_content = (
            "입력된 내용이 한 줄의 문장인지 여러 줄의 대화 내용인지 분류하세요. "
            "반드시 JSON 형식으로 응답하며, 키와 값은 모두 이중 따옴표로 감쌉니다."
            "- 단일 문장인 경우 : {\"multi_turn\": false, \"query\": \"입력 문장\"} "
            "- 여러 줄의 대화 내용인 경우 : {\"multi_turn\": true, \"query\": \"대화 내용을 종합하여 만든 새로운 질문\"}"
        )
    
    # LLM 호출
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": standalone_content},
            {"role": "user", "content": dialogue_content}
        ],
    )
    
    response = completion.choices[0].message.content
    cleaned_response = clean_json_response(response)
    
    try:
        json_response = json.loads(cleaned_response)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response", "response": response}
    
    return json_response


def domain_check(query, model, client, src_lang):
    if src_lang == "en":
        domain_content = (
            "Classify whether the input text is a general knowledge question or just a casual conversation. "
            "- A general knowledge question is one where the user seeks information or knowledge. Examples: 'How to research the classification of trees?', 'Who is Dmitri Ivanovsky?', 'Why are emotional behaviors important in relationships?'"
            "- A casual conversation is mainly about expressing emotions or opinions, or maintaining the flow of the conversation. Examples: 'I’ve been feeling exhausted lately.', 'I'm so excited because you answered so well!', 'Let's stop talking now!', 'I had a great time today!', 'Who are you?', 'What abilities do you have?'"
            " Respond strictly in JSON format. All keys and values must be enclosed in double quotes."
            "- If it is a general knowledge question: {\"out_of_domain\": false, \"query\": \"Return the input query as is\"}"
            "- If it is a casual conversation: {\"out_of_domain\": true, \"query\": \"Unable to provide an appropriate response.\"}"
        )
    else:
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


def translate_query(query, model, client):
    domain_content = (
        " 당신은 한국어를 영어로 번역하는 전문가입니다. "
        " 주어진 한국어 문장을 영어로 번역해주세요. 코드를 작성한다거나 이상한 행동을 하지 않고, 단순히 입력된 한국어 문장을 영어로 번역하기만 하면 됩니다."
        " 반드시 JSON 형식으로 응답하세요. 키와 값은 모두 이중 따옴표로 감싸야 합니다."
        " {\"query\": \"영어로 번역된 문장\"}"
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


def query_expansion(query, model:str, client):
    context = (
        """
        당신은 주어진 한국어 질의를 더 구체적이고 명확하게 만들어 검색 결과를 향상시키는 전문가입니다.
        주어진 질의는 BM25와 벡터 임베딩 검색 엔진에 사용되며, 검색 결과에서 더 정확한 문서가 검색될 수 있도록 질의를 적절히 개선하세요.
        개선된 질의는 원래의 의미를 유지하면서도 구체적으로 만들어야 합니다. 검색 목적이 명확해지도록 하며, 불필요한 정보는 포함하지 마세요.

        예제1
            나무의 분류에 대해 조사해 보기 위한 방법은?
            나무의 생물학적 분류 방법과 그 조사 방법에 대해 어떻게 연구할 수 있나요?

        예제2
            물리학에서 에너지 보존 법칙이 뭔데?
            물리학에서 에너지 보존 법칙의 정의와 그 법칙의 주요 개념은 무엇인가요?
        """
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role" : "system", "content" : context},
            {"role" : "user", "content" : query}
        ],
    )
    
    response = completion.choices[0].message.content

    return response