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


def query_refinement(query, model, client):
    context = (
        """
        당신은 주어진 한국어 질의를 이해하고, 핵심 의도를 파악하는 전문가입니다.
        주어진 질의에서 가장 중요한 정보를 추출하여 한 줄로 간결하게 표현하세요.
        답변은 반드시 한 줄의 한국어 문장으로 해야 하며, 제 요청 또는 질의의 주제를 벗어난 답변은 하지 마세요.

        예시1
            Dmitri Ivanovsky가 누구야? 
            Dmitri Ivanovsky(디미트리 이바노스키)가 어떤 인물인지 알고 싶어합니다.

        예시2
            피임을 하기 위한 방법중 약으로 처리하는 방법은 쓸만한가?
            피임약을 사용하는 것이 모든 피임 방법들 중 얼마나 효과적인지 알고 싶어합니다.
        예시3
            헬륨이 다른 원소들과 반응을 잘 안하는 이유는?
            헬륨(He)이 다른 화학 원소들과 잘 반응하지 않는 이유를 알고 싶어합니다.

        예시4 
            서울에서 가장 좋은 카페는 어디야?
            서울에서 가장 인기 많은 카페를 추천받고 싶어합니다.
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


def query_expansion(query, model: str, client):
    content = (
        """
        당신은 한국어 질의를 이해하여 정확한 문서들을 검색하도록 질의를 개선하는 전문가입니다.
        사용자의 질문 의도를 명확히 파악하고, 그 의도를 유지하면서 검색 시스템이 관련 문서를 더 잘 찾을 수 있도록 질의를 더 명확하게 만드세요.
        질문의 의미를 확장할 필요가 있다면, 추가 정보를 최소화하여 의도를 왜곡하지 않도록 주의하세요.

        다음은 몇 가지 예시입니다.

        예제1
            입력: Thomas Alva Edison은 누구인가요?
            출력: Thomas Alva Edison이라는 인물에 대해 묻는 질문입니다.

        예제2
            입력: 온난 전선이 발생하면 이후 날씨는 어떻게 되나?
            출력: 기상 현상에서 온난 전선이 발생한 후 날씨에 대해 묻는 질문입니다.

        확장된 질의는 자연스러운 문장 형태로 제공되어야 하며, 구체화만 필요할 때에만 확장을 수행하세요.
        """
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": content},
            {"role": "user", "content": query}
        ],
    )
    
    response = completion.choices[0].message.content
    
    return response.strip()
