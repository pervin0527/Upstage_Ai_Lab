from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def ollama_standalone_query(model):
    context = (
        "주어진 '대화내용'을 정리해서 하나의 '질문'을 생성해주세요.\n"
        "반드시 '질문'을 생성해야 하는 것입니다. 질문에 해당하는 답변을 생성하지 않게 주의해주세요."

        "예제 입력\n"
        "'user : 기억 상실증 걸리면 너무 무섭겠다.\n' 'assistant : 네 맞습니다.\n' 'user : 어떤 원인 때문에 발생하는지 궁금해.\n'"
        "예제 출력\n"
        "'기억 상실증에 걸리는 원인은 무엇인가요?'"

        "대화내용 : {chat_history}\n\n"

        "답변:"
    )

    prompt = ChatPromptTemplate.from_template(context)
    chain = prompt | model | StrOutputParser()

    return chain


def ollama_domain_check(model):
    context = (
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

    prompt = ChatPromptTemplate.from_template(context)
    chain = prompt | model | StrOutputParser()

    return chain


def ollama_contextual_retrieval(model):
    context = (
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

    prompt = ChatPromptTemplate.from_template(context)
    chain = prompt | model | StrOutputParser()

    return chain


def ollama_translate_query(model):
    context = (
        """
        당신은 한국어 문장을 영어로 번역하는 전문 번역가입니다.\n
        입력된 한국어 질의 문장을 영어 질의 문장으로 번역해주세요.\n
        답변은 오직 번역된 영어 문장으로만 하고 다른 것은 생성하지 마세요.\n\n

        입력 : {ko_query}\n\n
        출력 : 
        """
    )

    prompt = ChatPromptTemplate.from_template(context)
    chain = prompt | model | StrOutputParser()

    return chain


def ollama_query_expansion(model):
    context = (
        """
        당신은 주어진 한국어 질의를 더 구체적이고 명확하게 만들어 검색 결과를 향상시키는 전문가입니다.\n
        주어진 질의는 BM25와 벡터 임베딩 검색 엔진에 사용되며, 검색 결과에서 더 정확한 문서가 검색될 수 있도록 질의를 적절히 개선하세요.\n
        개선된 질의는 원래의 의미를 유지하면서도 구체적으로 만들어야 합니다. 검색 목적이 명확해지도록 하며, 불필요한 정보는 포함하지 마세요.\n\n

        예제입력1\n
        질의: 디엔에이 리가아제의 주요 기능이 뭐야?\n
        개선된 질의: 디엔에이(DNA) 리가아제의 주요 기능은 무엇인가요?\n\n

        예제입력2\n
        질의: 나무의 분류에 대해 조사해 보기 위한 방법은?\n
        개선된 질의: 나무의 생물학적 분류 방법과 그 조사 방법에 대해 어떻게 연구할 수 있나요?\n\n

        예제입력3\n
        질의: 물리학에서 에너지 보존 법칙이 뭔데?\n
        개선된 질의: 물리학에서 에너지 보존 법칙의 정의와 그 법칙의 주요 개념은 무엇인가요?\n\n

        주어진 질의를 개선해주세요.\n

        입력 : {query}\n
        출력 : 
        """
    )

    prompt = ChatPromptTemplate.from_template(context)
    chain = prompt | model | StrOutputParser()

    return chain
