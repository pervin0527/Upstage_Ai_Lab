import os
import copy
import time
import json
import huggingface_hub

from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv("../keys.env")

openai_api_key = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = openai_api_key


def format_rerank_results(query, top_docs, docs, scores):
    formatted_hits = []
    for idx, docid in enumerate(top_docs):
        # docid로 문서를 검색하여 해당 문서의 인덱스를 찾아냄
        matching_doc = next((doc for doc in docs if doc['metadata']['docid'] == docid), None)
        if matching_doc:
            formatted_hits.append({
                'idx': idx, 
                'content': matching_doc['content'], 
                'score': scores[idx]  # 실제 점수를 반영
            })
    result = {
        'query': query,
        'hits': formatted_hits
    }
    return result


def reranking(query, top_docs, docs, top_k, initial_scores):
    # 초기 점수를 reranking 함수로 전달
    new_item = permutation_pipeline(
        item=format_rerank_results(query, top_docs, docs, scores=initial_scores),  # 실제 점수 사용
        rank_start=0, 
        rank_end=len(top_docs), 
        model_name='gpt-4o',
        api_key=openai_api_key
    )
    reranked_doc_indices = [hit['idx'] for hit in new_item['hits']]
    reranked_scores = [hit['score'] for hit in new_item['hits']]

    return reranked_doc_indices[:top_k], reranked_scores[:top_k]


class OpenaiClient:
    def __init__(self, keys=None, start_id=None, proxy=None):
        from openai import OpenAI
        import openai
        if isinstance(keys, str):
            keys = [keys]
        if keys is None:
            raise "Please provide OpenAI Key."

        self.key = keys
        self.key_id = start_id or 0
        self.key_id = self.key_id % len(self.key)
        self.api_key = self.key[self.key_id % len(self.key)]
        self.client = OpenAI(api_key=self.api_key)

    def chat(self, *args, return_text=False, reduce_length=False, **kwargs):
        while True:
            try:
                completion = self.client.chat.completions.create(*args, **kwargs, timeout=30)
                break
            except Exception as e:
                print(str(e))
                if "This model's maximum context length is" in str(e):
                    print('reduce_length')
                    return 'ERROR::reduce_length'
                time.sleep(0.1)
        if return_text:
            completion = completion.choices[0].message.content
        return completion

    def text(self, *args, return_text=False, reduce_length=False, **kwargs):
        while True:
            try:
                completion = self.client.completions.create(
                    *args, **kwargs
                )
                break
            except Exception as e:
                print(e)
                if "This model's maximum context length is" in str(e):
                    print('reduce_length')
                    return 'ERROR::reduce_length'
                time.sleep(0.1)
        if return_text:
            completion = completion.choices[0].text
        return completion


def convert_messages_to_prompt(messages):
    #  convert chat message into a single prompt; used for completion model (eg davinci)
    prompt = ''
    for turn in messages:
        if turn['role'] == 'system':
            prompt += f"{turn['content']}\n\n"
        elif turn['role'] == 'user':
            prompt += f"{turn['content']}\n\n"
        else:  # 'assistant'
            pass
    prompt += "The ranking results of the 20 passages (only identifiers) is:"
    return prompt


def run_retriever(topics, searcher, qrels=None, k=100, qid=None):
    ranks = []
    if isinstance(topics, str):
        hits = searcher.search(topics, k=k)
        ranks.append({'query': topics, 'hits': []})
        rank = 0
        for hit in hits:
            rank += 1
            content = json.loads(searcher.doc(hit.docid).raw())
            if 'title' in content:
                content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
            else:
                content = content['contents']
            content = ' '.join(content.split())
            ranks[-1]['hits'].append({
                'content': content,
                'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
        return ranks[-1]

    for qid in tqdm(topics):
        if qid in qrels:
            query = topics[qid]['title']
            ranks.append({'query': query, 'hits': []})
            hits = searcher.search(query, k=k)
            rank = 0
            for hit in hits:
                rank += 1
                content = json.loads(searcher.doc(hit.docid).raw())
                if 'title' in content:
                    content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
                else:
                    content = content['contents']
                content = ' '.join(content.split())
                ranks[-1]['hits'].append({
                    'content': content,
                    'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
    return ranks


def get_prefix_prompt(query, num):
    return [{'role': 'system',
             'content': "당신은 RankGPT입니다. 주어진 쿼리에 대한 관련성에 따라 문서를 순위 매기는 지능형 어시스턴트입니다."
             },
            {'role': 'user',
             'content': f"제가 {num}개의 문서를 번호로 구분하여 제공하겠습니다.\n이 문서들을 쿼리: {query}와 얼마나 높은 관련성을 갖는지를 기준으로 순위 매겨 주세요."},
            {'role': 'assistant', 'content': '알겠습니다. 문서들을 제공해 주세요.'}]


def get_post_prompt(query, num):
    return f"검색 쿼리: {query}. \n위의 {num}개의 문서를 검색 쿼리에 대한 관련성에 따라 순위 매겨 주세요. 문서는 식별 번호를 사용하여 내림차순으로 나열해야 합니다. 가장 관련성 높은 문서부터 먼저 나열해야 합니다.  출력 형식은 [] > []로, 예: [1] > [2]와 같이 표기해야 합니다. 순위 결과만 응답하고, 다른 설명은 하지 마세요."

def create_permutation_instruction(item=None, rank_start=0, rank_end=100, model_name='gpt-4o'):
    query = item['query']
    num = len(item['hits'][rank_start: rank_end])

    messages = get_prefix_prompt(query, num)
    rank = 0
    for hit in item['hits'][rank_start: rank_end]:
        rank += 1
        content = hit['content']
        content = content.replace('Title: Content: ', '')
        content = content.strip()  # 문서 전체 내용을 사용
        messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
    messages.append({'role': 'user', 'content': get_post_prompt(query, num)})

    return messages


def run_llm(messages, api_key=None, model_name="gpt-4o"):
    Client = OpenaiClient

    agent = Client(api_key)
    response = agent.chat(model=model_name, messages=messages, temperature=0, return_text=True)

    return response


def clean_response(response: str):
    new_response = ''
    for c in response:
        if not c.isdigit():
            new_response += ' '
        else:
            new_response += c
    new_response = new_response.strip()
    return new_response


def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response


def receive_permutation(item, permutation, rank_start=0, rank_end=100):
    response = clean_response(permutation)
    response = [int(x) - 1 for x in response.split()]
    response = remove_duplicate(response)
    cut_range = copy.deepcopy(item['hits'][rank_start: rank_end])
    original_rank = [tt for tt in range(len(cut_range))]
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]
    for j, x in enumerate(response):
        item['hits'][j + rank_start] = copy.deepcopy(cut_range[x])
        if 'rank' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['rank'] = cut_range[j]['rank']
        if 'score' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['score'] = cut_range[j]['score']
    return item


def permutation_pipeline(item=None, rank_start=0, rank_end=100, model_name='gpt-4o', api_key=None):
    messages = create_permutation_instruction(item=item, rank_start=rank_start, rank_end=rank_end, model_name=model_name)
    permutation = run_llm(messages, api_key=api_key, model_name=model_name)
    item = receive_permutation(item, permutation, rank_start=rank_start, rank_end=rank_end)

    return item