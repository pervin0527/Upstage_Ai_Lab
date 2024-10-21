import json

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def clean_text(text):
    """불필요한 개행 문자를 제거하는 함수"""
    return text.replace('\n', ' ')

def process_and_save_documents(input_path='../dataset/documents.jsonl', output_path='../../dataset/processed_documents.jsonl'):
    """문서에서 개행 문자를 제거하고 결과를 JSONL 파일로 저장하는 함수"""
    raw_documents = load_jsonl(input_path)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in raw_documents:
            doc['content'] = clean_text(doc['content'])
            json.dump(doc, f, ensure_ascii=False)
            f.write('\n')

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]
    
def load_query(file_path):
    raw_queries = load_jsonl(file_path)

    queries = []
    for query in raw_queries:
        doc_id = query['docid']

        for i in range(1, 4):
            queries.append({"query": query[f'question{i}'], "metadata": {"docid": doc_id}})
    
    return queries

def load_eval(file_path):
    raw_evals = load_jsonl(file_path)

    evals = []
    for eval in raw_evals:
        eval_id = eval['eval_id']
        query = eval['query']
        evals.append({"eval_id":eval_id, "query":query})

    return evals

def load_document(file_path):
    raw_documents = load_jsonl(file_path)

    documents = []
    for doc in raw_documents:
        doc_id = doc['docid']
        content = doc['content']
        documents.append(Document(page_content=content, metadata={"docid": doc_id}))

    return documents

def chunking(args, documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )

    return text_splitter.split_documents(documents)