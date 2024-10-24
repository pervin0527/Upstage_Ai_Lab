import json

from langchain.schema import Document

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]
    

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


def load_hypothetical_quries(file_path):
    raw_queries = load_jsonl(file_path)

    queries = []
    for query in raw_queries:
        docid = query['docid']

        for i in range(1, 4):
            queries.append(Document(page_content=query[f'question{i}'], metadata={"docid":docid}))
    
    return queries


def load_document_summary(file_path):
    raw_documents = load_jsonl(file_path)

    documents = []
    for doc in raw_documents:
        doc_id = doc['docid']
        content = doc['content']
        documents.append(Document(page_content=content, metadata={"docid": doc_id}))

    return documents


def load_en_eval(file_path):
    raw_queries = load_jsonl(file_path)

    en_evals = []
    for query in raw_queries:
        eval_id = query['eval_id']
        en_query = query['en_query']

        en_evals.append({"eval_id":eval_id, "query":en_query})

    return en_evals


def load_doc_dataset(cfg):
    print("=" * 70)
    print("Document Loading")
    dataset = {}

    full_documents = load_document(cfg['dataset']['full_doc_file'])
    print(f"  {cfg['dataset']['full_doc_file']} Loaded. num of data : {len(full_documents)}")
    dataset['full_documents'] = full_documents

    documents = load_document(cfg['dataset']['doc_file'])
    print(f"  {cfg['dataset']['doc_file']} Loaded. num of data : {len(documents)}")
    dataset['documents'] = documents
    
    if not cfg['dataset']['summary_doc_file'] is None:
        summaries = load_document_summary(cfg['dataset']['summary_doc_file'])
        dataset["summaries"] = summaries
        print(f"  {cfg['dataset']['summary_doc_file']} Loaded. num of data : {len(summaries)}")
    
    if not cfg['dataset']['en_eval_file'] is None:
        en_queries = load_en_eval(cfg['dataset']['en_eval_file'])
        dataset["en_queries"] = en_queries
        print(f"  {cfg['dataset']['en_eval_file']} Loaded. num of data : {len(en_queries)}")

    if not cfg['dataset']['en_doc_file'] is None:
        en_documents = load_document(cfg['dataset']['en_doc_file'])
        dataset["en_documents"] = en_documents
        print(f"  {cfg['dataset']['en_doc_file']} Loaded. num of data : {len(en_documents)}")

    print(f"Dataset keys : {dataset.keys()}")
    
    return dataset
