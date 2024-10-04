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

def load_document(path='../dataset/documents.jsonl'):
    raw_documents = load_jsonl(path)

    documents = []
    for doc in raw_documents:
        doc_id = doc['docid']
        content = doc['content']
        documents.append(Document(page_content=content, metadata={"docid": doc_id}))

    return documents

def chunk_documents(documents, chunk_size=1000, chunk_overlap=100):
    """문서를 청크로 나누는 함수"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            chunked_docs.append(
                Document(page_content=chunk, metadata={"docid": doc.metadata['docid'], "chunk_id": i})
            )
    return chunked_docs


if __name__ == "__main__":
    process_and_save_documents("../../dataset/documents.jsonl")