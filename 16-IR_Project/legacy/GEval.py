import os
import re
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("../keys.env")

openai_api_key = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = openai_api_key

client = OpenAI()

def get_documents_by_ids(document_ids, documents):
    matched_documents = []
    for doc in documents:
        if doc['docid'] in document_ids:
            matched_documents.append(doc['content'])
    return matched_documents

def evaluate_relevance(query, documents):
    context = (
        """
        주어진 (쿼리, 문서들)을 보고, 각각의 문서와 쿼리가 얼마나 관련있는지 1~5점 사이의 점수로 평가하세요.
        핵심은 쿼리가 가진 질문자의 의도와 문서가 가진 핵심 내용이 얼마나 적합한가 입니다.
        """
    )
    combined_input = f"Query: {query}\nDocuments:\n" + "\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(documents)])

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": combined_input}
        ],
    )
    
    response = completion.choices[0].message.content
    
    ratings = re.findall(r'\b[1-5]\b', response)
    return [int(rating) for rating in ratings[:len(documents)]]

def main():
    csv_file_path = './outputs/UP-CH-ER.csv'
    jsonl_file_path = '../dataset/processed_documents.jsonl'

    print("Loading and processing data...")
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    extracted_data = []
    for line in lines:
        standalone_query_match = re.search(r'"standalone_query":\s*"([^"]+)"', line)
        topk_match = re.search(r'"topk":\s*\[([^\]]+)\]', line)
        
        if standalone_query_match and topk_match:
            standalone_query = standalone_query_match.group(1)
            topk = [doc_id.strip().strip('"') for doc_id in topk_match.group(1).split(',')]
            extracted_data.append({"standalone_query": standalone_query, "topk": topk})

    extracted_df = pd.DataFrame(extracted_data)

    documents = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            documents.append(json.loads(line))

    extracted_df['retrieved_documents'] = extracted_df['topk'].apply(lambda doc_ids: get_documents_by_ids(doc_ids, documents))

    print("Evaluating relevance scores...")
    tqdm.pandas()
    extracted_df['relevance_scores'] = extracted_df.progress_apply(lambda row: evaluate_relevance(row['standalone_query'], row['retrieved_documents']), axis=1)

    # Calculate average score for each query
    extracted_df['average_score'] = extracted_df['relevance_scores'].apply(np.mean)

    # Calculate overall average score
    overall_average_score = extracted_df['average_score'].mean()

    print(f"\nOverall average relevance score: {overall_average_score:.2f}")

    csv_output_path = './geval_results.csv'
    extracted_df.to_csv(csv_output_path, index=False)
    print(f"Results saved to {csv_output_path}")

if __name__ == "__main__":
    main()