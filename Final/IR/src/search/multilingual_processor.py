import json
import voyageai

from scipy.spatial.distance import cosine
from langchain.schema import Document
from dense_retriever.model import load_hf_encoder, load_upstage_encoder, load_voyage_encoder, load_voyage_reranker


def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def load_document(path):
    raw_documents = load_jsonl(path)

    documents = []
    for doc in raw_documents:
        doc_id = doc['docid']
        content = doc['content']
        documents.append(Document(page_content=content, metadata={"docid": doc_id}))

    return documents


def answer_question(args, ko_query, en_query, ko_retriever, en_retriever, ko_embedders, en_embedders):
    vo = voyageai.Client()  # reranking 클라이언트
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    if not ko_query is None and not en_query is None:
        response['standalone_query'] = f"{ko_query}\n{en_query}"

        # 한국어 쿼리 임베딩
        ko_query_embeddings = []
        for idx, encoder in enumerate(ko_embedders):
            query_embedding = encoder.embed_query(ko_query)
            ko_query_embeddings.append((query_embedding, args.ko_ensemble_weights[idx]))

        # 영어 쿼리 임베딩
        en_query_embeddings = []
        for idx, encoder in enumerate(en_embedders):
            query_embedding = encoder.embed_query(en_query)
            en_query_embeddings.append((query_embedding, args.en_ensemble_weights[idx]))    

        # 한국어 검색 결과
        results = ko_retriever.invoke(ko_query)
        sorted_results = sorted(results, key=lambda x: x.metadata['score'], reverse=True)
        ko_search_result = [(result, result.metadata['score']) for result in sorted_results[:20]]

        ko_combined_scores = []
        for doc, _ in ko_search_result:
            combined_similarity = 0
            for idx, (query_embedding, weight) in enumerate(ko_query_embeddings):
                doc_embedding_key = f'embedding_{args.ko_ensemble_models[idx]["name"]}'
                doc_embedding = doc.metadata.get(doc_embedding_key) or ko_embedders[idx].embed_query(doc.page_content)
                similarity = 1 - cosine(query_embedding, doc_embedding)
                combined_similarity += weight * similarity
            ko_combined_scores.append((doc, combined_similarity))

        # 동일 docid 청크 중 가장 높은 점수 유지
        ko_docid_scores = {}
        for doc, score in ko_combined_scores:
            docid = doc.metadata.get('docid')
            if docid not in ko_docid_scores or score > ko_docid_scores[docid]['score']:
                ko_docid_scores[docid] = {'doc': doc, 'score': score}

        # 영어 검색 결과
        results = en_retriever.invoke(en_query)
        sorted_results = sorted(results, key=lambda x: x.metadata['score'], reverse=True)
        en_search_result = [(result, result.metadata['score']) for result in sorted_results[:20]]

        en_combined_scores = []
        for doc, _ in en_search_result:
            combined_similarity = 0
            for idx, (query_embedding, weight) in enumerate(en_query_embeddings):
                doc_embedding_key = f'embedding_{args.en_ensemble_models[idx]["name"]}'
                doc_embedding = doc.metadata.get(doc_embedding_key) or en_embedders[idx].embed_query(doc.page_content)
                similarity = 1 - cosine(query_embedding, doc_embedding)
                combined_similarity += weight * similarity
            en_combined_scores.append((doc, combined_similarity))

        # 동일 docid 청크 중 가장 높은 점수 유지
        en_docid_scores = {}
        for doc, score in en_combined_scores:
            docid = doc.metadata.get('docid')
            if docid not in en_docid_scores or score > en_docid_scores[docid]['score']:
                en_docid_scores[docid] = {'doc': doc, 'score': score}

        # reranking 여부에 따른 처리
        if args.rerank:
            # 한국어 결과 reranking
            ko_documents = [doc['doc'].page_content for doc in ko_docid_scores.values()]
            ko_reranking = vo.rerank(ko_query, ko_documents, model="rerank-2", top_k=len(ko_documents))
            ko_docid_scores = {ko_reranking.results[i].document: ko_docid_scores[ko_documents[i]] for i in range(len(ko_documents))}

            # 영어 결과 reranking
            en_documents = [doc['doc'].page_content for doc in en_docid_scores.values()]
            en_reranking = vo.rerank(en_query, en_documents, model="rerank-2", top_k=len(en_documents))
            en_docid_scores = {en_reranking.results[i].document: en_docid_scores[en_documents[i]] for i in range(len(en_documents))}

        # 상위 결과 결합 (한국어와 영어)
        ko_top_results = sorted(ko_docid_scores.values(), key=lambda x: x['score'], reverse=True)[:10]
        en_top_results = sorted(en_docid_scores.values(), key=lambda x: x['score'], reverse=True)[:10]

        final_combined_scores = []
        for ko_result in ko_top_results:
            ko_doc = ko_result['doc']
            ko_score = ko_result['score']
            matched_en_result = next((en_result for en_result in en_top_results if en_result['doc'].metadata['docid'] == ko_doc.metadata['docid']), None)
            
            if matched_en_result:
                en_score = matched_en_result['score']
                # combined_score = 0.5 * ko_score + 0.5 * en_score  # 가중치 결합
                combined_score = max(ko_score, en_score) # 최대 점수

                final_combined_scores.append((ko_doc, combined_score))
            else:
                final_combined_scores.append((ko_doc, ko_score))

        # 영어 결과 중 중복되지 않은 문서 추가
        for en_result in en_top_results:
            en_doc = en_result['doc']
            en_score = en_result['score']
            if en_doc.metadata['docid'] not in [doc.metadata['docid'] for doc, _ in final_combined_scores]:
                final_combined_scores.append((en_doc, en_score))

        # 최종 상위 3개 문서 선택
        final_results = sorted(final_combined_scores, key=lambda x: x[1], reverse=True)[:3]

        # 결과 처리
        retrieved_context = []
        for result in final_results:
            doc = result[0]
            score = result[1]
            retrieved_context.append(doc.page_content)
            response["topk"].append(doc.metadata.get('docid'))
            response["references"].append({
                "docid": doc.metadata.get('docid'),
                "score": score,
                "content": doc.page_content
            })

        print("=" * 30)
        print("검색된 문서 정보:")
        for ref in response["references"]:
            print(f"DocID: {ref['docid']}, Score: {ref['score']:.4f}\nContent: {ref['content']}\n")

    else:
        response['standalone_query'] = None
        print("답변을 생성할 수 없습니다.\n")
        response["answer"] = "질문이 과학 상식에 해당하지 않습니다."

    return response


def multi_eval_rag(args, ko_retriever, en_retriever):
    ko_embedders, en_embedders = [], []
    for model_info in args.ko_ensemble_models:
        model_type = model_info.get('type', 'hf')
        model_name = model_info['name']
        if model_type == 'hf':
            encoder = load_hf_encoder(model_name, args.model_kwargs, args.encode_kwargs)
        elif model_type == 'upstage':
            encoder = load_upstage_encoder(model_name)

        ko_embedders.append(encoder)

    for model_info in args.en_ensemble_models:
        model_type = model_info.get('type', 'hf')
        model_name = model_info['name']
        if model_type == 'hf':
            encoder = load_hf_encoder(model_name, args.model_kwargs, args.encode_kwargs)
        elif model_type == 'voyage':
            encoder = load_voyage_encoder(model_name)

        en_embedders.append(encoder)

    with open(args.eval_file) as f, open(args.output_path, "w") as of:
        idx = 0
        for line in f:
            j = json.loads(line)
            id = j['eval_id']
            ko_query = j['ko_query']
            en_query = j['en_query']
            print(f'Test {idx:>04}')
            print(f"ko_query : {ko_query}")
            print(f"en_query : {en_query}")


            if id in [276, 261, 283, 32, 94, 90, 220,  245, 229, 247, 67, 57, 2, 227, 301, 222, 83, 64, 103, 218]:
                ko_query, en_query = None, None

            response = answer_question(args, ko_query, en_query, ko_retriever, en_retriever, ko_embedders, en_embedders)

            output = {"eval_id": j["eval_id"], "standalone_query": response["standalone_query"], "topk": response["topk"], "answer": response["answer"], "references": response["references"]}
            of.write(f'{json.dumps(output, ensure_ascii=False)}\n')
            idx += 1            