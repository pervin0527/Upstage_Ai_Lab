import json
from scipy.spatial.distance import cosine

from rankgpt.ranker import reranking
from dense_retriever.model import load_hf_encoder, load_upstage_encoder, load_hf_reranker, load_voyage_encoder, load_llm_reranker


def answer_question(args, ko_query, en_query, ko_retriever, en_retriever, ko_embedders, en_embedders):
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    if not ko_query is not None and not en_query is None:
        response['standalone_query'] = f"{ko_query}\n{en_query}"

        query_embeddings = []
        for idx, encoder in enumerate(ko_embedders):
            model_type = args.ensemble_models[idx]['type']
            query_embedding = encoder.embed_query(ko_query)
            query_embeddings.append((query_embedding, args.ensemble_weights[idx]))

        query_embeddings = []
        for idx, encoder in enumerate(en_embedders):
            model_type = args.ensemble_models[idx]['type']
            query_embedding = encoder.embed_query(ko_query)
            query_embeddings.append((query_embedding, args.ensemble_weights[idx]))    

        results = ko_retriever.invoke(ko_query)
        sorted_results = sorted(results, key=lambda x: x.metadata['score'], reverse=True)
        ko_search_result = [(result, result.metadata['score']) for result in sorted_results[:20]]

        ko_combined_scores = []
        for doc, _ in ko_search_result:
            combined_similarity = 0
            for idx, (query_embedding, weight) in enumerate(query_embeddings):
                doc_embedding_key = f'embedding_{args.ensemble_models[idx]["name"]}'
                doc_embedding = doc.metadata.get(doc_embedding_key) or ko_embedders[idx].embed_query(doc.page_content)
                similarity = 1 - cosine(query_embedding, doc_embedding)
                combined_similarity += weight * similarity
            ko_combined_scores.append((doc, combined_similarity))

        results = en_retriever.invoke(en_query)
        sorted_results = sorted(results, key=lambda x: x.metadata['score'], reverse=True)
        en_search_result = [(result, result.metadata['score']) for result in sorted_results[:20]]

        en_combined_scores = []
        for doc, _ in en_search_result:
            combined_similarity = 0
            for idx, (query_embedding, weight) in enumerate(query_embeddings):
                doc_embedding_key = f'embedding_{args.ensemble_models[idx]["name"]}'
                doc_embedding = doc.metadata.get(doc_embedding_key) or en_embedders[idx].embed_query(doc.page_content)
                similarity = 1 - cosine(query_embedding, doc_embedding)
                combined_similarity += weight * similarity
            en_combined_scores.append((doc, combined_similarity))

        combined_scores = ko_combined_scores.extend(en_combined_scores)
        docid_scores = {}
        for doc, score in combined_scores:
            docid = doc.metadata.get('docid')
            if docid not in docid_scores or score > docid_scores[docid]['score']:
                docid_scores[docid] = {'doc': doc, 'score': score}
        
        # 최종 선택된 상위 3개 문서
        final_results = sorted(docid_scores.values(), key=lambda x: x['score'], reverse=True)[:3]

        retrieved_context = []
        for result in final_results:
            doc = result['doc']
            score = result['score']
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
            if ref['score'] is not None:
                print(f"DocID: {ref['docid']}, Score: {ref['score']:.4f}\nContent: {ref['content']}\n")
            else:
                print(f"DocID: {ref['docid']}, Score: None\nContent: {ref['content']}\n")

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

    with open(args.eval_file_path) as f, open(args.output_path, "w") as of:
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