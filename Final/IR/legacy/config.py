class Args:
    retrieval_debug = False
    llm_model = "ollama"
    
    src_lang = "ko"
    if src_lang == "en":
        eval_file_path = "../dataset/eval.jsonl" ## "../dataset/en_eval.jsonl" --> 성능이 별로임.
        doc_file_path = "../dataset/en_4.0_document.jsonl" ## "../dataset/processed_documents.jsonl"
    else:
        eval_file_path = "../dataset/processed_eval.jsonl"
        doc_file_path = "../dataset/gpt_contextual_retrieval_documents_v3.jsonl"

    ## query expansion
    query_expansion = False
    if query_expansion:
        eval_file_path = "../dataset/expanded_eval.jsonl"

    score_thres = 0

    ## UP-ER-QEN-CR
    output_path = "./outputs/UP-ER-QEN-CRV3.csv"

    ## chunking
    chunking = False
    chunk_method = "recursive" ## recursive, semantic
    semantic_chunk_method = "upstage"
    chunk_size = 100
    chunk_overlap = 50

    ## "./index_files/upstage/CRV1"
    faiss_index_file = "./index_files/upstage/CRV3_2"
    retriever_weights = [0.5, 0.5] ## [sparse, dense] [0.3, 0.7]

    doc_method = "ensemble" ## "sparse" or "dense" or "ensemble"
    encoder_method = "upstage" ## "huggingface", "upstage", "openai", "voyage"
    ensemble_method = "rrf" ## "rrf", "cc"

    hf_model_name = "intfloat/multilingual-e5-large-instruct"
    model_kwargs = {"device": "cuda:0"}
    encode_kwargs = {"normalize_embeddings": True, "clean_up_tokenization_spaces": True}
    
    upstage_model_name = "solar-embedding-1-large-passage"
    openai_model_name = "text-embedding-3-large"
    voyage_model_name = "voyage-3"

    ## reranker
    rerank = False

    ## query ensemble
    query_ensemble = True
    ensemble_weights = [0.2, 0.2, 0.6]  ## 각각의 모델 가중치 설정
    ensemble_models = [
        ## 앙상블에 사용할 모델
        {'type': 'hf', 'name' : "BAAI/bge-m3"},
        {'type': 'hf', 'name': "dragonkue/bge-m3-ko"},
        {'type': 'upstage', 'name': "solar-embedding-1-large-query"},
    ]

    ## multiple query
    multiple_query = False