class MultiLingualArgs:
    llm_model = "ollama"
    
    eval_file = "../dataset/koen_eval.jsonl"
    ko_doc_file = "../dataset/gpt_contextual_retrieval_documents_v3.jsonl"
    en_doc_file = "../dataset/gpt_contextual_retrieval_documents_en_v3.jsonl"

    output_path = "./outputs/multi-lingual_output.csv"

    ## chunking
    chunking = False
    chunk_method = "recursive" ## recursive, semantic
    semantic_chunk_method = "upstage"
    chunk_size = 100
    chunk_overlap = 50

    ## "./index_files/upstage/CRV1"
    retriever_weights = [0.25, 0.25, 0.25, 0.25] ## [sparse, dense] [0.3, 0.7]

    doc_method = "ensemble" ## "sparse" or "dense" or "ensemble"
    ensemble_method = "rrf" ## "rrf", "cc"

    model_kwargs = {"device": "cuda:0"}
    encode_kwargs = {"normalize_embeddings": True, "clean_up_tokenization_spaces": True}
    
    upstage_model_name = "solar-embedding-1-large-passage"
    openai_model_name = "text-embedding-3-large"
    voyage_model_name = "voyage-3"

    ## reranker
    rerank = False

    ## query ensemble
    ko_ensemble_weights = [0.3, 0.7]
    ko_ensemble_models = [
        {'type': 'hf', 'name': "dragonkue/bge-m3-ko"},
        {'type': 'upstage', 'name': "solar-embedding-1-large-query"},
    ]

    en_ensemble_weights = [0.3, 0.7]
    en_ensemble_models = [
        {'type': 'voyage', 'name': "voyage-multilingual-2"},
        {'type': 'voyage', 'name': "voyage-large-2-instruct"},
    ]