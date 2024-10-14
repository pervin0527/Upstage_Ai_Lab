class Args:
    retrieval_debug = False
    llm_model = "ollama"
    
    src_lang = "ko"
    if src_lang == "en":
        eval_file_path = "../dataset/eval.jsonl" ## "../dataset/en_eval.jsonl" --> 성능이 별로임.
        doc_file_path = "../dataset/en_4.0_document.jsonl" ## "../dataset/processed_documents.jsonl"
    else:
        eval_file_path = "../dataset/eval.jsonl"
        doc_file_path = "../dataset/processed_documents.jsonl"

    output_path = "./outputs/output.csv"

    doc_method = "dense" ## "sparse" or "dense" or "ensemble"
    encoder_method = "upstage" ## "huggingface", "upstage", "openai", "voyage"
    faiss_index_file = "./index_files/upstage/solar-embedding-1-large-passage"
    retriever_weights = [0.3, 0.7] ## [sparse, dense]

    ## HuggingFace
    hf_model_name = "intfloat/multilingual-e5-large-instruct"
    model_kwargs = {"device": "cuda:0"}
    encode_kwargs = {"normalize_embeddings": False,
                     "clean_up_tokenization_spaces": True}
    
    ## Upstage
    upstage_model_name = "solar-embedding-1-large-passage"
    
    ## OpenAI
    openai_model_name = "text-embedding-3-large"

    ## Voyage
    voyage_model_name = "voyage-3"

    ## chunking
    chunking = True
    chunk_method = "recursive" ## recursive, semantic
    semantic_chunk_method = "upstage"
    chunk_size = 100
    chunk_overlap = 50

    ## query expension
    query_expansion = False

    ## query ensemble
    query_ensemble = True
    ## 앙상블에 사용할 모델
    ensemble_models = [
        {'type': 'hf', 'name': "BAAI/bge-m3"},
        {'type': 'hf', 'name': "intfloat/multilingual-e5-large"},
        {'type': 'upstage', 'name': "solar-embedding-1-large-query"},

        # {'type': 'hf', 'name': ""},
        # {'type': 'voyage', 'name': "voyage-3"},
        # {'type': 'hf', 'name': "nlpai-lab/KoE5"},
        # {'type': 'hf', 'name': "BAAI/bge-large-en-v1.5"},
        # {'type': 'hf', 'name': "sentence-transformers/all-MiniLM-L6-v2"},
    ]
    ensemble_weights = [0.3, 0.35, 0.35]  ## 각각의 모델 가중치 설정

    ## reranker
    rerank = False
    rerank_method = "voyage" ## "voyage", "huggingface"
    reranker_name = "rerank-2"  ## "BAAI/bge-reranker-large", "BAAI/bge-reranker-v2-m3", "rerank-2"