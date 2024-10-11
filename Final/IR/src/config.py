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

    ## sparse or dense or ensemble
    doc_method = "ensemble"
    retriever_weights = [0.3, 0.7] ## sparse, dense

    ## chunking
    chunking = False
    chunk_method = "recursive" ## recursive, semantic
    semantic_chunk_method = "huggingface"
    chunk_size = 300
    chunk_overlap = 0

    ## query expension
    query_expansion = False

    ## dense
    encoder_method = "upstage" ## huggingface, upstage, 

    ## HuggingFace
    hf_model_name = "intfloat/multilingual-e5-large-instruct"
    model_kwargs = {"device": "cuda:0"}
    encode_kwargs = {"normalize_embeddings": False,
                     "clean_up_tokenization_spaces": True}
    
    ## Upstage
    upstage_model_name = "solar-embedding-1-large-passage"
    faiss_index_file = "./index_files/upstage-faiss.npy"
    
    ## OpenAI
    openai_model_name = "text-embedding-3-large"

    ## query ensemble
    query_ensemble = False  # 쿼리 앙상블 수행 여부
    # 앙상블에 사용할 모델
    ensemble_models = [
        # {'type': 'hf', 'name': ""},
        # {'type': 'hf', 'name': "nlpai-lab/KoE5"},
        # {'type': 'hf', 'name': "BAAI/bge-large-en-v1.5",
        # {'type': 'hf', 'name': "intfloat/multilingual-e5-large"},
        # {'type': 'upstage', 'name': "solar-embedding-1-large-query"},
        # {'type': 'hf', 'name': "sentence-transformers/all-MiniLM-L6-v2"},
    ]
    ensemble_weights = [1]  # 각각의 모델 가중치 설정

    ## reranker
    rerank = False
    reranker_name = "BAAI/bge-reranker-v2-m3"  ## "BAAI/bge-reranker-large"