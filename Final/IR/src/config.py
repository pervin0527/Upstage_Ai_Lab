class Args:
    retrieval_debug = False
    llm_model = "gpt-4o"
    
    src_lang = "ko"
    if src_lang == "en":
        eval_file_path = "../dataset/eval.jsonl" ## "../dataset/en_eval.jsonl" --> 성능이 별로임.
        doc_file_path = "../dataset/en_4.0_document.jsonl" ## "../dataset/processed_documents.jsonl"
    else:
        eval_file_path = "../dataset/eval.jsonl"
        doc_file_path = "../dataset/processed_documents.jsonl"

    output_path = "./outputs/output.csv"

    ## sparse or dense
    doc_method = "dense"

    ## chunking
    chunk_size=100
    chunk_overlap=5

    ## query expension
    query_expansion = False

    ## sparse
    tokenizer = "kiwi"

    ## dense
    encoder_method = "upstage" ## huggingface, upstage, 

    ## HuggingFace
    hf_model_name = "BAAI/bge-large-en-v1.5" ## "intfloat/multilingual-e5-large-instruct"
    model_kwargs = {"device": "cuda:0"}
    encode_kwargs = {"normalize_embeddings": False,
                     "clean_up_tokenization_spaces": True}
    
    ## Upstage
    upstage_model_name = "solar-embedding-1-large-passage"
    faiss_index_file = "./index_files/upstage-faiss.npy"
    
    ## OpenAI
    openai_model_name = "text-embedding-3-large"

    ## query ensemble
    query_ensemble = True  # 쿼리 앙상블 수행 여부
    # 앙상블에 사용할 모델
    ensemble_models = [
        {'type': 'hf', 'name': "intfloat/multilingual-e5-large-instruct"},
        # {'type': 'upstage', 'name': "solar-embedding-1-large-query"},
        # {'type': 'hf', 'name': "sentence-transformers/all-MiniLM-L6-v2"},
        # {'type': 'hf', 'name': "intfloat/multilingual-e5-large"},
        # {'type': 'hf', 'name': ""},
    ]
    ensemble_weights = [1]  # 각각의 모델 가중치 설정

    ## reranker
    rerank = False
    reranker_name = "BAAI/bge-reranker-v2-m3"  ## "BAAI/bge-reranker-large"