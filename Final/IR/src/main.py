import os
import json
import faiss
import warnings
import huggingface_hub

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)

from dotenv import load_dotenv
load_dotenv("../keys.env")

upstage_api_key = os.getenv("UPSTAGE_API_KEY")
os.environ['UPSTAGE_API_KEY'] = upstage_api_key

openai_api_key = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = openai_api_key

hf_token = os.getenv("HF_TOKEN")
huggingface_hub.login(hf_token)

from openai import OpenAI

from langchain_community.vectorstores.faiss import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from data.data import load_document, chunk_documents

from sparse_retriever.kiwi_bm25 import KiwiBM25Retriever
from dense_retriever.doc_processor import score_normalizer
from dense_retriever.model import load_hf_encoder, load_openai_encoder, load_upstage_encoder

from search.utils import retrieval_debug
from search.answer_processor import eval_rag

class Args:
    retrieval_debug = False
    llm_model = "gpt-4o"
    doc_file_path = "../dataset/processed_documents.jsonl"
    eval_file_path = "../dataset/eval.jsonl"
    output_path = "./output.csv"

    doc_method = "dense"

    chunk_size=100
    chunk_overlap=5

    ## sparse
    tokenizer = "kiwi"

    ## dense
    encoder_method = "upstage"

    ## HuggingFace
    hf_model_name = "intfloat/multilingual-e5-large-instruct" ## "jhgan/ko-sroberta-multitask"
    model_kwargs = {"device": "cuda:0"}
    encode_kwargs = {"normalize_embeddings": False,
                     "clean_up_tokenization_spaces": True}
    
    ## Upstage
    upstage_model_name = "solar-embedding-1-large-passage"
    
    ## OpenAI
    openai_model_name = "text-embedding-3-small" ## "text-embedding-ada-002"
    dimensions = 512

    ## reranker
    rerank = False
    reranker_name = "BAAI/bge-reranker-v2-m3"
    

def main(args:Args):
    client = OpenAI()

    documents = load_document(path=args.doc_file_path)
    # documents = chunk_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    if args.doc_method == "dense":
        if args.encoder_method == "huggingface":
            encoder = load_hf_encoder(args.hf_model_name, args.model_kwargs, args.encode_kwargs)

        elif args.encoder_method == "upstage":
            encoder = load_upstage_encoder(args.upstage_model_name)

        elif args.encoder_method == "openai":
            encoder = load_openai_encoder(args.openai_model_name, args.dimensions)
            
        index = faiss.IndexFlatL2(len(encoder.embed_query("파이썬")))

        print("벡터 DB 생성 중")
        vector_store = FAISS(
            embedding_function=encoder,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            relevance_score_fn=score_normalizer
        )
        vector_store.add_documents(documents=documents)
        retrieval = vector_store
        print("완료")

    elif args.doc_method == "sparse":
        print("KiwiBM25 생성 중")
        retrieval = KiwiBM25Retriever.from_documents(documents)
        print("완료")

    if args.retrieval_debug:
        retrieval_debug("평균속도는 어떻게 계산하나요?", retrieval)

    print("검색 시작.")
    eval_rag(args, retrieval, client)
    print("완료.")
    
if __name__ == "__main__":
    main(Args)