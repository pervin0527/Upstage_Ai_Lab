import os
import copy
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

voyage_api_key = os.getenv('VOYAGE_API_KEY')
os.environ['VOYAGE_API_KEY'] = voyage_api_key

hf_token = os.getenv("HF_TOKEN")
huggingface_hub.login(hf_token)

from openai import OpenAI

from langchain.retrievers import EnsembleRetriever

from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Args
from data.data import load_document, chunking
from search.utils import retrieval_debug

from search.answer_processor import eval_rag
from search.ollama_processor import ollama_eval_rag

from dense_retriever.model import load_dense_model, load_sparse_model


def main(args: Args):
    os.makedirs("./outputs", exist_ok=True)

    print("\n프로세스 실행")

    print("+" * 30)
    print(f"사용언어 : {args.src_lang}")
    print("문서 로딩", end=" ")
    documents = load_document(path=args.doc_file_path)
    print(len(documents))
    print("-> 완료")

    if args.chunking:
        print("+" * 30)
        print(f"Document Chunking.")
        print(f"Method : {args.chunk_method}")
        print(f"Chunk Size : {args.chunk_size}. Chunk Overlap : {args.chunk_overlap}")
        
        documents = chunking(args, documents)
    print(len(documents))
    print("-> 완료")

    print("+" * 30)
    if args.doc_method == "dense":
        print(f"{args.encoder_method} Retriever 생성 중")
        retriever = load_dense_model(args, documents)

    elif args.doc_method == "sparse":
        print("BM25 Retriever 생성 중")
        retriever = load_sparse_model(documents, args.src_lang)

    elif args.doc_method == "ensemble":
        print("Ensemble Retriever 생성 중")
        sparse_retriever = load_sparse_model(documents, args.src_lang)
        sparse_retriever.k = 10
        
        dense_retriever = load_dense_model(args, documents).as_retriever(search_kwargs={"k": 10})

        retriever = EnsembleRetriever(
            retrievers=[sparse_retriever, dense_retriever],
            weights=args.retriever_weights,
            search_type="similarity_score_threshold" ## "mmr"
        )

    print("완료")

    print("+" * 30)
    print("검색 시작.")
    if args.llm_model == "gpt-4o":
        client = OpenAI()
        eval_rag(args, retriever, client)

    elif args.llm_model == "ollama":
        ollama_eval_rag(args, retriever)

    print("완료.")
    
if __name__ == "__main__":
    main(Args)
