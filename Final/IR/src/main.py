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

hf_token = os.getenv("HF_TOKEN")
huggingface_hub.login(hf_token)

from tqdm import tqdm
from openai import OpenAI
from langchain_community.chat_models import ChatOllama

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Args
from search.utils import retrieval_debug
from search.answer_processor import eval_rag
from search.ollama_processor import ollama_eval_rag, ollama_contextual_retrieval

from data.data import load_document, chunk_documents
from sparse_retriever.kiwi_bm25 import KiwiBM25Retriever
from dense_retriever.doc_processor import score_normalizer
from dense_retriever.model import load_hf_encoder, load_openai_encoder, load_upstage_encoder


def chunking(args, documents):
    if args.chunk_method == "recursive":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
    elif args.chunk_method == "semantic":
        if args.semantic_chunk_method == "huggingface":
            encoder = load_hf_encoder(args.hf_model_name, args.model_kwargs, args.encode_kwargs)
        elif args.semantic_chunk_method == "upstage":
            encoder = load_upstage_encoder(args.upstage_model_name)
        elif args.semantic_chunk_method == "openai":
            encoder = load_openai_encoder(args.openai_model_name)

        text_splitter = SemanticChunker(encoder)

    return text_splitter.split_documents(documents)


def load_sparse_model(documents):
    from konlpy.tag import Okt
    okt = Okt()
    def tokenize(text):
        tokens = okt.morphs(text)
        return tokens

    retriever = KiwiBM25Retriever.from_documents(documents)
    retriever = BM25Retriever.from_documents(documents, tokenizer=tokenize)
    
    return retriever


def load_dense_model(args, documents):
    if args.encoder_method == "huggingface":
        encoder = load_hf_encoder(args.hf_model_name, args.model_kwargs, args.encode_kwargs)
        print(f"Embedding Model : {args.hf_model_name}")

    elif args.encoder_method == "upstage":
        encoder = load_upstage_encoder(args.upstage_model_name)
        print(f"Embedding Model : {args.upstage_model_name}")

    elif args.encoder_method == "openai":
        encoder = load_openai_encoder(args.openai_model_name)
        print(f"Embedding Model : {args.openai_model_name}")

    index = faiss.IndexFlatL2(len(encoder.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=encoder,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        relevance_score_fn=score_normalizer
    )
    vector_store.add_documents(documents=documents)
    retriever = vector_store

    # faiss.write_index(index, f"./index_files/{args.encoder_method}-faiss.npy")
    print(f"FAISS 인덱스에 추가된 문서 수: {index.ntotal}")

    return retriever


def main(args: Args):
    os.makedirs("./outputs", exist_ok=True)

    print("\n프로세스 실행")

    print("+" * 30)
    print("문서 로딩", end=" ")
    documents = load_document(path=args.doc_file_path)
    print("-> 완료")

    if args.chunking:
        print("+" * 30)
        print(f"Document Chunking. Method : {args.chunk_method}")
        documents = chunking(args, documents)

    print("+" * 30)
    if args.doc_method == "dense":
        print(f"{args.encoder_method} Retriever 생성 중")
        retriever = load_dense_model(args, documents)

    elif args.doc_method == "sparse":
        print("BM25 Retriever 생성 중")
        retriever = load_sparse_model(documents)

    elif args.doc_method == "ensemble":
        print("Ensemble Retriever 생성 중")
        sparse_retriever = load_sparse_model(documents)
        sparse_retriever.k = 5
        
        dense_retriever = load_dense_model(args, documents).as_retriever(search_kwargs={"k": 5})
        # dense_retriever = FAISS.from_documents(documents, load_openai_encoder(args.openai_model_name)).as_retriever()

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
