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

from config import Args

from data.data import load_document, chunk_documents

from sparse_retriever.kiwi_bm25 import KiwiBM25Retriever
from dense_retriever.doc_processor import score_normalizer
from dense_retriever.model import load_hf_encoder, load_openai_encoder, load_upstage_encoder

from search.utils import retrieval_debug
from search.answer_processor import eval_rag
from search.ollama_processor import ollama_eval_rag

def main(args: Args):
    os.makedirs("./outputs", exist_ok=True)

    print("\n프로세스 실행")
    client = OpenAI()

    print("+" * 30)
    print("문서 로딩", end=" ")
    documents = load_document(path=args.doc_file_path)
    print("-> 완료")
    
    print("+" * 30)
    if args.doc_method == "dense":
        print("임베딩 방식 : Dense(Vector Embedding)")
        if args.encoder_method == "huggingface":
            encoder = load_hf_encoder(args.hf_model_name, args.model_kwargs, args.encode_kwargs)
            print(f"임베딩 모델 : {args.encoder_method}, {args.hf_model_name}")

        elif args.encoder_method == "upstage":
            encoder = load_upstage_encoder(args.upstage_model_name)
            print(f"임베딩 모델 : {args.encoder_method}, {args.upstage_model_name}")

        elif args.encoder_method == "openai":
            encoder = load_openai_encoder(args.openai_model_name)
            print(f"임베딩 모델 : {args.encoder_method}, {args.openai_model_name}")
    
        index = faiss.IndexFlatL2(len(encoder.embed_query("hello world")))

        print("+" * 30)
        print("벡터 DB 생성 중", end=" ")
        
        vector_store = FAISS(
            embedding_function=encoder,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            relevance_score_fn=score_normalizer
        )
        vector_store.add_documents(documents=documents)
        retrieval = vector_store

        faiss.write_index(index, f"./index_files/{args.encoder_method}-faiss.npy")
        print(f"FAISS 인덱스에 추가된 문서 수: {index.ntotal}")
        print("완료")

    elif args.doc_method == "sparse":
        print("KiwiBM25 생성 중")
        retrieval = KiwiBM25Retriever.from_documents(documents)
        print("완료")

    if args.retrieval_debug:
        retrieval_debug("평균속도는 어떻게 계산하나요?", retrieval)

    print("+" * 30)
    print("검색 시작.")

    if args.llm_model == "gpt-4o":
        eval_rag(args, retrieval, client)
    elif args.llm_model == "ollama":
        ollama_eval_rag(args, retrieval)
    
    print("완료.")
    
if __name__ == "__main__":
    main(Args)
