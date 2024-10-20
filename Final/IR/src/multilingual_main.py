import os
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
from langchain_teddynote.retrievers import EnsembleRetriever, EnsembleMethod

from multilingual_config import MultiLingualArgs
from data.data import load_document, chunking

from dense_retriever.model import load_dense_model
from sparse_retriever.model import load_sparse_model

from search.multilingual_processor import multi_eval_rag

def main(args: MultiLingualArgs):
    os.makedirs("./outputs", exist_ok=True)

    print("\n프로세스 실행")
    print(f"Eval File : {args.eval_file}")
    print(f"Korean Document File : {args.ko_doc_file}")
    print(f"English Document File : {args.en_doc_file}")


    print("+" * 70)
    ko_documents = load_document(path=args.ko_doc_file)
    en_documents = load_document(path=args.en_doc_file)
    print(f"Korean Document Length : {len(ko_documents)}")
    print(f"English Document Length : {len(en_documents)}")

    if args.chunking:
        print("+" * 70)
        print(f"Document Chunking.")
        print(f"Method : {args.chunk_method}")
        print(f"Chunk Size : {args.chunk_size}. Chunk Overlap : {args.chunk_overlap}")
        documents = chunking(args, documents)

    print("+" * 70)
    sparse_retriever = load_sparse_model(ko_documents, "ko")
    sparse_retriever.k = 50
    
    args.encoder_method = "upstage"
    args.faiss_index_file = "./index_files/upstage/CRV3_2"
    dense_retriever = load_dense_model(args, ko_documents).as_retriever(search_kwargs={"k": 50})

    if args.ensemble_method == "rrf":
        ko_retriever = EnsembleRetriever(retrievers=[sparse_retriever, dense_retriever], method=EnsembleMethod.RRF)
    elif args.ensemble_method == "cc":
        ko_retriever = EnsembleRetriever(retrievers=[sparse_retriever, dense_retriever], method=EnsembleMethod.CC)

    print("+" * 70)
    sparse_retriever = load_sparse_model(en_documents, "en")
    sparse_retriever.k = 50

    args.encoder_method = "voyage"
    args.faiss_index_file = "./index_files/voyage/voyage3"
    dense_retriever = load_dense_model(args, en_documents).as_retriever(search_kwargs={"k": 50})

    if args.ensemble_method == "rrf":
        en_retriever = EnsembleRetriever(retrievers=[sparse_retriever, dense_retriever], method=EnsembleMethod.RRF)
    elif args.ensemble_method == "cc":
        en_retriever = EnsembleRetriever(retrievers=[sparse_retriever, dense_retriever], method=EnsembleMethod.CC)

    print("+" * 70)
    print("검색 시작.\n\n")
    multi_eval_rag(args, ko_retriever, en_retriever)
    
if __name__ == "__main__":
    main(MultiLingualArgs)