import os
import json
import faiss
import warnings
import huggingface_hub

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)

from dotenv import load_dotenv
load_dotenv("../keys.env")

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
from dense_retriever.model import load_hf_encoder, load_openai_encoder

from search.utils import retrieval_debug
from search.answer_processor import eval_rag

class Args:
    retrieval_debug = False

    doc_method = "dense"

    chunk_size=100
    chunk_overlap=5

    ## sparse
    tokenizer = "kiwi"

    ## dense
    encoder_method = "openai"

    ## HuggingFace
    hf_model_name = "intfloat/multilingual-e5-large-instruct" ## "jhgan/ko-sroberta-multitask"
    model_kwargs = {"device": "cuda:0"}
    encode_kwargs = {"normalize_embeddings": False,
                     "clean_up_tokenization_spaces": True}
    
    ## OpenAI
    openai_model_name = "text-embedding-3-small" ## "text-embedding-ada-002"
    dimensions = 512
    

def main(args:Args):
    documents = load_document(path="../dataset/processed_documents.jsonl")
    # documents = chunk_documents(documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    if args.doc_method == "dense":
        if args.encoder_method == "huggingface":
            encoder = load_hf_encoder(args.hf_model_name, args.model_kwargs, args.encode_kwargs)

        if args.encoder_method == "openai":
            encoder = load_openai_encoder(args.openai_model_name, args.dimensions)
            
        index = faiss.IndexFlatL2(len(encoder.embed_query("파이썬")))

        vector_store = FAISS(
            embedding_function=encoder,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            relevance_score_fn=score_normalizer
        )
        vector_store.add_documents(documents=documents)
        retrieval = vector_store

    elif args.doc_method == "sparse":
        retrieval = KiwiBM25Retriever.from_documents(documents)


    if args.retrieval_debug:
        retrieval_debug("평균속도는 어떻게 계산하나요?", retrieval)

    client = OpenAI()
    model = "gpt-4o"
    eval_rag("../dataset/eval.jsonl", "./sample_submission.csv", retrieval, client, model)
    
if __name__ == "__main__":
    main(Args)