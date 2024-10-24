import os
import faiss

from langchain_community.vectorstores.faiss import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_compressors.rankllm_rerank import RankLLMRerank

from langchain_openai import OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings, VoyageAIRerank
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from langchain_cohere import CohereRerank
from langchain_upstage import UpstageEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

from dense_retriever.doc_processor import score_normalizer

def load_voyage_encoder(model_name):
    encoder = VoyageAIEmbeddings(model=model_name, show_progress_bar=True, batch_size=64)

    return encoder

def load_ollama_encoder(model_name):
    encoder = OllamaEmbeddings(model_name)

    return encoder

def load_upstage_encoder(model_name):
    encoder = UpstageEmbeddings(model=model_name, show_progress_bar=True, embed_batch_size=64)
    return encoder

def load_openai_encoder(model_name):
    encoder = OpenAIEmbeddings(model=model_name)

    return encoder

def load_hf_encoder(model_name, model_kwargs, encode_kwargs):
    encoder = HuggingFaceEmbeddings(model_name=model_name,
                                    model_kwargs=model_kwargs,
                                    encode_kwargs=encode_kwargs)
    
    return encoder

def load_cohere_reranker(model_name, retriever):
    compressor = CohereRerank(model=model_name)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

    return compression_retriever

def load_hf_reranker(model_name, retriever):
    reranker = HuggingFaceCrossEncoder(model_name=model_name)
    compressor = CrossEncoderReranker(model=reranker, top_n=3)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

    return compression_retriever


def load_voyage_reranker(model_name, retriever):
    compressor = VoyageAIRerank(model=model_name, top_k=3)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

    return compression_retriever


def load_dense_model(args, documents):
    if (not args.faiss_index_file is None) and os.path.exists(args.faiss_index_file):
        # 저장된 인덱스와 관련 데이터 불러오기
        print(f"FAISS 인덱스 로드 중: {args.faiss_index_file}")
        if args.encoder_method == "huggingface":
            encoder = load_hf_encoder(args.hf_model_name, args.model_kwargs, args.encode_kwargs)
        elif args.encoder_method == "upstage":
            encoder = load_upstage_encoder(args.upstage_model_name)
        elif args.encoder_method == "openai":
            encoder = load_openai_encoder(args.openai_model_name)
        elif args.encoder_method == "voyage":
            encoder = load_voyage_encoder(args.voyage_model_name)

        # load_local 메서드를 사용하여 인덱스, docstore, index_to_docstore_id 불러오기
        retriever = FAISS.load_local(args.faiss_index_file, encoder, allow_dangerous_deserialization=True)
        print(f"FAISS 인덱스 로드 완료, 총 문서 수: {retriever.index.ntotal}")
    
    else:
        folder_path = f"./index_files/{args.encoder_method}"
        
        if not args.chunking:
            args.chunk_size = 0
            args.chunk_overlap = 0

        if args.encoder_method == "huggingface":
            encoder = load_hf_encoder(args.hf_model_name, args.model_kwargs, args.encode_kwargs)
            folder_name = args.hf_model_name.replace("/", "-")
            folder_path = f"{folder_path}/{folder_name}-cs{args.chunk_size}-co{args.chunk_overlap}"
            print(f"Embedding Model : {args.hf_model_name}")
            print(f"saved at {folder_path}")

        elif args.encoder_method == "upstage":
            encoder = load_upstage_encoder(args.upstage_model_name)
            folder_path = f"{folder_path}/{args.upstage_model_name}-cs{args.chunk_size}-co{args.chunk_overlap}"
            print(f"Embedding Model : {args.upstage_model_name}")
            print(f"saved at {folder_path}")

        elif args.encoder_method == "openai":
            encoder = load_openai_encoder(args.openai_model_name)
            folder_path = f"{folder_path}/{args.openai_model_name}-cs{args.chunk_size}-co{args.chunk_overlap}"
            print(f"Embedding Model : {args.openai_model_name}")
            print(f"saved at {folder_path}")
        
        elif args.encoder_method == "voyage":
            encoder = load_voyage_encoder(args.voyage_model_name)
            folder_path = f"{folder_path}/{args.voyage_model_name}-cs{args.chunk_size}-co{args.chunk_overlap}"
            print(f"Embedding Model : {args.voyage_model_name}")
            print(f"saved at {folder_path}")

        # 인덱스 생성
        # index = faiss.IndexFlatL2(len(encoder.embed_query("hello world")))
        # vector_store = FAISS(
        #     embedding_function=encoder,
        #     index=index,
        #     docstore=InMemoryDocstore(),
        #     index_to_docstore_id={},  # 빈 딕셔너리로 초기화
        #     relevance_score_fn=score_normalizer
        # )
        # vector_store.add_documents(documents=documents)
        vector_store = FAISS.from_documents(documents, encoder)

        # 인덱스 및 관련 데이터 저장
        os.makedirs(folder_path, exist_ok=True)
        vector_store.save_local(folder_path)
        print(f"FAISS 인덱스 저장 완료: {folder_path}")

        retriever = vector_store

    return retriever