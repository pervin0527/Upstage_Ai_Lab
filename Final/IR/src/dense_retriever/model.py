from langchain_openai import OpenAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from langchain_upstage import UpstageEmbeddings
from langchain_community.embeddings import SolarEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

def load_upstage_encoder(model_name):
    encoder = UpstageEmbeddings(model=model_name)
    # encoder = SolarEmbeddings(model=model_name)

    return encoder

def load_openai_encoder(model_name, dimensions):
    encoder = OpenAIEmbeddings(model=model_name,
                               dimensions=dimensions)

    return encoder

def load_hf_encoder(model_name, model_kwargs, encode_kwargs):
    encoder = HuggingFaceEmbeddings(model_name=model_name,
                                    model_kwargs=model_kwargs,
                                    encode_kwargs=encode_kwargs)
    
    return encoder

def load_hf_reranker(model_name, retriever):
    reranker = HuggingFaceCrossEncoder(model_name=model_name)
    compressor = CrossEncoderReranker(model=reranker, top_n=3)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever.as_retriever(search_kwargs={"k": 10}))

    return compression_retriever