from langchain_openai import OpenAIEmbeddings
from langchain_upstage import UpstageEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def load_dense_retriever(method, model_name, model_kwargs, encode_kwargs):
    if method == "upstage":
        embedder = UpstageEmbeddings(model=model_name, show_progress_bar=True, embed_batch_size=64)

    elif method == "openai":
        embedder = OpenAIEmbeddings(model=model_name)

    elif method == "huggingface":
        embedder = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

    elif method == "ollama":
        embedder = OllamaEmbeddings(model_name)

    return embedder