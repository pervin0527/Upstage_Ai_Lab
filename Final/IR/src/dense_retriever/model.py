from langchain_openai import OpenAIEmbeddings

from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def load_openai_encoder():
    encoder = OpenAIEmbeddings()

    return encoder

def load_hf_encoder(model_name, model_kwargs, encode_kwargs):
    encoder = HuggingFaceEmbeddings(model_name=model_name,
                                    model_kwargs=model_kwargs,
                                    encode_kwargs=encode_kwargs)
    
    return encoder