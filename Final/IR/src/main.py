import os
import huggingface_hub
from dotenv import load_dotenv
load_dotenv("../keys.env")

hf_token = os.getenv("HF_TOKEN")
huggingface_hub.login(hf_token)

upstage_api_key = os.getenv("UPSTAGE_API_KEY")
openai_api_key = os.getenv('OPENAI_API_KEY')
voyage_api_key = os.getenv('VOYAGE_API_KEY')
os.environ['UPSTAGE_API_KEY'] = upstage_api_key
os.environ['OPENAI_API_KEY'] = openai_api_key
os.environ['VOYAGE_API_KEY'] = voyage_api_key

from utils.config_utils import load_config
from data.dataloader import load_doc_dataset
from chunking.chunk import chunking

from retriever.loader import load_retriever
from search.searching import start_rag

def main():
    cfg = load_config("./config.yaml")

    dataset = load_doc_dataset(cfg)
    if cfg['chunking']['apply']:
        chunk_size, chunk_overlap = cfg['chunking']['chunk_size'], cfg['chunking']['chunk_overlap']
        dataset['documents'] = chunking(dataset['documents'], chunk_size, chunk_overlap)

    retriever = load_retriever(cfg, dataset)
    start_rag(cfg, retriever, dataset)

    
if __name__ == "__main__":
    main()