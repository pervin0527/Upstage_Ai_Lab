from langchain_community.retrievers import BM25Retriever

def load_sparse_model(documents, lang):

    if lang == "ko":
        from konlpy.tag import Okt
        okt = Okt()
        def tokenize(text):
            tokens = okt.morphs(text)
            return tokens
        
    elif lang == "en":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('voyageai/voyage-3')

        def tokenize(text):
            tokens = tokenizer.tokenize(text)
            return tokens

    retriever = BM25Retriever.from_documents(documents, tokenizer=tokenize)
    
    return retriever