from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunking(dataset, chunk_size, chunk_overlap):
    documents = []
    for key in dataset.keys():
        documents.append(dataset[key])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )

    return text_splitter.split_documents(documents)