from typing import Optional, List
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever


class Search(BaseModel):
    """Search over a document database."""

    query: str = Field(
        ...,
        description="Similarity search query applied to document content.",
    )
    docid: Optional[str] = Field(None, description="Document ID for precise document retrieval.")
    content_snippet: Optional[str] = Field(None, description="A snippet of the document content to search for.")


def load_query_analyzer(llm_model_name):
    system = """You are an expert at converting user questions into database queries. \
    You have access to a database of documents. Given a question, return a list of database queries optimized to retrieve the most relevant results.

    If there are acronyms or words you are not familiar with, do not try to rephrase them."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(Search)
    query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm

    return query_analyzer