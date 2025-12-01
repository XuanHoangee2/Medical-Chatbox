from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage

# Load pdf file
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    documents = loader.load()
    return documents

# Data clean
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs:List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata = {"source":src} 
            )
        )
    return minimal_docs

# Text chunk
def text_split(mininal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20,
    )
    texts_chunk = text_splitter.split_documents(mininal_docs)
    return texts_chunk

# Dowload model
def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name = model_name,
    )
    return embeddings
 
def update_history(history, user_input, assistant_output):
    # Add the new user message
    history.append(HumanMessage(content=user_input))

    # Add the assistant message
    history.append(AIMessage(content=assistant_output))

    return history
    