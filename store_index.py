from dotenv import load_dotenv
import os
from pinecone import Pinecone
from pinecone import ServerlessSpec
from src.helper import load_pdf_files,filter_to_minimal_docs,text_split,download_embeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
PINECONE_API = os.getenv("PINECONE_API")
GROQ_API = os.getenv("GROQ_API")

os.environ["PINECONE_API"] = PINECONE_API
os.environ["GROQ_API"] = GROQ_API

extracted_data = load_pdf_files("data")
mininal_docs = filter_to_minimal_docs(extracted_data)
texts_chunk = text_split(mininal_docs)

embedding = download_embeddings()

pinecone_api_key = PINECONE_API
pc = Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbox"

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws",region="us-east-1") 
    )

index = pc.Index(index_name)

vector_store = PineconeVectorStore(
    embedding=embedding,
    index = index
)

vector_store.add_documents(documents=texts_chunk)