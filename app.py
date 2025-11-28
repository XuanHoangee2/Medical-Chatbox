from flask import Flask, render_template, jsonify, request
from langchain.chains import retrieval
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from dotenv import load_dotenv
from src.prompt import *
import os
from src.helper import *
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain

app = Flask(__name__)

load_dotenv()
PINECONE_API = os.environ.get('PINECONE_API')
GROQ_API = os.environ.get('GROQ_API')
os.environ["PINECONE_API"] = PINECONE_API
os.environ["GROQ_API"] = GROQ_API

embedding = download_embeddings()

index_name = "medical-chatbox"
pinecone_api_key = PINECONE_API
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)
vector_store = PineconeVectorStore(
    embedding=embedding,
    index = index
)
retriever = vector_store.as_retriever(search_type = "similarity",search_kwargs={"k":3})
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=GROQ_API
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human","{input}"),
    ]
)
history = []
question_answer_chain = create_stuff_documents_chain(llm,prompt)
rag_chain = create_retrieval_chain(retriever,question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get",methods= ["GET","POST"])
def chat():
    global history
    msg = request.form["msg"]
    input = msg
    print(input)
    response =  rag_chain.invoke({"input":input,"chat_history":history})
    history = update_history(history,input,response["answer"])
    print("Response : ", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)