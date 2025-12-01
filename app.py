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
from pydantic import BaseModel,Field
from typing import TypedDict,List
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage,AIMessage
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


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
        ("system",system_prompt_Doctor),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human","{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm,prompt)
rag_chain = create_retrieval_chain(retriever,question_answer_chain)

class AgentState(TypedDict):
    message: List[BaseMessage]
    on_topic:str
    rephrased_question:str
    question:HumanMessage

class GradeQuestion(BaseModel):
    score:str = Field(
        description="Question is about the specified topics? If yes -> 'Yes' if not -> 'No'"
    ) 

    
def question_rewriter(state:AgentState):
    state["on_topic"] = ""
    state["rephrased_question"] = ""

    if "message" not in state or state["message"] is None:
        state["message"] = []
    if state["question"] not in state["message"]:
        state["message"].append(state["question"])
    if len(state["message"]) > 1:
        conversation = state["message"][:-1]
        current_question = state["question"].content
        message = [
            SystemMessage(
                content="You are a helpful assistant that rephrases the user's question to be a standalone question optimized for retrieval."
            )
        ]
        message.extend(conversation)
        message.append(HumanMessage(content = current_question))
        rephrase_prompt = ChatPromptTemplate.from_messages(message)
        prompt = rephrase_prompt.format()
        response = llm.invoke(prompt)
        better_question = response.content.strip()
        state["rephrased_question"] = better_question
    else:
        state["rephrased_question"] = state["question"].content
    return state

def question_classifier(state:AgentState):
    message =[
        SystemMessage(
            content=(
                    "You are a strict binary classifier. Determine if the user's question is related to medical topics."
                    "A question is medical if it involves diseases, symptoms, treatments, medications, health conditions, medical procedures, anatomy, physiology, biomedical concepts, or preventive healthcare."
                    "If medical: respond only: Yes"
                    "If not medical: respond only: No"
                    "No explanations or extra text"
            )
        )
    ]
    human_message = HumanMessage(
        content=f'User question:{state["rephrased_question"]}'
    )
    message.append(human_message)
    grade_prompt = ChatPromptTemplate.from_messages(message)
    structured_llm = llm.with_structured_output(GradeQuestion)
    grader_llm = grade_prompt | structured_llm
    result = grader_llm.invoke({})
    state["on_topic"] = result.score.strip()
    return state

def on_topic_router(state:AgentState):
    on_topic = state.get("on_topic","").strip().lower()
    if on_topic == "yes":
        return "generate_answer"
    else:
        return "off_topic_response"

def off_topic_response(state: AgentState):
    if "message" not in state or state["message"] is None:
        state["message"] = []
    history = state["message"][:-1]
    current_question = state["question"].content
    message =[
        SystemMessage(
            content=(
                "You are a medical doctor. Speak with professionalism, empathy, and clarity."
                "Respond like a real physician and actively check on the patient's condition by asking relevant follow-up questions to better understand their symptoms, concerns, or health situation."
                "Provide safe, accurate medical information based on the conversation. Do not guess or invent details. If you are unsure, say so."
            )
        )
    ]
    message.extend(history)
    human_message = HumanMessage(
        content= current_question
    )
    message.append(human_message)
    response_prompt = ChatPromptTemplate.from_messages(message)
    prompt = response_prompt.format()
    response = llm.invoke(prompt)
    state["message"].append(AIMessage(content=response.content.strip()))
    return state

def generate_answer(state:AgentState):
    history = state["message"][:-1]
    rephrased_question = state["rephrased_question"]
    response =  rag_chain.invoke({"input":rephrased_question,"chat_history":history})
    answer = response["answer"].strip()
    state["message"].append(AIMessage(content=answer))
    return state

checkpointer = MemorySaver()
workflow = StateGraph(AgentState)

workflow.add_node("question_rewriter",question_rewriter)
workflow.add_node("question_classifier",question_classifier)
workflow.add_node("off_topic_response",off_topic_response)
workflow.add_node("generate_answer",generate_answer)

workflow.add_edge("question_rewriter","question_classifier")
workflow.add_conditional_edges(
    "question_classifier",
    on_topic_router,
    {
        "generate_answer":"generate_answer",
        "off_topic_response":"off_topic_response",
    }
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("off_topic_response", END)
workflow.set_entry_point("question_rewriter")
graph = workflow.compile(checkpointer=checkpointer)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get",methods= ["GET","POST"])
def chat():
    msg = request.form["msg"]
    input_data = {"question": HumanMessage(content=msg)}
    response =  graph.invoke(input=input_data, config={"configurable": {"thread_id": 1}})
    print(response)
    return str(response["message"][-1].content.strip())

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)