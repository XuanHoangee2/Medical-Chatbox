from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from typing import TypedDict,List
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage,AIMessage
from langchain.schema import Document
from pydantic import BaseModel,Field

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
