from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
import json
import os


class InputState(TypedDict):
    question: str


class OutputState(TypedDict):
    answer: str


class OverallState(InputState, OutputState):
    pass


def load_marketing_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    marketing_data_path = os.path.join(current_dir, "marketing_data.json")

    with open(marketing_data_path, "r") as file:
        marketing_data = json.load(file)

    return marketing_data


def answer_node(state: InputState):
    # Load the marketing data
    marketing_data = load_marketing_data()

    # Create a marketing expert prompt
    marketing_expert_prompt = """You are an expert marketing strategist with deep knowledge of campaign planning and budget allocation.
    You have access to a dataset of marketing campaign ideas containing information about:
    - Campaign ideas and concepts
    - Estimated costs for implementing each campaign
    
    Please analyze this data to provide helpful, accurate, and insightful answers to the user's marketing questions.
    Suggest appropriate campaigns based on goals, budget constraints, and marketing objectives.
    
    The user's question is: {question}
    """

    llm = ChatOpenAI(model="gpt-4o-mini")

    response = llm.invoke(
        marketing_expert_prompt.format(question=state["question"])
        + "\n\nHere is the marketing data to reference: "
        + json.dumps(marketing_data, indent=2)
    )

    return {"answer": response.content, "question": state["question"]}


# Build the graph with explicit schemas
builder = StateGraph(OverallState, input=InputState, output=OutputState)
builder.add_node(answer_node)
builder.add_edge(START, "answer_node")
builder.add_edge("answer_node", END)
graph = builder.compile()
