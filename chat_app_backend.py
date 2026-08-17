from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, ToolMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from groq import RateLimitError, APIError, APIConnectionError
from langchain_core.messages import HumanMessage, AIMessage
from sympy import sympify
from chat_app_backend_rag import generate_output, _get_vectorstore
import streamlit as st



@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = sympify(expression).evalf()
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def rag_tool(query : str):

    """Search and retrieve relevant information from the user's uploaded documents 
    or knowledge base.

    Use this tool whenever the user asks a question that could be answered by 
    specific facts, data, definitions, or content that may exist in their documents 
    — including questions about people, projects, numbers, dates, or anything not 
    considered common/general knowledge.

    Do NOT use this tool for:
    - Greetings or small talk (e.g. "hi", "how are you")
    - Simple math or logic questions
    - General knowledge the model already knows confidently
    - Follow-up questions that are just clarifying tone/formatting, not facts

    Args:
        query: A clear, standalone search query representing what the user wants 
        to find. Rephrase vague or pronoun-heavy user questions into a specific, 
        self-contained query (e.g., convert "what about its pricing?" into 
        "product pricing details").

    Returns:
        A string containing the most relevant retrieved passages, or a message 
        indicating no relevant documents were found.
    """


    vector_store = _get_vectorstore(st.session_state.get('kb_id', 'file_embeddings'))
    return generate_output(query, vector_store)





def get_summary_for_chatHead(user: str):
    prompt = f"""Generate a short, descriptive title for this conversation based on the user's message below. 
                Rules:
                - Maximum 5 words
                - No quotation marks, punctuation, or trailing periods
                - Capture the core topic or intent, not a generic summary
                - Do not include phrases like "Chat about" or "Conversation on"
                - Return ONLY the title text, nothing else
                User's message:{user}"""

    response = llm.invoke(prompt)
    return response.content


class MessageState(TypedDict):
    message : Annotated[list[BaseMessage], add_messages]


load_dotenv()
llm = ChatGroq(model='openai/gpt-oss-120b', temperature=0.2)
llm_with_tools = llm.bind_tools([rag_tool, calculator])


def chat_message(state: MessageState):
    # Copy instead of mutating the reducer-owned list from state directly.
    message = list(state['message'])
    try:
        content_get = llm_with_tools.invoke(message)

        if content_get.tool_calls:
            for tool_call in content_get.tool_calls:
                if tool_call["name"] == "rag_tool":
                    output = rag_tool.invoke(tool_call["args"])

                elif tool_call["name"] == "calculator":
                    output = calculator.invoke(tool_call["args"])
                else:
                    output = f"Error: unknown tool '{tool_call['name']}'"
                message.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))

            final_result = llm.invoke(message)
    
            response = final_result.content

        else:
            response = content_get.content

    except RateLimitError:
        response = AIMessage(
            content="I've hit the rate/token limit for this model right now. "
                    "Please wait a moment and try again."
        )
    except APIConnectionError:
        response = AIMessage(
            content="I'm having trouble connecting to the model service right now. "
                    "Please check your connection and try again."
        )
    except APIError as e:
        response = AIMessage(
            content=f"Something went wrong while generating a response: {e}"
        )
    except Exception as e:
        response = AIMessage(
            content=f"An unexpected error occurred: {e}"
        )
    return {'message': [response]}


graph = StateGraph(MessageState)
graph.add_node('chat_message', chat_message)
graph.add_edge(START, 'chat_message')
graph.add_edge('chat_message', END)

checkpointer = InMemorySaver()
chat = graph.compile(checkpointer=checkpointer)

# response = chat.invoke(
#             {'message': '2+2'}, 
#             config={'configurable': {'thread_id': '--1--'}}, 
#             )


# response = chat.get_state({'configurable': {'thread_id': '--1--'}})

# print(response.values.get('message')[1].content)