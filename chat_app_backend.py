from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from groq import RateLimitError, APIError, APIConnectionError
from langchain_core.messages import HumanMessage, AIMessage


def get_summary_for_chatHead(user: str):
    prompt = f"""Generate a short, descriptive title for this conversation based on the user's message below. 
                Rules:
                - Maximum 5 words
                - No quotation marks, punctuation, or trailing periods
                - Capture the core topic or intent, not a generic summary
                - Do not include phrases like "Chat about" or "Conversation on"
                - Return ONLY the title text, nothing else
                User's message:{user}"""

    resposce = llm.invoke(prompt)
    return resposce.content


class MessageState(TypedDict):
    message : Annotated[list[BaseMessage], add_messages]


load_dotenv()
llm = ChatGroq(model='llama-3.1-8b-instant', temperature=0.2)

def chat_message(state: MessageState):
    message = state['message']
    try:
        response = llm.invoke(message)
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

# responce = chat.invoke(
#             {'message': '2+2'}, 
#             config={'configurable': {'thread_id': '--1--'}}, 
#             )


# responce = chat.get_state({'configurable': {'thread_id': '--1--'}})

# print(responce.values.get('message')[1].content)