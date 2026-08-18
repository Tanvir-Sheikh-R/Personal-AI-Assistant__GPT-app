from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from groq import RateLimitError, APIError, APIConnectionError
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition
from functools import partial
from prompts import SYSTEM_PROMPT
from tools import calculator, rag_tool, web_search


load_dotenv()
llm = ChatGroq(model='openai/gpt-oss-120b', temperature=0.2)

tools = [rag_tool, calculator, web_search]
llm_with_tools = llm.bind_tools(tools)



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


def chat_message(state: MessageState):
    # Copy instead of mutating the reducer-owned list from state directly.
    message = list(state['message'])
    if not any(isinstance(m, SystemMessage) for m in message):
        message = [SYSTEM_PROMPT] + message
    try:
        response = llm_with_tools.invoke(message)

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
graph.add_node('tools', ToolNode(tools, messages_key="message"))

graph.add_edge(START, 'chat_message')
graph.add_conditional_edges("chat_message",  partial(tools_condition, messages_key="message"),)
graph.add_edge('tools', 'chat_message')

checkpointer = InMemorySaver()
chat = graph.compile(checkpointer=checkpointer)

# response = chat.invoke(
#             {'message': '2+2'}, 
#             config={'configurable': {'thread_id': '--1--'}}, 
#             )


# response = chat.get_state({'configurable': {'thread_id': '--1--'}})
# print(response.values.get('message')[1].content)





# ----------------------- TavilySearch Web Search Tool CAll ---------------------------


# from langchain_tavily import TavilySearch
# web_search_tool = TavilySearch(
#     max_results=2,
#     topic="general",
# )

# @tool
# def web_search_tavily(query: str) -> str:
#     """Search the web for current, real-time, or general-knowledge information
#     NOT found in the user's uploaded documents.

#     Use this tool when the user asks about:
#     - Current events, news, or anything time-sensitive (prices, weather, scores, "latest", "today")
#     - General facts not likely to be in their uploaded documents
#     - Anything explicitly about the internet/web

#     Do NOT use this tool for:
#     - Questions about the user's own uploaded documents (use rag_tool instead)
#     - Simple math (use calculator instead)
#     - Greetings or small talk

#     Args:
#         query: A clear, standalone search query.

#     Returns:
#         A string summarizing the top search results with their sources.
#     """
#     try:
#         response = web_search_tool.invoke({"query": query})
#         results = response.get("results", [])
#         if not results:
#             return "No relevant search results found."

#         formatted = []
#         for r in results:
#             formatted.append(f"- {r.get('title', 'Untitled')}: {r.get('content', '')} (Source: {r.get('url', '')})")
#         print("\n".join(formatted))
#         return "\n".join(formatted)
    
#     except Exception as e:
#         return f"Error performing web search: {e}"


