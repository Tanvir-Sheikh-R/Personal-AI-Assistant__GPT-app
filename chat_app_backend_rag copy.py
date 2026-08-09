import os
from typing import Literal, TypedDict, Annotated

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_text_splitters  import RecursiveCharacterTextSplitter
import chromadb
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

load_dotenv()

CHROMA_DIR = "vectorstore"
EMBED_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_cache")
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
MMR_K = 6
MMR_LAMBDA = 0.7
NUM_EXPANSIONS = 3

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

_embeddings_instance = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            cache_folder=EMBED_CACHE,
        )
    return _embeddings_instance


class RagState(TypedDict):
    message: Annotated[list[BaseMessage], add_messages]
    doc_paths: list[str]
    expanded_queries: list[str]
    context_chunks: list[str]
    source_docs: list[str]
    generated_queries: list[str]

    intermediate_steps: list[dict]


def get_vectorstore(collection: str | None = None) -> Chroma:
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
        collection_name=collection or "documents",
    )


def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )


def _load_document(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(path).load()
    if ext == ".docx":
        return Docx2txtLoader(path).load()
    if ext in (".txt", ".md"):
        return TextLoader(path, encoding="utf-8").load()
    raise ValueError(f"Unsupported file type: {ext}")


def ingest(state: RagState) -> dict:
    vectorstore = get_vectorstore()
    splitter = get_splitter()

    for path in state["doc_paths"]:
        docs = _load_document(path)
        chunks = splitter.split_documents(docs)
        for chunk in chunks:
            chunk.metadata["source"] = os.path.basename(path)
        vectorstore.add_documents(chunks)

    return {"intermediate_steps": [{"ingest": state["doc_paths"]}]}


def expand_query(state: RagState) -> dict:
    user_message = state["message"][-1]
    question = user_message.content

    prompt = (
        f"Generate {NUM_EXPANSIONS} additional, distinct search queries that explore "
        "different facets or phrasings of the question below. "
        f"Return exactly {NUM_EXPANSIONS} queries, one per line, nothing else.\n"
        f"Question: {question}"
    )
    try:
        resposce = llm.invoke(prompt)
        extra = [
            line.strip()
            for line in resposce.content.splitlines()
            if line.strip()
        ][:NUM_EXPANSIONS]
    except Exception:
        extra = []

    queries = [question, *extra]
    seen: set[str] = set()
    deduped = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            deduped.append(q)

    return {
        "expanded_queries": deduped,
        "generated_queries": extra,
        "intermediate_steps": [{"expanded_queries": deduped}],
    }


def retrieve(state: RagState) -> dict:
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": MMR_K, "lambda_mult": MMR_LAMBDA},
    )

    chunks: list[str] = []
    sources: list[str] = []
    seen = set()

    for query in state["expanded_queries"]:
        for doc in retriever.invoke(query):
            content = doc.page_content
            if content in seen:
                continue
            seen.add(content)
            chunks.append(content)
            sources.append(doc.metadata.get("source", "unknown"))

    return {
        "context_chunks": chunks,
        "source_docs": sources,
        "intermediate_steps": [{"retrieved": len(chunks)}],
    }


def answer(state: RagState) -> dict:
    question = state["message"][-1].content
    context = state["context_chunks"]

    if not context:
        response = AIMessage(
            content=(
                "I couldn't find any relevant information in your uploaded documents. "
                "Please ask me something one of the documents covers."
            )
        )
    else:
        numbered = "\n\n".join(
            f"[{i+1}] {chunk}" for i, chunk in enumerate(context)
        )
        sources = ", ".join(sorted(set(state["source_docs"])))

        prompt = (
            f"Answer the question using ONLY the context below. "
            "If the context does not contain the answer, say so clearly. "
            "Reference which source document the information comes from.\n\n"
            f"Context:\n{numbered}\n\n"
            f"Sources available: {sources}\n\n"
            f"Question: {question}"
        )
        try:
            resposce = llm.invoke(prompt)
            content = resposce.content
            if len(sources) > 1 and sources != "unknown":
                content += f"\n\nSources: {sources}"
            response = AIMessage(content=content)
        except Exception as e:
            response = AIMessage(content=f"An error occurred while answering: {e}")

    return {"message": [response]}


graph = StateGraph(RagState)
graph.add_node("ingest", ingest)
graph.add_node("expand_query", expand_query)
graph.add_node("retrieve", retrieve)
graph.add_node("answer", answer)

graph.add_edge(START, "ingest")
graph.add_edge("ingest", "expand_query")
graph.add_edge("expand_query", "retrieve")
graph.add_edge("retrieve", "answer")
graph.add_edge("answer", END)

checkpointer = InMemorySaver()
rag_chat = graph.compile(checkpointer=checkpointer)


def _raw_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection("documents")


def list_indexed_docs() -> list[str]:
    try:
        col = _raw_collection()
        metas = col.get(include=["metadatas"])["metadatas"]
        names = []
        seen = set()
        for m in metas or []:
            src = m.get("source", "unknown")
            if src not in seen:
                seen.add(src)
                names.append(src)
        return names
    except Exception:
        return []


def ingest_documents(paths: list[str]) -> list[str]:
    vectorstore = get_vectorstore()
    splitter = get_splitter()
    saved = []
    for path in paths:
        docs = _load_document(path)
        chunks = splitter.split_documents(docs)
        for chunk in chunks:
            chunk.metadata["source"] = os.path.basename(path)
        vectorstore.add_documents(chunks)
        saved.append(os.path.basename(path))
    return saved


def delete_docs(doc_names: list[str]) -> bool:
    try:
        col = _raw_collection()
        data = col.get(include=["metadatas"])
        ids = data["ids"]
        metas = data["metadatas"]
        to_delete = []
        for i, m in enumerate(metas or []):
            if m.get("source") in doc_names:
                to_delete.append(ids[i])
        if to_delete:
            col.delete(ids=to_delete)
        return True
    except Exception:
        return False


def clear_all_docs() -> bool:
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            client.delete_collection("documents")
        except Exception:
            pass
        client.get_or_create_collection("documents")
        return True
    except Exception:
        return False




# ******************************************
#  from chat_app_backend_rag import rag_chat, list_indexed_docs, ingest_documents, delete_docs, clear_all_docs