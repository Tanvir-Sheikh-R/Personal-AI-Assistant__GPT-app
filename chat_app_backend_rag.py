from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")

load_dotenv()
llm = ChatGroq(model='llama-3.1-8b-instant', temperature=0.2)

def Initialize_doc_to_VectorStore(pdf_file:list):
    pdfs = []
    for file in pdf_file:
        pdf_loader = PyPDFLoader(file)
        pdfs.extend(pdf_loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n\n", "\n\n", "\n", "  ", " ", ""]
    )
    chuncks = splitter.split_documents(pdfs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = Chroma.from_documents(chuncks, embedding=embeddings, collection_name="pdf_embeddings")

    return vector_store



def generate_output(query:str, vector_store):

    class SimilarQueries(BaseModel):
        similer_queries : list[str] = Field(description="2 diverse queries related to the original query")

    prompt_template = PromptTemplate(
        template="""
            Generate 2 more queries similar to the given query: {query}. 
            keep the content relevant to the context of the query and make sure the queries are 
            diverse and cover different aspects of the topic. Provide the queries in a list format.
        """,
        input_variables=['query']
    )

    prompt = prompt_template.format(query=query)

    structured_llm = llm.with_structured_output(SimilarQueries)
    result = structured_llm.invoke(prompt)
    final_prompt = [query] + result.similer_queries

    metadatas = []
    content = []
    for query in final_prompt:
        results = vector_store.max_marginal_relevance_search(query=query, k=3)
        content.extend([doc.page_content for doc in results])
        metadatas.extend([doc.metadata for doc in results])

    prompt_query = PromptTemplate(
        template="""
            Given the following context, answer the question: {query}\n
            Context: {context} """,
        input_variables=['query', 'context']
    )
    prompt = prompt_query.format(query=query, context=content, metadatas=metadatas)
    final_output = llm.invoke(prompt)

    return final_output.content



query = "Who is tanvir"

pdf_file = ['Tanvir AI-ML CV.pdf', 'Program-2026.pdf']
vector_store = Initialize_doc_to_VectorStore(pdf_file)
generated_output = generate_output(query, vector_store)

print("Generated Output:\n", generated_output)

