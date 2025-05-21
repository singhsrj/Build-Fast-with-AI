import os
import streamlit as st
import bs4
from operator import itemgetter
# from chromadb import Client as ChromaClient
# from chromadb.config import Settings
from langchain_community.vectorstores import FAISS

# LangChain and components
from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq


from dotenv import load_dotenv
import os

# Load variables from .env into environment
load_dotenv()

# Get the API key
groq_api_key = os.getenv("GROQ_API_KEY")


# Streamlit app config
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG Chatbot - LangChain + Groq(llm)")

def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

# Cache heavy operations (loading & indexing)
@st.cache_resource
def build_rag_chain():
    # Load web page content
    loader = WebBaseLoader(
        web_paths=["https://lilianweng.github.io/posts/2017-10-15-word-embedding/"],
        bs_kwargs={"parse_only": bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))},
    )
    docs = loader.load()
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    # Embed with HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # vectorstore = Chroma.from_documents(splits, embedding=embeddings)

    # # Create retriever
    # retriever = vectorstore.as_retriever()

#     chroma_client = ChromaClient(Settings(
#     chroma_db_impl="duckdb+parquet",
#     persist_directory="./chroma_db"
# ))


    vectorstore = FAISS.from_documents(...)


    # Create retriever
    retriever = vectorstore.as_retriever()
    
    template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
    Generate multiple search queries related to: {question} \n
    Output (4 queries):"""
    prompt_rag_fusion = ChatPromptTemplate.from_template(template)


    # Connect to Groq's LLM
    llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)
    generate_queries = (
    prompt_rag_fusion 
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
    )
    
    
    retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion

    prompt = PromptTemplate.from_template("""
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, say you don't know. 
    Keep the answer concise and only use the context provided.
    Try to answer in atleast 100 words
    Question: {question} 

    Context: {context} 

    Answer:
    """)

    # Connect to Groq's LLM
    llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

    # Format retrieved context
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    content = "\n\n".join(doc.page_content for doc in docs)

    output_file = "llm_training_article.txt"

    if not os.path.exists(output_file):
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)
    
    # RAG pipeline
    final_rag_chain = (
    {"context": retrieval_chain_rag_fusion, 
     "question": itemgetter("question")} 
    | prompt
    | llm
    | StrOutputParser()
)

    return final_rag_chain

rag_chain = build_rag_chain()

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Initial message
with st.chat_message("assistant"):
    st.markdown("Ask me anything about Word Embeddings!")

# Chat input
question = st.chat_input("Ask a question...")
if question:
    # Display user message
    st.chat_message("user").markdown(question)

    # Run RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = rag_chain.invoke({"question": question})
            st.markdown(answer)

    # Save to chat history
    st.session_state.history.append({"question": question, "answer": answer})
