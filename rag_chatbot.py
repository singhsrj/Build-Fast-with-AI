import os
import streamlit as st
import bs4

# LangChain and components
from langchain_core.prompts import PromptTemplate
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

# Cache heavy operations (loading & indexing)
@st.cache_resource
def build_rag_chain():
    # Load web page content
    loader = WebBaseLoader(
        web_paths=["https://lilianweng.github.io/posts/2021-09-25-train-large/"],
        bs_kwargs={"parse_only": bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))},
    )
    docs = loader.load()
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    # Embed with HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(splits, embedding=embeddings)

    # Create retriever
    retriever = vectorstore.as_retriever()

    prompt = PromptTemplate.from_template("""
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, say you don't know. 
    Keep the answer concise and only use the context provided.

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
    rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

    return rag_chain

rag_chain = build_rag_chain()

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Initial message
with st.chat_message("assistant"):
    st.markdown("Ask me anything about Training of Large Language Models on Multiple Gpu's!")

# Chat input
question = st.chat_input("Ask a question...")
if question:
    # Display user message
    st.chat_message("user").markdown(question)

    # Run RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = rag_chain.invoke(question)
            st.markdown(answer)

    # Save to chat history
    st.session_state.history.append({"question": question, "answer": answer})
