# 🧠 RAG Chatbot: Build Fast with AI (LangChain + Groq)

Welcome to the **Build Fast with AI** project — an interactive **RAG (Retrieval-Augmented Generation)** chatbot that uses **LangChain**, **Groq LLM**, **FAISS**, and **Streamlit** to answer questions on the topic of **Word Embeddings**.

🔗 **Live App**: [https://ragbuildfastai.streamlit.app/](https://ragbuildfastai.streamlit.app/)  
📁 **Source Code**: [GitHub Repository](https://github.com/singhsrj/Build-Fast-with-AI)

---

## ✨ What It Does

This chatbot is designed to help users understand **Word Embeddings** by retrieving accurate, context-aware responses from the following source:

📄 [Blog Post by Lilian Weng: *From Word Embeddings to Document Distances*](https://lilianweng.github.io/posts/2017-10-15-word-embedding/)

### 🔧 Tech Stack

- 🦜 **LangChain**: Framework for chaining together LLMs and tools.
- 💡 **Groq API**: Blazing-fast inference using Groq’s large language models.
- 🧩 **FAISS**: Efficient similarity search to store and query embeddings.
- 📊 **HuggingFace Transformers**: Sentence Transformer model `all-MiniLM-L6-v2` used to generate dense vector embeddings.
- 📚 **RecursiveCharacterTextSplitter**: Used to chunk the source content into manageable pieces.
- 🎈 **Streamlit**: Frontend UI to interact with the chatbot.

---

## 🚀 How It Works

1. **Web Scraping**: Pulls the content from Lilian Weng's blog post.
2. **Chunking**: Splits content into semantically meaningful chunks using LangChain's `RecursiveCharacterTextSplitter`.
3. **Embeddings**: Uses HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` model to embed chunks.
4. **Vector Storage**: Stores the embeddings in a FAISS vector database.
5. **RAG Pipeline**: At query time, relevant chunks are retrieved and passed to the Groq LLM for response generation.

---

## 🛠️ Getting Started

### 1. Clone the Repository

git clone https://github.com/singhsrj/Build-Fast-with-AI.git
cd Build-Fast-with-AI

###2. Create & Activate a Virtual Environment

# Create venv (Linux/macOS)
python3 -m venv venv
source venv/bin/activate

# Or on Windows
python -m venv venv
venv\Scripts\activate

3. Install Dependencies
Make sure you have pip updated, then run:

pip install -r requirements.txt

💡 Deployment
This application is deployed and publicly accessible on Streamlit Community Cloud:

🌐 Live Demo: https://ragbuildfastai.streamlit.app/

To deploy your own version:

-Push your code to a public GitHub repository.

-Go to Streamlit Cloud, log in, and connect the repository.

-Set your Groq API key in the Streamlit secrets.

