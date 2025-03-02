Name : Varad Nirgude

Lnkedin : https://www.linkedin.com/in/varad-nirgude-88a973234

email : varadnirgude@gmail.com

# VC AI Chatbot

## 🤖 Overview
VC AI Chatbot is an intelligent assistant built using **LangChain**, **ChromaDB**, and **Streamlit**. It utilizes **LLM-based conversation handling** with **retrieval-augmented generation (RAG)** to provide contextual and meaningful responses. The chatbot maintains a **memory buffer** for past interactions and fetches **relevant knowledge** from a **vector database** for improved responses.

## 🛠️ Tech Stack
### Backend
- **LLM Model:** `llama3-70b-8192` (via `LangChain-Groq`)
- **Vector Database:** `ChromaDB` (stores embeddings for context retrieval)
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (via `LangChain-HuggingFace`)
- **Memory Management:** `st.session_state` (for persisting chat history across interactions in Streamlit.)
- **Text Similarity:** `SentenceTransformers` (for semantic response evaluation)
- **File Processing:** `PyPDF2` (for future document processing)
- **Server:** `Streamlit` (UI & API for chatbot interaction)

### Frontend
- **Framework:** `Streamlit`
- **UI Features:** Message alignment, chat input, and memory persistence

## ⚙️ Setup Guide
### 1️⃣ Install Dependencies
Create a virtual environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

#### `requirements.txt`
```txt
numpy>=1.21.0
PyPDF2>=3.0.0
langchain>=0.1.0
chromadb>=0.4.24
langchain-huggingface>=0.1.0
streamlit>=1.25.0
groq>=0.3.0
langchain_groq
sentence-transformers
```

### 2️⃣ Run the Chatbot
Run the Streamlit app:
```bash
streamlit run app.py
```

## 🎯 Features
### ✅ Conversational Memory
- Uses **LangChain's `ConversationBufferMemory`** to track past conversations.
- Retrieves and includes chat history in responses.

### ✅ Retrieval-Augmented Generation (RAG)
- **Vector database:** `ChromaDB` stores knowledge chunks.
- **Embedding model:** `all-MiniLM-L6-v2` generates embeddings.
- **Retrieval:** Queries **contextually relevant** information for better responses.

### ✅ Semantic Response Evaluation
- Uses `SentenceTransformers` to compare bot responses with relevant context.
- Measures **cosine similarity** to assess response relevance.

### ✅ Chat UI with Streamlit
- User messages aligned **right** (👤).
- Bot messages aligned **left** (🤖).
- Chat history persists across interactions.

## 🧩 Chunking Strategy
To **store and retrieve knowledge efficiently**, the chatbot splits text into **small chunks** before embedding them in the vector database.

### Chunking Method: Fixed-Length Chunks
- Splits text into **500-character chunks**.
- Ensures **better semantic retrieval** when matching queries.
- Uses **LangChain's `RecursiveCharacterTextSplitter`** for chunking.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_text("Your document or text here")
```

## 🗄️ Vector Database: ChromaDB
- **Why?** Lightweight, fast, and supports `HuggingFace` embeddings.
- **Storage:** Uses **persistent mode** (`PersistentClient`).
- **Querying:** Finds top-k most relevant chunks using cosine similarity.

```python
import chromadb

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="ai_knowledge_base")
```

## 🧠 Memory & Context Retrieval
### Memory Handling
Stores past chat logs in **LangChain’s `ConversationBufferMemory`**:
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
```

### Retrieving Context from ChromaDB
```python
def retrieve_context(query, top_k=1):
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results.get("documents", [[]])[0] if results else ["No relevant context found."]
```

## 🗣️ AI Response Generation
### Invoking LLaMA 3
The chatbot **retrieves chat history & context**, then sends a structured prompt to the **LLM**:
```python
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

chat = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", groq_api_key="your_api_key")

messages = [
    SystemMessage(content="Your system prompt"),
    HumanMessage(content=f"{combined_context}\n\nUser: {user_query}")
]

response = chat.invoke(messages)
```

## 🎭 Streamlit UI
### Main UI Features
- **Chat history is displayed persistently**
- **Messages are aligned:**  
  - 🤖 **Bot** → **Left**
  - 👤 **User** → **Right**

### UI Code:
```python
import streamlit as st

# ✅ Page Setup
st.set_page_config(page_title="VC AI Assistant", page_icon="🤖", layout="wide")
st.title("🤖 VC AI Assistant")
st.write("Let's chat! Ask me anything.")

# ✅ Initialize Chat History
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# ✅ Display Previous Messages with Left/Right Alignment
for message in st.session_state.chat_log:
    if message["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.write(message["content"])

# ✅ Handle User Input
user_prompt = st.chat_input("Enter your message...")

if user_prompt:
    # Store and display user input on the **right** with a user icon
    with st.chat_message("user", avatar="👤"):
        st.write(user_prompt)
    st.session_state.chat_log.append({"role": "user", "content": user_prompt})

    # Generate AI response
    bot_reply = query_llama3(user_prompt)

    # Store and display AI response on the **left** with a bot icon
    with st.chat_message("assistant", avatar="🤖"):
        st.write(bot_reply)
    st.session_state.chat_log.append({"role": "assistant", "content": bot_reply})
```

## 🚀 Summary of the Project
| Feature | Implementation |
|---------|---------------|
| **LLM Model** | LLaMA 3 (via Groq) |
| **Embedding Model** | `all-MiniLM-L6-v2` |
| **Vector Database** | ChromaDB |
| **Chunking Method** | Fixed-length (256 chars) |
| **Memory** | `ConversationBufferMemory` |
| **Semantic Evaluation** | SentenceTransformers (`cosine similarity`) |
| **UI** | Streamlit |
| **Message Alignment** | User → **Right** (👤), Bot → **Left** (🤖) |

## 💡 Next Steps & Future Enhancements
- 🔹 **Support for PDFs & Docs**
- 🔹 **Improved Prompt Engineering**
- 🔹 **Fine-Tuning Chat Memory**
- 🔹 **Web Deployment**

---

This README provides everything you need to understand, build, and improve your **VC AI Chatbot**. 🚀

