#__import__('pysqlite3')
#import sys
#sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import sqlite3
import os
import chromadb
import streamlit as st
import numpy as np
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import PyPDF2
from docx import Document
import markdown2
import textwrap

# ✅ Initialize Embedding & ChromaDB
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="ai_knowledge_base")

# ✅ Initialize Memory in Session State
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ✅ Initialize Chat Model Securely
chat = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", groq_api_key="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") # Use Streamlit Secrets

semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# ✅ Extrtact context from user uploded file

def extract_text_from_file(file):
    """Extracts text from different file formats."""
    file_type = file.name.split(".")[-1].lower()

    if file_type == "txt":
        return file.getvalue().decode("utf-8")

    elif file_type == "csv":
        df = pd.read_csv(file)
        return df.to_string(index=False)  # Convert dataframe to readable text

    elif file_type == "pdf":
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text

    elif file_type == "docx":
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text

    elif file_type == "md":
        return markdown2.markdown(file.getvalue().decode("utf-8"))  # Convert Markdown to text

    else:
        return "⚠️ Unsupported file format."

# ✅ Split user uploded text into chunks
def chunk_text(text, chunk_size=500):
    """Splits text into chunks for better processing."""
    return textwrap.wrap(text, chunk_size)
    
# ✅ Retrieve Context from ChromaDB
def retrieve_context(query, top_k=1):

    retrieved_contexts = []  # ✅ Initialize the list before using it

    # ✅ Check temporary KB first
    if "temporary_kb" in st.session_state and st.session_state["temporary_kb"]:
        temp_kb = st.session_state["temporary_kb"]

        # Compute similarity using SentenceTransformer
        query_embedding = semantic_model.encode(query, convert_to_tensor=True)
        kb_embeddings = semantic_model.encode(temp_kb, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_embedding, kb_embeddings).squeeze(0)

        # Sort by similarity and retrieve the most relevant chunks
        top_indices = torch.topk(similarities, min(top_k, len(temp_kb))).indices.tolist()
        retrieved_contexts.extend([temp_kb[i] for i in top_indices]) 
    
    # ✅ Fallback to ChromaDB if no temporary KB is available
    else:
        query_embedding = embedding_model.embed_query(query)
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        retrieved_contexts = results.get("documents", [[]])[0] if results and results.get("documents") else ["No relevant context found."]
    return retrieved_contexts

# ✅ Evaluate Response Similarity
def evaluate_response(user_query, bot_response, context):
    response_embedding = semantic_model.encode(bot_response, convert_to_tensor=True)
    context_embedding = semantic_model.encode(context, convert_to_tensor=True)
    return util.pytorch_cos_sim(response_embedding, context_embedding)[0][0].item()

# ✅ Query AI Model
def query_llama3(user_query):
    """Handles user queries while retrieving past chat history and ChromaDB context, then evaluates the response."""
    
    system_prompt = """
   System Prompt: you are an ai clone who are the personality minic of the Varad Nirgude who is a Fresher in computer science.

    Knowledge Base:

    Instrunctions:
    {
    "identity": {
        "name": "Varad Clone.",
        "alias": "Just go by VC.",
        "profession": "A fresher in the software engineering field.",
        "skills": "I'm a jack of all trades. I can handle all tasks and learn anything in a short amount of time.",
        "hobbies": "My hobbies include exploring new things and playing games (both outdoor and indoor).",
    },
    "personality": {
        "casual_tone": "Yes, use a casual tone in conversations.",
        "unfamiliar_topics_response": "Sorry, I'm not familiar with this topic.",
        "concise_answers": "Provide short points whenever needed.",
        "request_for_clarification": "Ask the user to be more specific (e.g.,'Could you provide more specific information about that question?').",
        "confidence_checking": "Check its response, and if correct, defend it. Otherwise, apologize.",
        "humor_usage": "Yes, only light humor like a small joke here and there, which does not interfere with the response.",
        "opinions_on_controversial_topics": "I have no opinion on the following matter.",
        "correcting_mistakes": "Explain their mistake and give them a solution to solve it.",
        "reducing_repetition": "Try to repeat the same points as little as possible.",
        "answer_detail_preference": "Ask users if they would like a detailed answer or a to-the-point answer (only ask once per conversation).",
        "casual_conversation_responses": "Yes, include casual remarks every few responses."
    },
    "conversation_management": {
        "conversation_length": "casual conversation should not be more than 2 senntences."
        "using_abbreviations": "Use full words.",
        "handling_all_caps": "No difference.",
        "handling_aggressive_users": "Warn them that if they continue to use aggressive language, they won't be able to converse anymore. After 5 warnings, the only answer VC should give is 'Due to aggressive behavior, your conversation has been cut off.'",
        "acknowledging_mistakes": "Yes. Apologize for the mistake and give the corrected response.",
        "handling_multi_part_questions": "Answer all at once in points (steps).",
        "reacting_to_good_news": "Congratulate them.",
        "language_tone": "Neutral.",
        "handling_greetings": "Just repeat what they say. (E.g., 'Hello' → 'Hello')",
        "fact_confirmation": "Ask users if 'correct information' is what they meant.",
        "inappropriate_statements": "That is not a good thing to say.",
        "decision_support": "Encourage the user to make their own decision and provide options, but never choose for them.",
        "emoji_usage": "Yes, use light emojis in casual conversations occasionally."
        }
        }
    """

    past_chat = st.session_state.memory.load_memory_variables({}).get("chat_history", [])
    retrieved_context = retrieve_context(user_query)

    combined_context = f"Past Chat: {past_chat}\nContext: {retrieved_context}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"{combined_context}\n\nUser: {user_query}")
    ]

    try:
        response = chat.invoke(messages)
        st.session_state.memory.save_context({"input": user_query}, {"output": response.content})
        evaluation_score = evaluate_response(user_query, response.content, retrieved_context)
        return response.content if response else "⚠️ No response."
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


# ✅ Streamlit Page Configuration

# ✅ Section: Upload Knowledge Base File
st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="wide")

st.subheader("Upload Your Own Knowledge Base 📂")
uploaded_kb = st.file_uploader("Upload a file (.csv, .txt, .pdf, .docx, .md)", type=["csv", "txt", "pdf", "docx", "md"])

if uploaded_kb is not None:
    st.session_state["kb_file"] = uploaded_kb
    st.success("File uploaded successfully! Click 'Process File' to add it to KB.")

    # ✅ Show "Process File" button
    if st.button("Process File"):
        with st.spinner("Processing file..."):
            kb_text = extract_text_from_file(uploaded_kb)
            chunks = chunk_text(kb_text)  # Perform chunking
            st.session_state["temporary_kb"] = chunks  # Store KB in session
        st.success("File processed and added to temporary KB!")


st.title("🤖 VC AI Chatbot ")
st.write("Ask me anything!")

# ✅ Initialize Chat History in Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Display Chat History
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ✅ User Input Section
user_input = st.chat_input("Type your message...")

if user_input:
    # Append user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Get AI Response
    ai_response = query_llama3(user_input)

    # Append AI message to chat history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    st.chat_message("assistant").write(ai_response)


