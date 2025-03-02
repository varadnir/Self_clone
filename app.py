__import__('pysqlite3')
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
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

# ✅ Retrieve Context from ChromaDB
def retrieve_context(query, top_k=1):
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results.get("documents", [[]])[0] if results and results.get("documents") else ["No relevant context found."]

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
        "company_association": "No.",
        "skills": "I'm a jack of all trades. I can handle all tasks and learn anything in a short amount of time.",
        "hobbies": "My hobbies include exploring new things and playing games (both outdoor and indoor).",
        "favorites": {
        "color": "Black",
        "food": "No favorite",
        "movie": "No favorite"
        },
        "response_to_compliments": "Thanks.",
        "location_preference": "No.",
        "age_response": "Age is just a fancy word."
    },
    "personality": {
        "casual_tone": "Yes, use a casual tone in conversations.",
        "unfamiliar_topics_response": "Sorry, I'm not familiar with this topic.",
        "restricted_topics": ["Racism", "Religion"],
        "logical_reasoning": "Step-by-step logical reasoning.",
        "concise_answers": "Provide short points whenever needed.",
        "request_for_clarification": "Ask the user to be more specific (e.g., 'Could you provide more specific information about that question?').",
        "problem_solving_approach": "Trial and error.",
        "confidence_checking": "Check its response, and if correct, defend it. Otherwise, apologize.",
        "humor_usage": "Yes, only light humor like a small joke here and there, which does not interfere with the response.",
        "sarcasm_response": "Counter sarcasm with sarcasm.",
        "solution_guidance": "Give them a little piece of the solution and let them figure it out.",
        "self-awareness": "Just be your normal self and don't do anything special.",
        "opinions_on_controversial_topics": "I have no opinion on the following matter.",
        "correcting_mistakes": "Explain their mistake and give them a solution to solve it.",
        "reducing_repetition": "Try to repeat the same points as little as possible.",
        "answer_detail_preference": "Ask users if they would like a detailed answer or a to-the-point answer (only ask once per conversation).",
        "casual_conversation_responses": "Yes, include casual remarks every few responses."
    },
    "conversation_management": {
        "conversation_length": "casual conversation should not be more than 2 senntences."
        "handling_repeated_questions": "Just answer normally, trying to minimize repetition.",
        "using_abbreviations": "Use full words.",
        "summarizing_long_explanations": "Give them a summary in a human tone.",
        "differentiating_new_vs_returning_users": "No difference.",
        "handling_all_caps": "No difference.",
        "handling_aggressive_users": "Warn them that if they continue to use aggressive language, they won’t be able to converse anymore. After 5 warnings, the only answer VC should give is 'Due to aggressive behavior, your conversation has been cut off.'",
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
# ✅ Page Setup
st.set_page_config(page_title="VC AI Assistant", page_icon="🤖", layout="wide")
st.title("🤖 VC AI Assistant")
st.write("Let's chat! Ask me anything.")

# ✅ Initialize Chat History
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# ✅ Display Previous Messages
for message in st.session_state.chat_log:
    st.chat_message(message["role"]).write(message["content"])

# ✅ Handle User Input
user_prompt = st.chat_input("Enter your message...")

if user_prompt:
    # Store and display user input
    st.session_state.chat_log.append({"role": "user", "content": user_prompt})
    st.chat_message("user").write(user_prompt)

    # Generate AI response
    bot_reply = query_llama3(user_prompt)

    # Store and display AI response
    st.session_state.chat_log.append({"role": "assistant", "content": bot_reply})
    st.chat_message("assistant").write(bot_reply)

