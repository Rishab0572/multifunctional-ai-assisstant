import streamlit as st
import openai
import faiss
import sys
import os
import numpy as np
import logging
import hashlib
import diskcache as dc
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Get the root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add src directory to sys.path
sys.path.append(ROOT_DIR)

print(f"Added to sys.path: {ROOT_DIR}")  # Debugging line

from src.retrieval import build_faiss_index, embed_texts, load_data
from src.safety_utils import ContentSafetyFilter, safe_gpt_call



# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Persistent caching
cache = dc.Cache("./cache_directory")

def generate_cache_key(query, persona):
    return hashlib.md5(f"{persona}-{query}".encode()).hexdigest()

def get_cached_response(cache_key):
    return cache.get(cache_key)

def store_in_cache(cache_key, response):
    cache[cache_key] = response

# Load FAISS index & sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
documents = load_data("data/sample.txt")
faiss_index = build_faiss_index(embed_texts(documents)) if documents else None

def retrieve_relevant_docs(query):
    if faiss_index is None:
        return []
    query_vector = model.encode([query])
    D, I = faiss_index.search(query_vector, k=3)
    return [documents[i] for i in I[0] if i >= 0] or []

@safe_gpt_call
def chat_with_gpt(persona, history, user_input):
    safety_check = ContentSafetyFilter.analyze_content(user_input)

    # If content is not safe, return an appropriate message
    if not safety_check['is_safe']:
        logging.warning(f"Unsafe content detected: {safety_check['risk_level']}")
        logging.warning(f"Matched patterns: {safety_check['matched_patterns']}")
        return f"⚠️ Your message contains potentially inappropriate content. Risk level: {safety_check['risk_level']}."
    
    # Optional: Filter the content if needed
    user_input = ContentSafetyFilter.filter_content(user_input)

    cache_key = generate_cache_key(user_input, persona)
    cached_response = get_cached_response(cache_key)
    
    if cached_response:
        logging.info("Cache hit for query: %s", user_input)
        return cached_response  # ✅ Return cached response
    
    system_prompt = {
        "Casual": "You are a friendly AI assistant. Only answer using the provided context and if the context does not match then just return a message saying the prompt was out of context.",
        "Professional": "You are a professional AI assistant. Use the provided context only and if the context does not match then just return a message saying the prompt was out of context..",
        "Technical": "You are an AI researcher. Base your response strictly on the provided documents and if the context does not match then just return a message saying the prompt was out of context.."
    }
    
    relevant_docs = retrieve_relevant_docs(user_input)
    if not relevant_docs:
        return "🚫 Your question is out of context. Please ask something related to the available documents."


    full_prompt = f"{system_prompt[persona]}\n\nContext:\n" + "\n".join(relevant_docs) + f"\n\nUser: {user_input}\n\nAssistant:"
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": full_prompt}, *history, {"role": "user", "content": user_input}],
            stream=True
        )
        full_response = "".join([chunk["choices"][0].get("delta", {}).get("content", "") for chunk in response])
        
        store_in_cache(cache_key, full_response)  # ✅ Store in cache
        return full_response
    except Exception as e:
        logging.error("Error calling OpenAI API: %s", str(e))
        return "⚠️ An error occurred while generating a response."

# Streamlit UI
st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Smart AI Assistant")

persona = st.sidebar.radio("Choose Persona:", ["Casual", "Professional", "Technical"])

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation
for message in st.session_state.messages:
    role, content = message["role"], message["content"]
    with st.chat_message(role):
        st.markdown(content)

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    # Display user input before processing
    with st.chat_message("user"):
        st.markdown(user_input)
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    response = chat_with_gpt(persona, st.session_state.messages, user_input)
    
    with st.chat_message("assistant"):
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
