import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Groq API key from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Custom CSS styling
st.markdown("""
<style>
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    .stSelectbox svg {
        fill: white !important;
    }
    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 DeepSeek Code Companion")
st.caption("🚀 Your AI Pair Programmer with Debugging Superpowers")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["llama-3.1-8b-instant", "openai/gpt-oss-20b"],  
        index=0
    )
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Python Expert
    - 🐞 Debugging Assistant
    - 📝 Code Documentation
    - 💡 Solution Design
    """)
    st.divider()
    st.markdown("Built with [Groq](https://groq.com/) | [LangChain](https://python.langchain.com/)")

# Initialize Groq LLM
llm_engine = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model=selected_model,
    temperature=0.3
)

# System prompt configuration
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions "
    "with strategic print statements for debugging. Always respond in English."
)

# Session state management
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm Yatin Kamal. How can I help you code today? 💻"}]

# Chat container
chat_container = st.container()

with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
user_query = st.chat_input("Type your coding question here...")

def generate_ai_response(prompt_chain):
    response = prompt_chain.invoke()
    return response.content

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    with st.spinner("🧠 Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    st.rerun()

