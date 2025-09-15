import streamlit as st 
import streamlit as st 
import os
from dotenv import load_dotenv
import time
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate as ChatPromptTemplate2
)
from langchain_community.embeddings import HuggingFaceEmbeddings



load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
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

PDF_STORAGE_PATH = 'documents_store/pdfs'
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided document context to answer the user's query accurately and concisely.
- If the user asks for a summary, summarize the key points clearly.
- If the user asks for an explanation, expand or clarify the content.
- If the user asks a specific question, provide a factual answer based only on the context.
Do not make assumptions beyond the document.  
Keep answers clear, concise, and factual (3-5 sentences max unless the user asks for detailed explanation).

Document Context:
{document_context}

User Query:
{user_query}

Answer:
"""

def save_uploaded_file(uploaded_file):
    file_path = uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversational_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    LANGUAGE_MODEL = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)
    response_chain = conversational_prompt | LANGUAGE_MODEL
    response = response_chain.invoke({"user_query": user_query, "document_context": context_text})
    return response.content

def generate_ai_response(prompt_chain, llm_engine):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

def build_prompt_chain(system_prompt, message_log):
    prompt_sequence = [system_prompt]
    for msg in message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate2.from_messages(prompt_sequence)

with st.sidebar:
    st.header("🛠️ Choose Mode")
    app_mode = st.radio("Select Application", ["📄 Document Assistant", "💬 Smart Chat Assistant"])

if app_mode == "📄 Document Assistant":
    st.title("📄 DOCUMENT AI")
    st.markdown("Your intelligent research assistant")
    st.markdown("-----------")

    uploaded_pdf = st.file_uploader(
        "Upload your research documents in PDF format",
        type="pdf",
        help="Select a PDF document for analysis",
        accept_multiple_files=False
    )

    if uploaded_pdf:
        saved_path = save_uploaded_file(uploaded_pdf)
        raw_docs = load_pdf_documents(saved_path)
        processed_chunks = chunk_documents(raw_docs)
        index_documents(processed_chunks)

        st.success("✅ Document processed successfully. Ask your questions below.")
        user_input = st.chat_input("Enter your question about the document...")

        if user_input:
            with st.chat_message("user"):
                st.write(user_input)

            with st.spinner("🔍 Analyzing documents..."):
                start_time = time.time()
                relevent_docs = find_related_documents(user_input)
                ai_response = generate_answer(user_input, relevent_docs)
                elapsed = time.time() - start_time

            with st.chat_message("assistant"):
                st.write(ai_response)
                st.caption(f"⏱️ Answered in {elapsed:.2f} seconds")
# Smart Chat Assistant Mode 
elif app_mode == "💬 Smart Chat Assistant":
    st.title("🤖 Smart AI Assistant")
    st.caption("Switch between concise or detailed responses based on your needs")

    with st.sidebar:
        st.subheader("⚙️ Response Mode")
        response_mode = st.selectbox(
            "Choose Answer Style",
            ["📝 Concise Answer", "📖 Detailed Answer"],
            index=0
        )

    model_map = {
        "📝 Concise Answer": "llama-3.1-8b-instant",
        "📖 Detailed Answer": "openai/gpt-oss-20b"
    }
    selected_model = model_map[response_mode]

    llm_engine = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model=selected_model,
        temperature=0.4
    )

    if response_mode == "📝 Concise Answer":
        system_prompt = SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant. Provide short, concise, and to-the-point answers."
        )
    else:
        system_prompt = SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant. Provide detailed, expanded, and in-depth answers with clear explanations."
        )

    if "message_log" not in st.session_state:
        st.session_state.message_log = [{"role": "ai", "content": "Hello 👋 How can I help you today?"}]

    chat_container = st.container()
    with chat_container:
        for message in st.session_state.message_log:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    user_query = st.chat_input("Type your question here...")

    if user_query:
        st.session_state.message_log.append({"role": "user", "content": user_query})

        with st.spinner("🤔 Thinking..."):
            start_time = time.time()
            prompt_chain = build_prompt_chain(system_prompt, st.session_state.message_log)
            ai_response = generate_ai_response(prompt_chain, llm_engine)
            elapsed = time.time() - start_time

        st.session_state.message_log.append({"role": "ai", "content": f"{ai_response}\n\n⏱️ Answered in {elapsed:.2f} seconds"})
        st.rerun()


