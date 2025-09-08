import streamlit as st 
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq   # <-- Groq import

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#for styling we can use this pre made snippet
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    /* ... (rest of your CSS unchanged) ... */
    </style>
    """, unsafe_allow_html=True)

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


PDF_STORAGE_PATH = 'documents_store/pdfs'
EMBEDDING_MODEL = OllamaEmbeddings(model = "mxbai-embed-large")   # <-- Embeddings switched
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)  # <-- Groq with API key

def save_uploaded_file(uploaded_file):
    file_path =  uploaded_file.name
    with open(file_path, "wb") as file:
       file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor =  RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index = True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversational_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversational_prompt | LANGUAGE_MODEL
    response = response_chain.invoke({"user_query": user_query, "document_context": context_text})
    return response.content  # <-- Extract only the content

#UI CONFIGURATION
st.title("DOCUMENT AI")
st.markdown("YOUR INTELLIGENT DOCUMENT ASSISTANT")
st.markdown("-----------")

#file upload section
uploaded_pdf = st.file_uploader(
    "Upload your research documents in PDF format",
    type = "pdf",
    help = "Select a PDF document for analysis",
    accept_multiple_files=False
)

if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_docs = load_pdf_documents(saved_path)
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)

    st.success("Documents processed successfully. Ask your questions below.")
    user_input = st.chat_input("Enter your question about the document...")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Analyzing documents..."):
            relevent_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevent_docs)

        with st.chat_message("assistant"):
            st.write(ai_response)
