import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Constants
PDFS_DIRECTORY = 'pdfs'
FAISS_DB_PATH = "vectorstore/db_faiss"
OLLAMA_MODEL_NAME = "deepseek-r1:1.5b"
LLM_MODEL = ChatGroq(model="deepseek-r1:1.5b")

# Create PDFs directory if it doesn't exist
os.makedirs(PDFS_DIRECTORY, exist_ok=True)

# Prompt template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Don't provide anything out of the given context.

Question: {question} 
Context: {context} 
Answer:
"""

# Functions
def upload_pdf(file):
    file_path = os.path.join(PDFS_DIRECTORY, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    return splitter.split_documents(documents)

def get_embedding_model():
    return OllamaEmbeddings(model=OLLAMA_MODEL_NAME)

def create_vector_store(text_chunks):
    db = FAISS.from_documents(text_chunks, get_embedding_model())
    db.save_local(FAISS_DB_PATH)
    return db

def retrieve_docs(faiss_db, query):
    return faiss_db.similarity_search(query)

def get_context(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def answer_query(docs, query):
    context = get_context(docs)
    prompt = ChatPromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE)
    chain = prompt | LLM_MODEL
    return chain.invoke({"question": query, "context": context})

# UI
st.title("📄🧠 AI Lawyer Assistant")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
user_query = st.text_area("Enter your prompt:", height=150, placeholder="Ask anything about the uploaded PDF...")
ask_question = st.button("Ask AI Lawyer")

if ask_question:
    if uploaded_file and user_query:
        try:
            file_path = upload_pdf(uploaded_file)
            documents = load_pdf(file_path)
            chunks = create_chunks(documents)
            db = create_vector_store(chunks)
            docs = retrieve_docs(db, user_query)
            response = answer_query(docs, user_query)

            st.chat_message("user").write(user_query)
            st.chat_message("AI Lawyer").write(response)

        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")
    else:
        st.error("⚠️ Please upload a PDF and enter a query.")
