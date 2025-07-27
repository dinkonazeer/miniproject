import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import os
from datetime import datetime

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


# Extract text from PDFs
def get_pdf_etext(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Split text into chunks                                                                        
def get_text_chunks(text, model_name):
    if model_name == "Google AI":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


# Generate embeddings and store in vector database
def get_vector_store(text_chunks, model_name, api_key=None):
    if model_name == "Google AI":
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store


# Load QA chain
def get_conversational_chain(model_name, vectorstore=None, api_key=None):
    if model_name == "Google AI":
        prompt_template = """
Answer the question as detailed as possible from the provided context. 
If the answer is not in the context, say: "Answer is not available in the context." Do not guess.

Context:
{context}

Question:
{question}

Answer:
"""
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            google_api_key=api_key
        )
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain


# Main question handler
def user_input(user_question, model_name, api_key, pdf_docs, conversation_history):
    if api_key is None or not pdf_docs:
        st.warning("Please upload PDF and provide a valid API key.")
        return

    text = get_pdf_etext(pdf_docs)
    text_chunks = get_text_chunks(text, model_name)
    vector_store = get_vector_store(text_chunks, model_name, api_key)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(model_name, vectorstore=new_db, api_key=api_key)

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    answer = response['output_text']

    pdf_names = [pdf.name for pdf in pdf_docs]
    conversation_history.append(
        (user_question, answer, model_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ", ".join(pdf_names))
    )

    # Display chat
    st.markdown(chat_html(user_question, answer), unsafe_allow_html=True)

    for question, answer, model_name, timestamp, pdf_name in reversed(conversation_history[:-1]):
        st.markdown(chat_html(question, answer), unsafe_allow_html=True)

    # CSV export
    if conversation_history:
        df = pd.DataFrame(conversation_history, columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download conversation history as CSV</button></a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)

    st.markdown("✅ You can download your conversation from the sidebar.")
    st.snow()


# Format HTML chat message
def chat_html(user_question, response):
    return f"""
        <style>
            .chat-message {{
                padding: 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                display: flex;
            }}
            .chat-message.user {{
                background-color: #2b313e;
            }}
            .chat-message.bot {{
                background-color: #475063;
            }}
            .chat-message .avatar {{
                width: 20%;
            }}
            .chat-message .avatar img {{
                max-width: 78px;
                max-height: 78px;
                border-radius: 50%;
                object-fit: cover;
            }}
            .chat-message .message {{
                width: 80%;
                padding: 0 1.5rem;
                color: #fff;
            }}
        </style>
        <div class="chat-message user">
            <div class="avatar">
                <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
            </div>    
            <div class="message">{user_question}</div>
        </div>
        <div class="chat-message bot">
            <div class="avatar">
                <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp" >
            </div>
            <div class="message">{response}</div>
        </div>
    """


# Streamlit app main
def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon="📚")
    st.header(" Chat with Multiple PDFs")
    st.header("📚StudyBot")

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    st.sidebar.title("📌 Controls")
    model_name = st.sidebar.radio("Select Model:", ("Google AI",))

    api_key = st.sidebar.text_input("🔑 Google API Key:", type="password")
    st.sidebar.markdown("[Get your API key](https://ai.google.dev/)")

    if not api_key:
        st.sidebar.warning("Please enter your Google API key.")
        return

    pdf_docs = st.sidebar.file_uploader("📄 Upload PDFs", accept_multiple_files=True)

    if st.sidebar.button("Submit & Process"):
        if pdf_docs:
            with st.spinner("🔄 Processing PDFs..."):
                st.success("✅ PDFs processed!")
        else:
            st.warning("Please upload PDFs before processing.")

    col1, col2 = st.sidebar.columns(2)
    if col1.button("🔁 Rerun"):
        if st.session_state.conversation_history:
            st.warning("Last query will be reloaded.")
    if col2.button("🗑️ Reset"):
        st.session_state.conversation_history = []
        st.experimental_rerun()

    user_question = st.text_input("💬 Ask a question from the PDFs:", key="user_input")

    if user_question:
        user_input(user_question, model_name, api_key, pdf_docs, st.session_state.conversation_history)
        st.session_state.user_input = ""


if __name__ == "__main__":
    main()
