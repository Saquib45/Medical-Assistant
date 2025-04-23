# import os
# import streamlit as st
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEndpoint
# from dotenv import load_dotenv, find_dotenv

# # Load environment variables
# load_dotenv(find_dotenv())
# HF_TOKEN = os.environ.get("HF_TOKEN")
# HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
# DB_FAISS_PATH = "vectorstore/db_faiss"

# # Apply custom styling
# st.markdown("""
#     <style>
#         html, body, [class*="css"] {
#             font-family: 'Segoe UI', sans-serif;
#             background-color: #0f172a;
#             color: white;
#         }
#         .title {
#             text-align: center;
#             font-size: 3rem;
#             font-weight: bold;
#             padding: 2rem 0;
#             background: linear-gradient(to right, #00c6ff, #0072ff);
#             -webkit-background-clip: text;
#             -webkit-text-fill-color: transparent;
#         }
#         .chat-box {
#             max-width: 700px;
#             margin: 1rem auto;
#             padding: 1rem 1.5rem;
#             border-radius: 12px;
#             font-size: 1rem;
#             white-space: pre-wrap;
#             word-wrap: break-word;
#             box-shadow: 0 0 10px rgba(0,0,0,0.2);
#         }
#         .user {
#             background: linear-gradient(to right, #1e3a8a, #3b82f6);
#             color: white;
#             text-align: left;
#         }
#         .assistant {
#             background-color: #1f2937;
#             border: 1px solid #374151;
#             color: #e5e7eb;
#         }
#         .chat-input {
#             background-color: #1e293b;
#             color: white;
#             border-radius: 10px;
#             padding: 1rem;
#             border: 1px solid #334155;
#         }
#         .block-container {
#             padding-top: 2rem;
#         }
#         /* 🚫 Hide Streamlit deploy button */
#         [data-testid="stDeployButton"] {
#             display: none !important;
#         }
#     </style>
# """, unsafe_allow_html=True)

# # Cache vectorstore to avoid reloading every time
# @st.cache_resource
# def get_vectorstore():
#     embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#     db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
#     return db

# # Prompt formatter
# def set_custom_prompt(custom_prompt_template):
#     prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
#     return prompt

# # Load Hugging Face model
# def load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN):
#     llm = HuggingFaceEndpoint(
#         repo_id=HUGGINGFACE_REPO_ID,
#         task="text-generation",
#         temperature=0.5,
#         max_new_tokens=512,
#         huggingfacehub_api_token=HF_TOKEN
#     )
#     return llm

# # Main app logic
# def main():
#     st.markdown('<div class="title">💬 Ask MediBot</div>', unsafe_allow_html=True)

#     if 'messages' not in st.session_state:
#         st.session_state.messages = []

#     for message in st.session_state.messages:
#         role_class = 'user' if message['role'] == 'user' else 'assistant'
#         st.markdown(f'<div class="chat-box {role_class}"><strong>{message["role"].capitalize()}:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)

#     prompt = st.chat_input("Type your question here...")

#     if prompt:
#         st.markdown(f'<div class="chat-box user"><strong>You:</strong><br>{prompt}</div>', unsafe_allow_html=True)
#         st.session_state.messages.append({'role': 'user', 'content': prompt})

#         CUSTOM_PROMPT_TEMPLATE = """
#         Use the pieces of information provided in the context to answer user's question.
#         If you don't know the answer, just say you don't know. Do not make up an answer. 
#         Only use the context provided.

#         Context: {context}
#         Question: {question}

#         Start your answer directly without any greeting or small talk.
#         """

#         try:
#             vectorstore = get_vectorstore()
#             if vectorstore is None:
#                 st.error("Failed to load the vector store.")

#             qa_chain = RetrievalQA.from_chain_type(
#                 llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN),
#                 chain_type="stuff",
#                 retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
#                 return_source_documents=True,
#                 chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
#             )

#             response = qa_chain.invoke({'query': prompt})
#             result = response["result"]

#             st.markdown(f'<div class="chat-box assistant"><strong>Assistant:</strong><br>{result}</div>', unsafe_allow_html=True)
#             st.session_state.messages.append({'role': 'assistant', 'content': result})

#         except Exception as e:
#             st.error(f"Error: {str(e)}")

# if __name__ == "__main__":
#     main()



import os
import requests
import streamlit as st
from urllib.parse import urlparse, parse_qs
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv

# Load .env file
load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Custom CSS for styling
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
            background-color: #1e1e2f;
            color: #ffffff;
        }
        .title {
            text-align: center;
            font-size: 3rem;
            font-weight: bold;
            background: linear-gradient(to right, #00c6ff, #0072ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding: 2rem 0;
        }
        .chat-box {
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .user {
            background-color: #3a3a5d;
        }
        .assistant {
            background-color: #29304d;
        }
        .error {
            color: #ff4d4d;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Validate JWT token with Spring Boot backend
def validate_token(token: str) -> bool:
    try:
        response = requests.get(
            "http://localhost:8081/dashboard",  # Your Spring Boot validation endpoint
            headers={"Authorization": f"Bearer {token}"}
        )
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error validating token: {e}")
        return False

# Load FAISS vector store
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(template: str):
    return PromptTemplate(template=template, input_variables=["context", "question"])

def load_llm():
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        task="text-generation",
        temperature=0.5,
        max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN
    )

def main():
    # Get token from query params
    query_params = st.query_params
    token = query_params.get("token")

    if not token or not validate_token(token):
        st.markdown("<p class='error'>❌ Invalid or missing token. Please log in from the dashboard.</p>", unsafe_allow_html=True)
        st.stop()

    st.markdown('<div class="title">💬 Ask MedS</div>', unsafe_allow_html=True)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        role_class = 'user' if message['role'] == 'user' else 'assistant'
        st.markdown(f'<div class="chat-box {role_class}"><strong>{message["role"].capitalize()}:</strong><br>{message["content"]}</div>', unsafe_allow_html=True)

    prompt = st.chat_input("Type your question here...")

    if prompt:
        st.markdown(f'<div class="chat-box user"><strong>You:</strong><br>{prompt}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you don't know the answer, just say that you don't know. Do not make up an answer.
        Do not provide anything outside of the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

        try:
            vectorstore = get_vectorstore()
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            st.markdown(f'<div class="chat-box assistant"><strong>Assistant:</strong><br>{result}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
