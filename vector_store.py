from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

import streamlit as st

api_key = st.session_state.api_keys["GOOGLE_API_KEY"]

def create_vector_store(chunks, embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)):
    # Creates the vector store
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    # Stores it locally
    vector_store.save_local("vector_store")