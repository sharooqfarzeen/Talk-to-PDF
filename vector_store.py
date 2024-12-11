from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

import streamlit as st


def create_vector_store(chunks, embeddings):
    # Creates the vector store
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    # Stores it locally
    vector_store.save_local("vector_store")