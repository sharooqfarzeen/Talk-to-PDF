from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

def create_vector_store(chunks, embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)):
    # Creates the vector store
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    # Stores it locally
    vector_store.save_local("vector_store")