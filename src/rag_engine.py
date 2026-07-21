import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# 1. Environment variables load karna
load_dotenv()

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"

def initialize_knowledge_base():
    """PDFs ko padh kar local vector store banane ka function"""
    # Agar data folder nahi hai toh banao
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 '{DATA_DIR}' folder bana diya gaya hai. Isme apni PDFs daalein.")
        return None

    # Folder khali hone par warning
    if not os.listdir(DATA_DIR):
        print(f"⚠️ Warning: '{DATA_DIR}' folder khali hai! PDFs daalna zaroori hai.")
        return None

    print("📄 PDFs ko read kiya jaa raha hai...")
    loader = PyPDFDirectoryLoader(DATA_DIR)
    docs = loader.load()
    print(f"📚 Total {len(docs)} pages load hue.")

    print("✂️ Text ko chunks me toda jaa raha hai...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_docs = text_splitter.split_documents(docs)
    print(f"🧩 Total {len(final_docs)} chunks banaye gaye.")

    print("🧠 Local HuggingFace Engine se Vector Database taiyar ho raha hai...")
    
    # 📌 100% Stable Local Model - API Key ki koi zaroorat nahi hai
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # ChromaDB directory local disk par generate hogi
    vectorstore = Chroma.from_documents(
        documents=final_docs, 
        embedding=embeddings, 
        persist_directory=CHROMA_DIR
    )
    print("💾 ChromaDB local disk par safely save ho gaya!")
    return vectorstore

if __name__ == "__main__":
    print("🚀 Fresh Initialization Test Shuru...")
    initialize_knowledge_base()