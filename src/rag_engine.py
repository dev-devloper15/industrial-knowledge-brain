import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"

def initialize_knowledge_base():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"📁 '{DATA_DIR}' folder is created. insert PDFs.")
        return None

    if not os.listdir(DATA_DIR):
        print(f"⚠️ Warning: '{DATA_DIR}' folder is empty please insert pdf file")
        return None

    print("📄 PDFs read is in process")
    loader = PyPDFDirectoryLoader(DATA_DIR)
    docs = loader.load()
    print(f"📚 Total {len(docs)} pages load")

    print("✂️ Text break into chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_docs = text_splitter.split_documents(docs)
    print(f"🧩 Total {len(final_docs)} chunks")

    print("🧠 Local HuggingFace Engine to Vector Database going to start")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # ChromaDB directory generated on local disk 
    vectorstore = Chroma.from_documents(
        documents=final_docs, 
        embedding=embeddings, 
        persist_directory=CHROMA_DIR
    )
    print("💾 ChromaDB safely save!")
    return vectorstore

if __name__ == "__main__":
    print("🚀 Fresh Initialization Test started")
    initialize_knowledge_base()
