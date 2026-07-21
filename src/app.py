import streamlit as st
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Environment variables load karein
load_dotenv()

st.set_page_config(page_title="Industrial Knowledge Brain", page_icon="🏭", layout="wide")
st.title("🏭 AI Industrial Knowledge Intelligence Platform")
st.markdown("---")

CHROMA_DIR = "chroma_db"
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("❌ Error: GROQ_API_KEY aapki `.env` file mein nahi mili!")
elif not os.path.exists(CHROMA_DIR):
    st.warning("⚠️ Knowledge base taiyar nahi hai. Pehle terminal mein `python src/rag_engine.py` run karein.")
else:
    # Local Embeddings ke sath Chroma DB load karna
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 🚀 Groq LLM Engine (Super Fast Llama3 Model)
    llm = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name="llama-3.3-70b-versatile",
        temperature=0.2
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask about store metrics, recommendations, or python code..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("🧠 Analyzing industrial documents via Groq LLM..."):
                try:
                    # 1. Context retrieval from local vector DB
                    relevant_docs = retriever.invoke(user_query)
                    context = "\n\n".join([d.page_content for d in relevant_docs])

                    prompt_text = f"""You are an expert Industrial Data Analyst. Answer the question based ONLY on the provided context.
If you don't know the answer, say you don't know. Keep it professional and structured.

Context:
{context}

Question: {user_query}

Answer:"""

                    # 2. Call Groq
                    response = llm.invoke(prompt_text)
                    ai_response = response.content

                    st.markdown(ai_response)
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")