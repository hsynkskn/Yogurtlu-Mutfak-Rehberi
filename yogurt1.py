import os
import streamlit as st
import nest_asyncio
from dotenv import load_dotenv
from deep_translator import GoogleTranslator

# LangChain ve ilgili kütüphaneler
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain

# Ortam değişkenlerini yükle
load_dotenv()
nest_asyncio.apply()

# =======================
# Embedding Model
# =======================
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# =======================
# HuggingFace LLM
# =======================
def get_llm():
    return HuggingFaceHub(
        repo_id="google/flan-t5-base",  # ücretsiz küçük model
        model_kwargs={"temperature": 0.2, "max_length": 512}
    )

# =======================
# FAISS Vektör DB
# =======================
FAISS_PATH = "faiss_index"

@st.cache_resource(show_spinner="Vektör veritabanı hazırlanıyor...")
def load_vectordb():
    if os.path.exists(FAISS_PATH):
        vectordb = FAISS.load_local(
            FAISS_PATH, get_embedding_model(), allow_dangerous_deserialization=True
        )
        st.success("Vektör veritabanı diskten yüklendi ✅")
        return vectordb
    else:
        loader = PyPDFLoader("yogurt-uygarligi.pdf")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        yogurt_docs = text_splitter.split_documents(documents)

        embedding = get_embedding_model()
        vectordb = FAISS.from_documents(yogurt_docs, embedding)
        vectordb.save_local(FAISS_PATH)
        st.success("Vektör veritabanı oluşturuldu ve kaydedildi ✅")
        return vectordb

# =======================
# RAG Zinciri
# =======================
@st.cache_resource(show_spinner="RAG zinciri hazırlanıyor...")
def create_rag_chain():
    vectordb = load_vectordb()
    llm = get_llm()
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever()
    )
    return rag_chain

# =======================
# Streamlit Arayüz
# =======================
st.title("🥛 Yoğurtlu Mutfak Asistanı (HuggingFace sürümü)")

rag_chain = create_rag_chain()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Soru sor:", "")

if user_input:
    result = rag_chain.invoke({"question": user_input, "chat_history": st.session_state.chat_history})

    st.session_state.chat_history.append((user_input, result["answer"]))

    st.write("### Yanıt:")
    st.write(result["answer"])

