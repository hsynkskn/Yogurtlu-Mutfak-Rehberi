import os
import nest_asyncio
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
import streamlit as st

# ===== LangChain importları =====
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# ===== API Anahtarı =====
load_dotenv()
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ===== Patch: asyncio uyumsuzluğu çözmek için =====
nest_asyncio.apply()

# ===== Embedding Model (HuggingFace) =====
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ===== FAISS DB yükleme/kaydetme =====
@st.cache_resource
def load_vectordb():
    persist_directory = "faiss_index"

    if os.path.exists(persist_directory):
        vectordb = FAISS.load_local(persist_directory, get_embedding_model(), allow_dangerous_deserialization=True)
    else:
        loader = PyPDFLoader("yogurt-recipes.pdf")  # kendi PDF dosyanı koy
        documents = loader.load()

        # Metin parçalayıcı
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        yogurt_docs = text_splitter.split_documents(documents)

        # HuggingFace embedding kullan
        embedding = get_embedding_model()
        vectordb = FAISS.from_documents(yogurt_docs, embedding)

        # Kaydet
        vectordb.save_local(persist_directory)

    return vectordb

# ===== LLM (Gemini) =====
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

# ===== Streamlit Uygulaması =====
st.set_page_config(page_title="Yoğurtlu Mutfak Asistanı", layout="wide")
st.title("🥛 Yoğurtlu Mutfak Asistanı")

# Vektör DB
vectordb = load_vectordb()
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# Bellek
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Soru-Cevap zinciri
llm = get_llm()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    memory=memory,
    chain_type="stuff"
)

# Kullanıcı input
user_query = st.text_input("Tarif veya soru sorun (ör: 'Yoğurtlu çorba tarifi'):")

if user_query:
    response = qa_chain.run(user_query)

    # Çeviri (isteğe bağlı)
    translated_response = GoogleTranslator(source="auto", target="tr").translate(response)

    st.markdown("### 🍽️ Asistanın Yanıtı")
    st.write(translated_response)
