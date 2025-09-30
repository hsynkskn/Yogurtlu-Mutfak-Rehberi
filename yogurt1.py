import os
from pathlib import Path
import streamlit as st
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# LangChain bileşenlerini ekliyoruz
from langchain.prompts import PromptTemplate 
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ================== Dil Seçimi ==================
languages = {
    "Türkçe TR": "tr",
    "English GB": "en",
}

col1, col2 = st.columns([6, 4])
with col1:
    selected_lang = st.radio(
        "🌐 Language:", options=list(languages.keys()), index=0, horizontal=True
    )
# Seçilen dilin kısa kodu (örneğin 'tr' veya 'en')
target_lang = languages[selected_lang] 
st.session_state["target_lang"] = target_lang # Dil kodunu session state'e kaydediyoruz

# ================== Vektör DB Yapılandırması ==================
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index"
PDF_FOLDER = "pdfs"
GROQ_MODEL = "llama-3.1-8b-instant"

@st.cache_resource
def get_embeddings():
    """HuggingFace gömme modelini yükler."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

@st.cache_data
def create_and_save_vectordb(_pdf_folder=PDF_FOLDER, _db_path=FAISS_INDEX_PATH):
    """PDF'leri yükler, parçalar, gömer ve FAISS veritabanı oluşturup kaydeder."""
    pdf_folder_path = Path(_pdf_folder)
    if not pdf_folder_path.exists():
        st.error(f"'{_pdf_folder}' klasörü bulunamadı. Lütfen PDF dosyalarınızı buraya ekleyin.")
        return None

    docs = []
    for pdf_file in pdf_folder_path.glob("*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs.extend(loader.load())
        except Exception as e:
            st.warning(f"'{pdf_file.name}' yüklenemedi: {e}")

    if not docs:
        st.warning("Hiçbir PDF bulunamadı.")
        return None

    embeddings = get_embeddings()
    try:
        vectordb = FAISS.from_documents(docs, embeddings)
        vectordb.save_local(_db_path)
        st.success(f"Vektör veritabanı '{_db_path}' adresine kaydedildi ✅")
        return vectordb
    except Exception as e:
        st.error(f"FAISS oluşturulamadı: {e}")
        return None

def load_local_vectordb(_db_path=FAISS_INDEX_PATH):
    """Yerel olarak kaydedilmiş FAISS veritabanını yükler."""
    embeddings = get_embeddings()
    if Path(_db_path).exists():
        try:
            return FAISS.load_local(_db_path, embeddings)
        except ValueError as e:
            st.warning(f"FAISS index yüklenemedi: {e}. Yeniden oluşturmayı deneyin.")
            return None
    return None

# ================== Groq API ve Model ==================
def get_groq_llm():
    """LangChain için Groq Chat Modelini döndürür."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("❌ GROQ_API_KEY ortam değişkeni bulunamadı. Lütfen ayarla.")
        return None
    
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=512
    )
    return llm

# ================== Prompt Tanımı (DİL DESTEĞİ EKLENDİ) ==================
# Yeni input_variables: ["context", "question", "language"]
prompt_template = PromptTemplate(
    input_variables=["context", "question", "language"],
    template="""
You are a chef assistant specializing in yogurt-based recipes. The text below contains recipe information about yogurt:

{context}

Based on the ingredients provided by the user, suggest recipes that primarily contain yogurt.
Prioritize Turkish cuisine. Provide a list of ingredients and preparation steps.
Write in a simple, fluent, and user-friendly tone. Suggest alternative ingredients if necessary.

---
VERY IMPORTANT: **Translate the entire response into the language specified by the 'language' variable.** Language Code: {language}
---

Ingredients: {question}
"""
)

# ================== RAG Zinciri (LCEL) ==================
def create_rag_chain_lcel(_vectordb):
    """LCEL kullanarak RAG zincirini oluşturur."""
    llm = get_groq_llm()
    if llm is None:
        return None

    retriever = _vectordb.as_retriever(search_kwargs={"k": 3})

    # LCEL Zinciri:
    # 1. 'context' ve 'language' için iki ayrı girdi pas-through ile alınıyor.
    #    'context': retriever'dan gelen belgeler.
    #    'question': Kullanıcının orijinal sorusu.
    #    'language': Streamlit Session State'den alınan dil kodu.
    rag_chain = (
        {
            # Belgeleri al ve string'e çevir
            "context": retriever | (lambda docs: "\n".join([doc.page_content for doc in docs])), 
            # Kullanıcının orijinal sorusunu koru
            "question": RunnablePassthrough(),
            # Session State'deki dil kodunu doğrudan prompt'a gönder
            "language": (lambda x: st.session_state["target_lang"])
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return rag_chain


# ================== Streamlit UI ==================
st.title("🥛 Yoğurtlu Mutfak Asistanı")

# Vektör veritabanını yükle veya oluştur
vectordb = load_local_vectordb()
if vectordb is None:
    st.info("Yerel FAISS index bulunamadı. PDF’lerden oluşturuluyor...")
    vectordb = create_and_save_vectordb()

# Sorgulama arayüzü
if vectordb is not None:
    # RAG Zincirini oluştur
    rag_chain = create_rag_chain_lcel(vectordb)
    
    # Kullanıcının diline göre input alanını güncelleyelim
    input_label = "Aradığınız malzemeyi veya tarifi yazın:" if target_lang == "tr" else "Enter the ingredient or recipe you are looking for:"

    if rag_chain is not None:
        user_question = st.text_input(input_label)
        
        if user_question:
            with st.spinner("Cevap hazırlanıyor (Groq API)..." if target_lang == "tr" else "Preparing the answer (Groq API)..."):
                # Zinciri çalıştırma
                # Burada RunnablePassthrough() kullandığımız için sadece user_question'ı invoke etmemiz yeterlidir.
                answer = rag_chain.invoke(user_question) 
                st.markdown(f"**Cevap ({selected_lang}):** {answer}" if target_lang == "tr" else f"**Answer ({selected_lang}):** {answer}")
    else:
        st.error("RAG zinciri başlatılamadı (GROQ_API_KEY eksik olabilir)." if target_lang == "tr" else "RAG chain could not be initialized (GROQ_API_KEY might be missing).")
else:
    st.warning("Vektör veritabanı yüklenemedi. Lütfen PDF klasörünüzü kontrol edin." if target_lang == "tr" else "Vector database could not be loaded. Please check your PDF folder.")
