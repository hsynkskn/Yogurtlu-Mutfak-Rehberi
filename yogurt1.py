import os
from pathlib import Path
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# LangChain bileşenlerini içe aktarma
from langchain.prompts import PromptTemplate 
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ================== Yapılandırma Sabitleri ==================
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index"
PDF_FOLDER = "pdfs"
GROQ_MODEL = "llama-3.1-8b-instant"

# ================== Dil Seçimi ve Global Durum ==================
languages = {
    "Türkçe TR": "tr",
    "English GB": "en",
}

col1, col2 = st.columns([6, 4])
with col1:
    selected_lang_name = st.radio(
        "🌐 Language:", options=list(languages.keys()), index=0, horizontal=True
    )
# Seçilen dilin kısa kodu ('tr' veya 'en')
target_lang = languages[selected_lang_name] 
st.session_state["target_lang"] = target_lang

# ================== Embeddings ve Vektör DB Fonksiyonları ==================

@st.cache_resource
def get_embeddings():
    """HuggingFace gömme modelini yükler."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

@st.cache_data
def create_and_save_vectordb(_pdf_folder=PDF_FOLDER, _db_path=FAISS_INDEX_PATH):
    """PDF'leri yükler ve FAISS veritabanı oluşturup kaydeder."""
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
        st.error(f"FAISS oluşturulamadı: {e}. Lütfen FAISS paketini kurun: pip install faiss-cpu")
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
    try:
        # Önce st.secrets'tan almaya çalış
        if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
        else:
            # Sonra ortam değişkenlerine bak
            api_key = os.getenv("GROQ_API_KEY")
        
        # API anahtarını kontrol et
        if not api_key:
            st.error("""
            ❌ GROQ_API_KEY bulunamadı. 
            
            **Lütfen aşağıdakilerden birini yapın:**
            
            **Streamlit Cloud için:**
            1. Uygulamanın "Manage App" → "Settings" → "Secrets" bölümüne gidin
            2. Aşağıdakini ekleyin:
            ```
            GROQ_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
            ```
            
            **Yerel geliştirme için:**
            1. .streamlit/secrets.toml dosyası oluşturun
            2. Aşağıdakini ekleyin:
            ```
            GROQ_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
            ```
            """)
            return None
            
        # Groq modelini oluştur
        llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=0.2,
            max_tokens=512,
            groq_api_key=api_key 
        )
        return llm
        
    except Exception as e:
        st.error(f"Groq modeli oluşturulamadı: {e}")
        return None

# ================== Prompt Tanımları (Dile Göre Ayrı) ==================

# 1. Türkçe Prompt Template
TR_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Sen bir şef asistanısın. Aşağıda yoğurtla ilgili tarif bilgileri içeren bir metin var:

{context}

Kullanıcının verdiği malzemelere uygun, sadece yoğurt içeren tarifler öner.
Türk mutfağına öncelik ver. Malzeme listesi ve yapılış adımlarını **Türkçe** olarak yaz.
Sade, akıcı ve kullanıcı dostu bir dille yaz. Gerekiyorsa alternatif malzemeler de öner.

Malzemeler: {question}
"""
)

# 2. İngilizce Prompt Template
EN_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a chef assistant specializing in yogurt-based recipes. The text below contains yogurt recipe information:

{context}

Based on the ingredients provided by the user, suggest recipes that primarily contain yogurt.
Prioritize Turkish cuisine. Provide a list of ingredients and preparation steps **in English**.
Write in a simple, fluent, and user-friendly tone. Suggest alternative ingredients if necessary.

Ingredients: {question}
"""
)

# ================== RAG Zinciri (LCEL) ==================
def create_rag_chain_lcel(_vectordb, _target_lang):
    """LCEL kullanarak RAG zincirini oluşturur."""
    llm = get_groq_llm()
    if llm is None:
        return None

    # Seçilen dile göre doğru prompt'u kullan
    prompt_template = TR_PROMPT if _target_lang == "tr" else EN_PROMPT
    
    # Retriever (FAISS veritabanından belge alıcı)
    retriever = _vectordb.as_retriever(search_kwargs={"k": 3})

    # LCEL Zinciri
    rag_chain = (
        {
            "context": retriever | (lambda docs: "\n".join([doc.page_content for doc in docs])), 
            "question": RunnablePassthrough(),
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
    st.info("Yerel FAISS index bulunamadı. PDF'lerden oluşturuluyor...")
    vectordb = create_and_save_vectordb()

# Sorgulama arayüzü
if vectordb is not None:
    # RAG Zincirini, seçilen dil kodu ile oluştur
    rag_chain = create_rag_chain_lcel(vectordb, target_lang)
    
    # Kullanıcının diline göre UI metinleri
    if target_lang == "tr":
        input_label = "Aradığınız malzemeyi veya tarifi yazın:"
        spinner_text = "Cevap hazırlanıyor (Groq API)..."
        error_text = "RAG zinciri başlatılamadı (API Anahtarı eksik veya geçersiz)."
        answer_prefix = f"**Cevap ({selected_lang_name}):**"
        db_warning = "Vektör veritabanı yüklenemedi. Lütfen PDF klasörünüzü kontrol edin."
    else: # English
        input_label = "Enter the ingredient or recipe you are looking for:"
        spinner_text = "Preparing the answer (Groq API)..."
        error_text = "RAG chain could not be initialized (API Key missing or invalid)."
        answer_prefix = f"**Answer ({selected_lang_name}):**"
        db_warning = "Vector database could not be loaded. Please check your PDF folder."

    if rag_chain is not None:
        user_question = st.text_input(input_label)
        
        if user_question:
            with st.spinner(spinner_text):
                try:
                    # Zinciri çalıştırma
                    answer = rag_chain.invoke(user_question) 
                    st.markdown(f"{answer_prefix} {answer}")
                except Exception as e:
                    st.error(f"Sorgu sırasında hata oluştu: {e}")
    else:
        st.error(error_text)
else:
    # db_warning değişkenini burada tanımlayalım
    if target_lang == "tr":
        db_warning = "Vektör veritabanı yüklenemedi. Lütfen PDF klasörünüzü kontrol edin."
    else:
        db_warning = "Vector database could not be loaded. Please check your PDF folder."
    st.warning(db_warning)
