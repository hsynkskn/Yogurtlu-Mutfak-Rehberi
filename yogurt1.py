import os
from pathlib import Path
import streamlit as st
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
# LangChain bileşenlerini ekliyoruz
from langchain.prompts import PromptTemplate 

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
target_lang = languages[selected_lang]

# ================== Vektör DB ==================
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index"
PDF_FOLDER = "pdfs"

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
        # FAISS veritabanını oluşturma
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
            st.warning(f"FAISS index yüklenemedi: {e}")
            return None
    return None

# ================== Groq API ==================
@st.cache_resource
def get_groq_client():
    """Groq API istemcisini oluşturur."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("❌ GROQ_API_KEY ortam değişkeni bulunamadı. Lütfen ayarla.")
        return None
    return Groq(api_key=api_key)

def query_groq(prompt: str, model="llama-3.1-8b-instant"):
    """Groq API'sine sorgu gönderir."""
    client = get_groq_client()
    if client is None:
        return "Groq API anahtarı bulunamadı."

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Groq API hatası: {e}"

# ================== Prompt Tanımı ==================
# İstediğiniz PromptTemplate
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Sen bir şef asistanısın. Aşağıda yoğurtla ilgili tarif bilgileri içeren bir metin var:

{context}

Kullanıcının verdiği malzemelere uygun, sadece yoğurt içeren tarifler öner.
Türk mutfağına öncelik ver. Malzeme listesi ve yapılış adımlarını yaz.
Sade, akıcı ve kullanıcı dostu bir dille yaz. Gerekiyorsa alternatif malzemeler de öner.

Malzemeler: {question}
"""
)

# ================== RAG Chain ==================
def create_rag_chain(_vectordb):
    """RAG zincirini (fonksiyonunu) oluşturur."""
    def rag_answer(query):
        if _vectordb is None:
            return "Veritabanı mevcut değil."
        try:
            # 1. Alaka düzeyi araması (Retrieval)
            docs = _vectordb.similarity_search(query, k=3)
            if not docs:
                return "İlgili bilgi bulunamadı."
            
            # 2. Context (Bağlam) metnini birleştirme
            context_text = "\n".join([doc.page_content for doc in docs])
            
            # 3. Prompt oluşturma (Augmentation)
            # PromptTemplate'i kullanarak tam prompt metnini oluşturma
            full_prompt = prompt_template.format(
                context=context_text,
                question=query
            )
            
            # 4. Groq API'ye sorgu gönderme (Generation)
            return query_groq(full_prompt)
        except Exception as e:
            return f"Cevap üretilirken hata: {e}"
    return rag_answer

# ================== Streamlit UI ==================
st.title("🥛 Yoğurtlu Mutfak Asistanı")

# Vektör veritabanını yükle veya oluştur
vectordb = load_local_vectordb()
if vectordb is None:
    st.info("Yerel FAISS index bulunamadı. PDF’lerden oluşturuluyor...")
    vectordb = create_and_save_vectordb()

# Sorgulama arayüzü
if vectordb is not None:
    rag_chain = create_rag_chain(vectordb)
    user_question = st.text_input("Aradığınız malzemeyi veya tarifi yazın:")
    
    if user_question:
        with st.spinner("Cevap hazırlanıyor (Groq API)..."):
            answer = rag_chain(user_question)
            st.markdown(f"**Cevap:** {answer}")
else:
    st.warning("Vektör veritabanı yüklenemedi. Lütfen PDF klasörünüzü kontrol edin.")
