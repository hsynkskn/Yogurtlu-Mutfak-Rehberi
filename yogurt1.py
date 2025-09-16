import os
from pathlib import Path
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# ================== Dil Seçimi ==================
languages = {
    "Türkçe TR": "tr",
    "English GB": "en",
    "Français FR": "fr",
    "Deutsch DE": "de",
    "Español ES": "es",
    "Русский RU": "ru"
}

col1, col2 = st.columns([6, 4])
with col1:
    selected_lang = st.radio(
        "🌐 Language:", options=list(languages.keys()), index=0, horizontal=True
    )
target_lang = languages[selected_lang]

# ================== Vektör DB İşlemleri ==================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index"
PDF_FOLDER = "pdfs"

@st.cache_resource  # Use cache_resource for models/embeddings
def get_embeddings():
    """Loads and caches the embeddings model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

@st.cache_data  # Use cache_data for data-like objects (like the vector DB itself)
def create_and_save_vectordb(_pdf_folder=PDF_FOLDER, _db_path=FAISS_INDEX_PATH):
    """Loads PDFs, creates FAISS index, and saves it."""
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
            st.warning(f"'{pdf_file.name}' dosyası yüklenirken hata oluştu: {e}")

    if not docs:
        st.warning("Hiçbir PDF dosyası bulunamadı veya okunamadı.")
        return None

    embeddings = get_embeddings()
    try:
        vectordb = FAISS.from_documents(docs, embeddings)
        vectordb.save_local(_db_path)
        st.success(f"Vektör veritabanı başarıyla oluşturuldu ve '{_db_path}' adresine kaydedildi. ✅")
        return vectordb
    except Exception as e:
        st.error(f"Vektör veritabanı oluşturulurken hata oluştu: {e}")
        return None

def load_local_vectordb(_db_path=FAISS_INDEX_PATH):
    """Loads the FAISS index from the local path."""
    embeddings = get_embeddings()
    if Path(_db_path).exists():
        try:
            return FAISS.load_local(_db_path, embeddings)
        except ValueError as e:
            st.warning(f"FAISS index yüklenirken bir hata oluştu: {e}. Yeniden oluşturulması gerekebilir.")
            return None
    return None

# ================== HuggingFace Local Model ==================
@st.cache_resource
def get_llm_local():
    """Loads and caches the local LLM pipeline."""
    model_name = "google/flan-t5-small"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.2
        )
    except Exception as e:
        st.error(f"LLM modeli yüklenirken hata oluştu: {e}")
        return None

# ================== RAG Chain Oluşturma ==================
def create_rag_chain(_vectordb):
    """Creates the RAG answering function."""
    llm = get_llm_local()
    if llm is None:
        st.error("LLM modeli yüklenemediği için RAG zinciri oluşturulamıyor.")
        return None

    def rag_answer(query):
        if _vectordb is None:
            return "Veritabanı mevcut değil, cevap verilemiyor."
        try:
            docs = _vectordb.similarity_search(query, k=3)
            if not docs:
                return "İlgili bilgi bulunamadı."
            context_text = "\n".join([doc.page_content for doc in docs])
            input_text = f"Context: {context_text}\n\nQuestion: {query}\nAnswer:"
            result = llm(input_text)
            return result[0]["generated_text"]
        except Exception as e:
            return f"Cevap üretilirken bir hata oluştu: {e}"

    return rag_answer

# ================== Streamlit UI ==================
st.title("Yoğurtlu Mutfak Asistanı - Offline RAG 🌐")

# Lokal FAISS index yükle, yoksa PDF’den oluştur
vectordb = load_local_vectordb()

if vectordb is None:
    st.warning("Yerel vektör veritabanı bulunamadı veya bozuk. PDF dosyalarınızdan oluşturuluyor...")
    vectordb = create_and_save_vectordb()

if vectordb is not None:
    rag_chain = create_rag_chain(vectordb)
    if rag_chain: # Check if rag_chain was created successfully
        user_question = st.text_input("Aradığınız malzemeyi veya tarifi yazın:")
        if user_question:
            with st.spinner("Cevap hazırlanıyor..."):
                answer = rag_chain(user_question)
                st.markdown(f"**Cevap:** {answer}")
else:
    st.warning("Vektör veritabanı yüklenemedi veya oluşturulamadı. Lütfen PDF klasörünüzü kontrol edin.")
