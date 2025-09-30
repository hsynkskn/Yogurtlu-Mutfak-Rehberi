import os
from pathlib import Path
import streamlit as st
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# ================== Dil Seçimi ==================
languages = {
    "Türkçe TR": "tr"
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
GROQ_MODEL = "llama-3.1-8b-instant"

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

@st.cache_data
def create_and_save_vectordb(_pdf_folder=PDF_FOLDER, _db_path=FAISS_INDEX_PATH):
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
    embeddings = get_embeddings()
    if Path(_db_path).exists():
        try:
            return FAISS.load_local(_db_path, embeddings)
        except ValueError as e:
            st.warning(f"FAISS index yüklenemedi: {e}")
            return None
    return None

# ================== Prompt Tanımı ==================
SYSTEM_PROMPT = """
Sen bir şef asistanısın. Görevin, aşağıda sunulan yoğurtla ilgili tarif metinlerini ('Context') kullanarak,
kullanıcının verdiği malzemelere uygun tarifler önermektir.

Kurallar:
1. Yalnızca yoğurt içeren tarifler öner.
2. Türk mutfağına öncelik ver.
3. Yanıt, net bir 'Malzemeler' listesi ve 'Yapılış Adımları' içermelidir.
4. Gerekliyse, bağlamda (Context) bulunmayan ancak mantıklı olan alternatif malzemeler de öner.
5. Sade, akıcı ve kullanıcı dostu bir dille **Türkçe** olarak yaz.
"""
# ================== Groq API ==================
# Mevcut get_groq_client() fonksiyonu buraya kalsın...

def query_groq(prompt: str, context: str): 
    client = get_groq_client()
    if client is None:
        return "Groq API istemcisi başlatılamadı."

    # Groq Chat API'si için Gelişmiş Mesaj Yapısı
    # 1. System Prompt: Modelin rolünü ve kurallarını belirler.
    # 2. User Prompt: Bağlamı (Context) ve Kullanıcı Sorusunu iletir.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nMalzemeler: {prompt}"}
    ]

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL, 
            messages=messages, # <-- Güncellenmiş mesaj yapısı
            temperature=0.1,
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Groq API hatası: {e}"

# ================== RAG Chain (Güncellenmiş) ==================
def create_rag_chain(_vectordb):
    def rag_answer(query):
        if _vectordb is None:
            return "Veritabanı mevcut değil."
        try:
            # 1. Retrieval (Belge Alma)
            docs = _vectordb.similarity_search(query, k=3)
            if not docs:
                return "İlgili bilgi bulunamadı. Lütfen daha genel malzemelerle tekrar deneyin."
            
            context_text = "\n".join([doc.page_content for doc in docs])
            
            # 2. Generation (Sorgulama)
            # HATA DÜZELTİLDİ: Artık iki parametreyi de gönderiyoruz: (kullanıcı sorusu, context)
            return query_groq(query, context_text) # <-- Bu satır düzeltildi
            
        except Exception as e:
            # Bu hata yakalama bloğu, artık sizin aldığınız hatayı yakalamalı ve 
            # düzgün bir şekilde raporlamalıydı, ama şimdi onu da düzelttik.
            return f"Cevap üretilirken hata: {e}"
    return rag_answer
# ================== Streamlit UI ==================
st.title("🥛 Yoğurtlu Mutfak Asistanı - Groq RAG")

vectordb = load_local_vectordb()
if vectordb is None:
    st.warning("Yerel FAISS index bulunamadı. PDF’lerden oluşturuluyor...")
    vectordb = create_and_save_vectordb()

if vectordb is not None:
    rag_chain = create_rag_chain(vectordb)
    user_question = st.text_input("Aradığınız malzemeyi veya tarifi yazın:")
    if user_question:
        with st.spinner("Cevap hazırlanıyor (Groq API)..."):
            answer = rag_chain(user_question)
            st.markdown(f"**Cevap:** {answer}")
else:
    st.warning("Vektör veritabanı yüklenemedi. Lütfen PDF klasörünüzü kontrol edin.")
