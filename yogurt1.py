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
target_lang = languages[selected_lang]

# ================== Vektör DB Yapılandırması ==================
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index"
PDF_FOLDER = "pdfs"
GROQ_MODEL = "llama-3.1-8b-instant" # Kullanacağımız Groq modeli

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
            # Sadece PyPDFLoader değil, tüm dosya yükleyicilerini desteklemek için (Gerekliyse)
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
            # Eğer FAISS index'in formatı değişmişse bu hatayı alabiliriz.
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
    
    # LangChain-Groq entegrasyonu, doğrudan os.getenv() içindeki anahtarı kullanır.
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=512
    )
    return llm

# ================== Prompt Tanımı ==================
# PromptTemplate'i seçilen dile göre oluştur
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=PROMPT_TEMPLATES[target_lang]
)
# ================== Dil Bazlı Prompt Şablonları ==================
PROMPT_TEMPLATES = {
    "tr": """
Sen bir şef asistanısın. Aşağıda yoğurtla ilgili tarif bilgileri içeren bir metin var:

{context}

Kullanıcının verdiği malzemelere uygun, sadece yoğurt içeren tarifler öner.
Türk mutfağına öncelik ver. Malzeme listesi ve yapılış adımlarını yaz.
Sade, akıcı ve kullanıcı dostu bir dille yaz. Gerekiyorsa alternatif malzemeler de öner.

Malzemeler: {question}
""",
    "en": """
You are a chef assistant. Below is a text containing yogurt-based recipe information:

{context}

Suggest recipes that include only yogurt and match the user's provided ingredients.
Prioritize Turkish cuisine. Provide a clear list of ingredients and step-by-step instructions.
Use simple, fluent, and user-friendly language. Suggest alternative ingredients if needed.

Ingredients: {question}
"""
}
)

# ================== RAG Zinciri (LCEL) ==================
def create_rag_chain_lcel(_vectordb, lang="tr"):
    """LCEL kullanarak RAG zincirini oluşturur."""
    llm = get_groq_llm()
    if llm is None:
        return None

    retriever = _vectordb.as_retriever(search_kwargs={"k": 3})

    # Dil bazlı prompt
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATES[lang]
    )

    rag_chain = (
        {"context": retriever | (lambda docs: "\n".join([doc.page_content for doc in docs])), 
         "question": RunnablePassthrough()
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
    rag_chain = create_rag_chain_lcel(vectordb, lang=target_lang)  # 👈 burada dil geçiriliyor
    
    if rag_chain is not None:
        # Dil bazlı input etiketi
        input_labels = {"tr": "Aradığınız malzemeyi veya tarifi yazın:", "en": "Enter the ingredients or recipe you're looking for:"}
        user_question = st.text_input(input_labels[target_lang])
        
        if user_question:
            with st.spinner("Cevap hazırlanıyor (Groq API)..." if target_lang == "tr" else "Generating answer (Groq API)..."):
                answer = rag_chain.invoke(user_question)
                label = "**Cevap:**" if target_lang == "tr" else "**Answer:**"
                st.markdown(f"{label} {answer}")
    else:
        st.error("RAG zinciri başlatılamadı (GROQ_API_KEY eksik olabilir)." if target_lang == "tr" 
                  else "Failed to initialize RAG chain (GROQ_API_KEY may be missing).")
else:
    st.warning("Vektör veritabanı yüklenemedi. Lütfen PDF klasörünüzü kontrol edin." if target_lang == "tr"
               else "Vector database could not be loaded. Please check your PDF folder.")
