import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from deep_translator import GoogleTranslator
import nest_asyncio

# --- Gerekli ayarlar ---
nest_asyncio.apply()
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("⚠️ GOOGLE_API_KEY bulunamadı. Lütfen .env dosyanıza ekleyin.")
    st.stop()

# --- PDF yükleme ---
@st.cache_resource
def load_vectordb():
    # PDF dosyalarını oku
    docs = []
    pdf_files = ["data/yogurt-recipes.pdf"]  # PDF dosyalarının yolu
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        docs.extend(loader.load())

    # Küçük parçalara böl
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    yogurt_docs = splitter.split_documents(docs)

    # Embedding
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

    # FAISS veritabanı oluştur
    vectordb = FAISS.from_documents(yogurt_docs, embedding)
    return vectordb

vectordb = load_vectordb()

# --- Model ve QA chain ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# --- Streamlit Arayüzü ---
st.title("🥛 Yoğurtlu Mutfak Rehberi")
st.write("PDF tabanlı çok dilli yoğurt tarifleri asistanı")

languages = {
    "English": "en",
    "Français": "fr",
    "Deutsch": "de",
    "Español": "es",
    "Русский": "ru",
    "Türkçe": "tr"
}

lang_choice = st.selectbox("Dil seçin:", list(languages.keys()))

user_q = st.text_input("Bir tarif sorusu yazın:")

if st.button("Sor"):
    if user_q:
        # Kullanıcı sorusunu İngilizceye çevir
        q_en = GoogleTranslator(source=languages[lang_choice], target="en").translate(user_q)

        # Yanıt al
        response = qa_chain({"query": q_en})
        answer_en = response["result"]

        # Yanıtı seçilen dile çevir
        answer_translated = GoogleTranslator(source="en", target=languages[lang_choice]).translate(answer_en)

        st.subheader("Cevap:")
        st.write(answer_translated)

        with st.expander("Kaynaklar"):
            for doc in response["source_documents"]:
                st.write(doc.metadata.get("source", "Bilinmiyor"))
    else:
        st.warning("Lütfen bir soru yazın.")

