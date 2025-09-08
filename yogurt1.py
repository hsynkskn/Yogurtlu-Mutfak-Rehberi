import os
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
import streamlit as st

# ===== Güncel LangChain importları =====
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# === Streamlit Sayfa Ayarı ===
st.set_page_config(page_title="Yoğurtlu Mutfak Rehberi", page_icon="🍳")

# === Ortam Değişkenlerini Yükle ===
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# === Dil Seçenekleri ===
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
    selected_lang = st.radio("🌐 Language:", options=list(languages.keys()), index=0, horizontal=True)
target_lang = languages[selected_lang]

def translate(text, target_lang):
    if target_lang == "tr":
        return text
    return GoogleTranslator(source='auto', target=target_lang).translate(text)

# === Uygulama Başlığı ===
st.title(translate("👨🏻‍🍳 Yoğurtlu Mutfak Rehberi ", target_lang))
st.subheader(translate("Malzeme girişinize göre yoğurtlu tarifler önerilir", target_lang))

# === PDF ve FAISS VectorStore ===
pdf_path = r"C:\Users\SLAYER\OneDrive\Desktop\Python Çalışma\Yoğurtlu Mutfak Rehberi\yogurt-uygarligi.pdf"

# Dosya yolunu kontrol et
if not os.path.exists(pdf_path):
    st.error(f"❌ PDF dosyası bulunamadı: {pdf_path}")
    st.stop()

@st.cache_resource
def load_vectordb():
    # Embedding tanımı
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    # PDF yükleme
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Sadece yoğurt içeren sayfalar
    yogurt_docs = [doc for doc in docs if "yoğurt" in doc.page_content.lower()]

    # FAISS vectorstore
    vectordb = FAISS.from_documents(yogurt_docs, embedding)
    return vectordb

vectordb = load_vectordb()
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# === LLM Tanımı ===
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY
)

# === Prompt Tanımı ===
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

# === QA Zinciri Tanımı ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt_template}
)

# === Kullanıcı Girişi ===
input_label = translate("Malzemelerinizi yazın...", target_lang)
user_input = st.chat_input(input_label)

if "messages" not in st.session_state:
    st.session_state.messages = []

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Kullanıcı girişi önce Türkçeye çevrilir
    query_in_tr = GoogleTranslator(source='auto', target='tr').translate(user_input)
    with st.chat_message("assistant"):
        with st.spinner(translate("Tarif hazırlanıyor...", target_lang)):
            try:
                result = qa_chain.run(query_in_tr)
                result_translated = translate(result, target_lang)
                st.write(result_translated)
                st.session_state.messages.append({"role": "assistant", "content": result_translated})
            except Exception as e:
                st.error("❌ " + str(e))
