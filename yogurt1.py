import os
import streamlit as st
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# ======= PDF Yükleme ve Vektör DB Oluşturma =======
@st.cache_data
def load_vectordb(pdf_folder="pdfs", db_path="faiss_index"):
    pdf_folder_path = Path(pdf_folder)
    if not pdf_folder_path.exists():
        st.error(f"{pdf_folder} klasörü bulunamadı.")
        return None

    # PDF dosyalarını yükle
    docs = []
    for pdf_file in pdf_folder_path.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        docs.extend(loader.load())

    if not docs:
        st.warning("PDF dosyası bulunamadı.")
        return None

    # FAISS ile vektör DB oluştur
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(docs, embeddings)
    vectordb.save_local(db_path)
    st.success("Vektör veritabanı oluşturuldu ve kaydedildi ✅")
    return vectordb

@st.cache_data
def load_vectordb_local(db_path="faiss_index"):
    if Path(db_path).exists():
        return FAISS.load_local(db_path, embedding=None)
    return None

# ======= HuggingFace Local Model =======
@st.cache_resource
def get_llm_local():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.2
    )

# ======= RAG Chain Oluşturma =======
@st.cache_resource
def create_rag_chain(_vectordb):
    llm = get_llm_local()

    # Basit retrieval + generation pipeline
    def rag_answer(query):
        # Benzer dokümanları bul
        docs = vectordb.similarity_search(query, k=3)
        context_text = "\n".join([doc.page_content for doc in docs])
        input_text = f"Context: {context_text}\n\nQuestion: {query}\nAnswer:"
        result = llm(input_text)
        return result[0]["generated_text"]

    return rag_answer

# ======= Streamlit UI =======
st.title("Yoğurtlu Mutfak Asistanı - Offline RAG")

# PDF yükleme veya vektör DB yükleme
vectordb = load_vectordb_local() or load_vectordb()

if vectordb is not None:
    rag_chain = create_rag_chain(vectordb)
    user_question = st.text_input("Sorunuz:")
    if user_question:
        answer = rag_chain(user_question)
        st.markdown(f"**Cevap:** {answer}")
else:
    st.warning("Vektör veritabanı yüklenemedi.")


