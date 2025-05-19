import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# GOOGLE API Key'i kontrol et
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print("Google API Key:", GOOGLE_API_KEY)  # Test amaçlı, sonra kaldırabilirsiniz

# Embed modeli tanımla
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# PDF dosyasını yükle
def load_pdf_docs():
    pdf_path = "data/yogurt-uygarligi.pdf"
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print("Toplam belge sayısı:", len(docs))

    yogurt_docs = [doc for doc in docs if "yoğurt" in doc.page_content.lower()]
    print("Yoğurt geçen belge sayısı:", len(yogurt_docs))
    return yogurt_docs

# Metinleri parçala
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    print("Parçalanmış metin sayısı:", len(split_docs))
    return split_docs

# Vektör DB oluştur
# @st.cache_resource  # TEST AŞAMASINDA YORUM SATIRINDA
def load_vectordb():
    yogurt_docs = load_pdf_docs()
    split_docs = split_documents(yogurt_docs)

    # Her metin parçası için embed denemesi yap
    for i, doc in enumerate(split_docs):
        try:
            text = doc.page_content.strip()
            if not text:
                print(f"[{i}] Boş içerik atlandı.")
                continue
            _ = embedding.embed_query(text)
        except Exception as e:
            print(f"[{i}] Embed hatası: {e}")
            continue

    # Eğer hepsi embedlenebiliyorsa, FAISS vektör veritabanı oluştur
    vectordb = FAISS.from_documents(split_docs, embedding)
    print("Vektör veritabanı başarıyla oluşturuldu.")
    return vectordb

# Test embed fonksiyonu (tek metin için)
def test_single_embed():
    text = "Yoğurtla yapılan yemekleri çok severim."
    try:
        result = embedding.embed_query(text)
        print("Başarılı! İlk 5 vektör değeri:", result[:5])
    except Exception as e:
        print("Embed hatası (tek örnek):", e)

# Ana akış
if __name__ == "__main__":
    test_single_embed()
    vectordb = load_vectordb()
