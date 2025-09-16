import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader  # <-- Burada değişiklik
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema import HumanMessage, AIMessage

# --- API Anahtarı Yönetimi ---
try:
    gemini_api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
except KeyError:
    st.error("GOOGLE_API_KEY Streamlit Secrets'ta bulunamadı. Lütfen Streamlit Cloud'da 'Secrets' bölümünü kontrol edin.")
    st.stop()

# --- Embedding ve LLM Modellerini Yükleme ---
@st.cache_resource
def get_embedding_model():
    try:
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    except Exception as e:
        st.error(f"Embedding modeli yüklenirken hata oluştu: {e}")
        st.stop()

@st.cache_resource
def get_llm_model():
    try:
        return ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    except Exception as e:
        st.error(f"LLM modeli yüklenirken hata oluştu: {e}")
        st.stop()

# --- Vektör Veritabanını Yükleme/Oluşturma ---
@st.cache_resource(show_spinner="Vektör veritabanı hazırlanıyor...")
def load_vectordb():
    st.info("PDF dosyası yükleniyor ve vektör veritabanı oluşturuluyor...")

    # 1. PDF Dosyasını Yükle
    try:
        loader = PyPDFLoader("yogurt-uygarligi.pdf")  # <-- Burada değişiklik
        documents = loader.load()
    except FileNotFoundError:
        st.error("yogurt-uygarligi.pdf dosyası bulunamadı. Lütfen dosyanın uygulamanızla aynı dizinde olduğundan emin olun.")
        st.stop()
    except Exception as e:
        st.error(f"PDF dosyası yüklenirken hata oluştu: {e}")
        st.stop()

    # 2. Metni Parçalara Ayır
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    yogurt_docs = text_splitter.split_documents(documents)

    # 3. Embedding Modelini Kullanarak FAISS Oluştur
    embedding = get_embedding_model()
    
    try:
        vectordb = FAISS.from_documents(yogurt_docs, embedding)
        st.success("Vektör veritabanı başarıyla hazırlandı!")
        return vectordb
    except Exception as e:
        st.error(f"FAISS vektör veritabanı oluşturulurken hata oluştu: {e}")
        st.stop()

# --- RAG Zinciri Oluşturma ---
@st.cache_resource
def create_rag_chain():
    llm = get_llm_model()
    vectordb = load_vectordb()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Sen uzman bir yoğurtlu tarifler şefisin. Kullanıcının sorusunu belgelerden alınan bilgilere göre yanıtla. Yalnızca verilen belgelerdeki bilgileri kullan."),
        ("human", "{input}"),
        ("system", "Konuyla ilgili belge parçacıkları: {context}"),
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain

# --- Streamlit Uygulaması ---
st.set_page_config(page_title="Yoğurtlu Mutfak Rehberi", layout="centered")
st.title("👨‍🍳 Yoğurtlu Mutfak Rehberi")
st.write("Yoğurt ile hazırlanan tarifler hakkında bana sorular sorabilirsiniz!")

rag_chain = create_rag_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sohbet geçmişini göster
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Kullanıcıdan girdi al
user_query = st.chat_input("Yoğurtla ne yapabilirim?")

if user_query:
    st.session_state.messages.append(HumanMessage(content=user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Yanıt aranıyor..."):
            try:
                response = rag_chain.invoke({"input": user_query})
                ai_response_content = response["answer"]
                st.markdown(ai_response_content)
                st.session_state.messages.append(AIMessage(content=ai_response_content))
            except Exception as e:
                st.error(f"Yanıt alınırken hata oluştu: {e}")
