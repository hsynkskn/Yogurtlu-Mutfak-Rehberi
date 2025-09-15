import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema import HumanMessage, AIMessage

# --- API Anahtarı Yönetimi ---
# Streamlit Cloud'da 'Secrets' bölümüne GEMINI_API_KEY olarak anahtarınızı ekleyin.
# Yerel ortamda çalıştırıyorsanız, bu satırı yorum satırı yapıp
# 'export GEMINI_API_KEY="AIza..."' komutunu terminalde çalıştırabilirsiniz.
try:
    # Streamlit Cloud için st.secrets'tan API anahtarını alıyoruz
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = gemini_api_key # Langchain için de ayarla
except KeyError:
    st.error("GEMINI_API_KEY Streamlit Secrets'ta bulunamadı. Lütfen anahtarınızı ayarlayın.")
    st.stop() # Anahtar yoksa uygulamayı durdur

# --- Embedding ve LLM Modellerini Yükleme ---
# Bu modellerin yüklenmesi ve yapılandırılması da önbelleğe alınabilir, ancak
# genellikle tek seferlik olduğu için @st.cache_resource ile önbelleğe alınması en iyisidir.
@st.cache_resource
def get_embedding_model():
    """Embedding modelini yükler ve önbelleğe alır."""
    try:
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    except Exception as e:
        st.error(f"Embedding modeli yüklenirken hata oluştu: {e}")
        st.stop()

@st.cache_resource
def get_llm_model():
    """LLM modelini yükler ve önbelleğe alır."""
    try:
        return ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    except Exception as e:
        st.error(f"LLM modeli yüklenirken hata oluştu: {e}")
        st.stop()

# --- Vektör Veritabanını Yükleme/Oluşturma (Önbellekli) ---
@st.cache_resource(show_spinner="Vektör veritabanı hazırlanıyor...")
def load_vectordb():
    """Belgeleri yükler, böler, gömer ve FAISS vektör veritabanını oluşturur.
    Bu işlem Streamlit tarafından önbelleğe alınır, böylece kota aşımı önlenir."""
    
    st.info("Vektör veritabanı ilk kez oluşturuluyor/yükleniyor. Bu biraz zaman alabilir.")

    # 1. Metin Dosyasını Yükle
    try:
        loader = TextLoader("yogurt_tarifleri.txt", encoding="utf-8")
        documents = loader.load()
    except FileNotFoundError:
        st.error("yogurt_tarifleri.txt dosyası bulunamadı. Lütfen dosyanın uygulamanızla aynı dizinde olduğundan emin olun.")
        st.stop()
    except Exception as e:
        st.error(f"Metin dosyası yüklenirken hata oluştu: {e}")
        st.stop()

    # 2. Metni Parçalara Ayır
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    yogurt_docs = text_splitter.split_documents(documents)

    # 3. Embedding Modelini Kullanarak Gömme ve FAISS Oluşturma
    embedding = get_embedding_model() # Önbelleğe alınmış embedding modelini kullan
    
    try:
        vectordb = FAISS.from_documents(yogurt_docs, embedding)
        st.success("Vektör veritabanı başarıyla hazırlandı!")
        return vectordb
    except Exception as e:
        st.error(f"FAISS vektör veritabanı oluşturulurken hata oluştu: {e}. Detaylar: {e}")
        st.info("Bu genellikle API kotası aşıldığında veya API anahtarıyla ilgili bir sorun olduğunda meydana gelir.")
        st.stop()

# --- RAG Zinciri Oluşturma ---
def create_rag_chain():
    """RAG (Retrieval Augmented Generation) zincirini oluşturur."""
    llm = get_llm_model() # Önbelleğe alınmış LLM modelini kullan
    vectordb = load_vectordb() # Önbelleğe alınmış vektör veritabanını kullan

    # RAG için prompt şablonu
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Sen uzman bir yoğurtlu tarifler şefisin. Kullanıcının sorusunu belgelerden alınan bilgilere göre yanıtla. Yalnızca verilen belgelerdeki bilgileri kullan."),
        ("human", "{input}"),
        ("system", "Konuyla ilgili belge parçacıkları: {context}"),
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Retriever oluştur (en alakalı 4 belgeyi getir)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    
    # Retrieval zincirini oluştur
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# --- Streamlit Uygulaması ---
st.set_page_config(page_title="Yoğurtlu Mutfak Rehberi", layout="centered")
st.title("👨‍🍳 Yoğurtlu Mutfak Rehberi")
st.write("Yoğurt ile hazırlanan tarifler hakkında bana sorular sorabilirsiniz!")

# RAG zincirini yükle (ilk çalıştırmada biraz zaman alabilir)
rag_chain = create_rag_chain()

# Sohbet geçmişini saklamak için Streamlit session state kullan
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
    # Kullanıcının mesajını sohbet geçmişine ekle
    st.session_state.messages.append(HumanMessage(content=user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Yanıt aranıyor..."):
            try:
                # RAG zinciri ile yanıt al
                response = rag_chain.invoke({"input": user_query})
                ai_response_content = response["answer"]
                st.markdown(ai_response_content)
                st.session_state.messages.append(AIMessage(content=ai_response_content))
            except Exception as e:
                st.error(f"Yanıt alınırken hata oluştu: {e}")
                st.info("Lütfen bir süre bekleyip tekrar deneyin veya Google Cloud konsolunuzdaki kotanızı kontrol edin.")

