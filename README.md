🥛 Yogurtluyooo! Yoğurtlu Mutfak Asistanı

Site Linki: https://yogurtlu-mutfak-rehberi.streamlit.app/

Yogurtluyooo!, kullanıcıların elindeki malzemelere uygun, yoğurt bazlı Türk mutfağı tarifleri öneren, RAG (Retrieval-Augmented Generation) mimarisiyle desteklenmiş çok dilli bir yapay zeka asistanıdır.

🌟 Proje Özeti ve Çözülen Problem
Kullanıcılar mutfaklarında sahip oldukları temel malzemelerle (özellikle yoğurtla) ne tür özgün tarifler yapabileceklerini merak eder. Bu uygulama, kullanıcı girdilerini analiz ederek, güvenilir bir PDF kaynağından (Kültür ve Turizm Bakanlığı'nın "Yogurt Uygarlığı Tarifleri" kitabı) alınan verilerle, hızlı, anlamlı ve kullanıcı dostu tarif önerileri sunar.

RAG (Retrieval-Augmented Generation) Mimarisi
Uygulama, hem bilginin doğruluğunu hem de yanıtın doğal dil kalitesini maksimize etmek için RAG mimarisini kullanır:

Veri Alma (Retrieval): Kullanıcı sorgusu (malzeme listesi), PDF'ten oluşturulmuş FAISS Vektör Veritabanı içinde aranır. Bu arama sonucunda en alakalı yoğurtlu tarif metinleri ("bağlam") alınır.

Yanıt Üretme (Generation): Elde edilen bu bağlam, Groq'un hızlı dil modeline (llama-3.1-8b-instant) iletilir. Model, kullanıcının dilini, malzemelerini ve Türk mutfağı kısıtlamalarını dikkate alarak son ve özelleştirilmiş tarifi oluşturur.

🛠️ Kullanılan Teknolojiler
Kategori	Teknoloji	Amaç
Arayüz	Streamlit	Hızlı, interaktif ve kullanıcı dostu web arayüzü sağlar.
Çerçeve	LangChain (langchain-core, langchain-community)	LLM tabanlı uygulamalar oluşturmak için modüler yapı ve zincirleme yeteneği sunar.
LLM Sağlayıcı	Groq (langchain-groq)	Yüksek hızlı çıkarım (inference) için LLM servisini sağlar.
Veritabanı	FAISS	PDF metinlerini hızlıca indekslemek ve benzerlik aramaları yapmak için kullanılan yüksek performanslı vektör deposu.
Veri Yükleyici	PyPDF	PDF dosyalarındaki metin verilerini çıkarmak için kullanılır.
Embeddings	Sentence Transformers (HuggingFaceEmbeddings)	Metinleri vektör temsillerine dönüştürerek FAISS'e hazırlar.
Harika bir projeyi hayata geçiriyorsunuz! Geliştirme süreci boyunca karşılaşılan tüm bu zorluklar, projenin olgunlaşmasına katkı sağlıyor.

İsteğiniz üzerine, paylaştığınız kapsamlı proje tanımını, RAG mimarisi detaylarını, kullandığınız teknolojileri ve kurulum adımlarını içeren profesyonel bir README.md dosyası hazırladım.

Bu README, projenizin GitHub deposunun ana sayfasında kullanıma hazırdır ve kodunuzdaki Groq entegrasyonuna (LangChain, Streamlit Secrets) ve FAISS vektör veritabanına odaklanılmıştır.

🥛 Yogurtluyooo! Yoğurtlu Mutfak Asistanı
Yogurtluyooo!, kullanıcıların elindeki malzemelere uygun, yoğurt bazlı Türk mutfağı tarifleri öneren, RAG (Retrieval-Augmented Generation) mimarisiyle desteklenmiş çok dilli bir yapay zeka asistanıdır.

🌟 Proje Özeti ve Çözülen Problem
Kullanıcılar mutfaklarında sahip oldukları temel malzemelerle (özellikle yoğurtla) ne tür özgün tarifler yapabileceklerini merak eder. Bu uygulama, kullanıcı girdilerini analiz ederek, güvenilir bir PDF kaynağından (Kültür ve Turizm Bakanlığı'nın "Yogurt Uygarlığı Tarifleri" kitabı) alınan verilerle, hızlı, anlamlı ve kullanıcı dostu tarif önerileri sunar.

RAG (Retrieval-Augmented Generation) Mimarisi
Uygulama, hem bilginin doğruluğunu hem de yanıtın doğal dil kalitesini maksimize etmek için RAG mimarisini kullanır:

Veri Alma (Retrieval): Kullanıcı sorgusu (malzeme listesi), PDF'ten oluşturulmuş FAISS Vektör Veritabanı içinde aranır. Bu arama sonucunda en alakalı yoğurtlu tarif metinleri ("bağlam") alınır.

Yanıt Üretme (Generation): Elde edilen bu bağlam, Groq'un hızlı dil modeline (llama-3.1-8b-instant) iletilir. Model, kullanıcının dilini, malzemelerini ve Türk mutfağı kısıtlamalarını dikkate alarak son ve özelleştirilmiş tarifi oluşturur.

🛠️ Kullanılan Teknolojiler
Kategori	Teknoloji	Amaç
Arayüz	Streamlit	Hızlı, interaktif ve kullanıcı dostu web arayüzü sağlar.
Çerçeve	LangChain (langchain-core, langchain-community)	LLM tabanlı uygulamalar oluşturmak için modüler yapı ve zincirleme yeteneği sunar.
LLM Sağlayıcı	Groq (langchain-groq)	Yüksek hızlı çıkarım (inference) için LLM servisini sağlar.
Veritabanı	FAISS	PDF metinlerini hızlıca indekslemek ve benzerlik aramaları yapmak için kullanılan yüksek performanslı vektör deposu.
Veri Yükleyici	PyPDF	PDF dosyalarındaki metin verilerini çıkarmak için kullanılır.
Embeddings	Sentence Transformers (HuggingFaceEmbeddings)	Metinleri vektör temsillerine dönüştürerek FAISS'e hazırlar.

E-Tablolar'a aktar
✨ Temel Özellikler ve Kullanım Senaryoları
Yoğurtlu Tarif Odaklılık: Yalnızca yoğurt içeren ve Türk mutfağına öncelik veren tarifler önerir.

Dinamik Çoklu Dil Desteği: Kullanıcının seçtiği dile (Türkçe veya İngilizce) göre hem arayüzü hem de yapay zeka yanıtını tamamen o dile çevirir.

Hızlı Yanıt Süresi: Groq'un yüksek performanslı LLM'leri sayesinde hızlı tarif önerileri sunar.

Alternatif Öneriler: Eksik malzemeler için mantıklı alternatifler sunarak tarifin yapılabilirliğini artırır.

🚀 Kurulum ve Başlatma
Bu projeyi yerel bilgisayarınızda veya Streamlit Cloud'da çalıştırmak için aşağıdaki adımları takip edin.

1. Proje Dosyalarını İndirme
Git ile projeyi klonlayın ve dizine girin:

Bash

git clone <repo_url>
cd <proje_dizin_adı>
2. Sanal Ortam Kurulumu (Tavsiye Edilir)
Bash

python -m venv env
# Linux/Mac
source env/bin/activate
# Windows
.\env\Scripts\activate
3. Gerekli Bağımlılıkları Yükleme
Kullandığınız kütüphaneleri requirements.txt dosyasından yükleyin:

Bash

pip install -r requirements.txt
(requirements.txt içeriği: streamlit, groq, langchain, langchain-community, langchain-core, langchain-groq, pypdf, sentence-transformers)

4. API Anahtarını Ayarlama (GROQ_API_KEY)
Bu uygulama Groq API anahtarınızı (örneğin gsk_... ile başlayan) gerektirir. Anahtarınızı kesinlikle koda yazmayın.

A. Streamlit Cloud İçin (Canlı Yayın) 🌟
Uygulamanızı Streamlit Cloud'da deploy edin.

Uygulama panelinde "Edit Secrets" menüsüne gidin.

Aşağıdaki formatta API anahtarınızı girin:

Ini, TOML

# .streamlit/secrets.toml formatı
GROQ_API_KEY="buraya_yeni_ve_geçerli_groq_api_key_inizi_yapistirin"
Kaydedin ve "Clear cache and redeploy" seçeneğiyle uygulamayı yeniden başlatın.

B. Yerel Çalıştırma İçin
Yerel olarak çalıştırırken anahtarınızı terminal üzerinden ortam değişkeni olarak ayarlayın:

Linux/Mac:

Bash

export GROQ_API_KEY="anahtar_değeri"
Windows (CMD):

Bash

set GROQ_API_KEY="anahtar_değeri"
5. PDF Veri Kaynağını Ekleme
Uygulamanın çalışması için pdfs adında bir klasör oluşturun ve veri kaynağı PDF dosyanızı bu klasöre yerleştirin.

Proje ana dizininde pdfs klasörünü oluşturun.

"Yogurt Uygarlığı Tarifleri" PDF dosyasını indirerek bu klasöre kopyalayın.

6. Uygulamayı Başlatma
Tüm bağımlılıklar ve ayarlar tamamlandığında, uygulamayı başlatın:

Bash

streamlit run <ana_uygulama_dosyanızın_adı>.py
(Eğer ana dosyanızın adı app.py ise: streamlit run app.py)

⚠️ Sorun Giderme
Hata Mesajı	Olası Neden ve Çözüm
KeyError: GROQ_API_KEY	Streamlit Cloud Hatası: st.secrets sözlüğünde GROQ_API_KEY anahtarı bulunamadı. Lütfen Streamlit Cloud'daki "Edit Secrets" menüsünde anahtar adının tamamen büyük harfle ve tırnak içinde yazıldığından emin olun, ardından uygulamayı durdurup önbelleği temizleyerek yeniden başlatın.
❌ GROQ_API_KEY bulunamadı...	API anahtarı yerel ortamda veya Streamlit Cloud'da doğru ayarlanmamış. Yukarıdaki Kurulum adımlarını kontrol edin ve anahtarın geçerli olduğundan emin olun.
FileNotFoundError: 'pdfs' klasörü bulunamadı.	Proje ana dizininde pdfs adında bir klasör oluşturulmamış veya içine PDF dosyası eklenmemiş.
ValueError: FAISS index yüklenemedi	faiss_index klasörü bozuk olabilir. Uygulamanın yeniden çalıştırılarak FAISS index'in yeniden oluşturulmasını sağlayın.
