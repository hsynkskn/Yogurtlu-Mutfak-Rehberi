Yoğurtlu Mutfak Rehberi

Site Linki:https://yogurtlu-mutfak-rehberi.streamlit.app/

Projenin Kapsamı:
Bu proje, kullanıcıların belirttiği malzemelere uygun yoğurtlu tarifler öneren bir Yogurt Tarif Asistanı uygulaması geliştirmeyi amaçlıyor. Kullanıcı, uygulamaya girdiklerinde sahip oldukları malzemeleri belirtir ve sistem, yalnızca bu malzemeleri kullanarak, özellikle yoğurt içeren Türk mutfağı tariflerini önerir. Proje, Streamlit tabanlı bir arayüzle, LangChain ve Google Generative AI teknolojileriyle desteklenen bir yapay zeka çözümü sunuyor.

Çözülen Problem:
Kullanıcılar, mutfakta hangi malzemelere sahip olduklarını bilseler de, bu malzemelerle ne tür tarifler hazırlayabileceklerini bazen bilemezler. Özellikle yoğurt gibi temel bir malzeme ile hangi tariflerin yapılabileceğini öğrenmek isteyenler için, bu uygulama doğru, anlamlı ve kullanıcı dostu tarif önerileri sunarak bu boşluğu doldurur. Türk mutfağına özgü tarifler önerilerek, kullanıcıların kültürel bağlamda anlamlı tarifler elde etmeleri sağlanır.

RAG (Retrieval-Augmented Generation) Mimarisi:
Proje, RAG mimarisi kullanılarak yapılandırılmıştır. Bu mimari, bilgiyi önceden büyük veritabanlarında aramak ve ardından bu bilgiyi kullanarak doğru cevabı geliştirmek (generate) için iki aşamalı bir yaklaşım sunar. RAG mimarisi, aşağıdaki adımlarla çalışır:

Veri Alma (Retrieval):

Uygulama, bir PDF dosyasındaki yoğurtlu tarif bilgilerini Chroma vektör veritabanına yükler. Bu veritabanı, içerdiği metinlere dayalı olarak arama yapabilmek için önceden dizine eklenmiş verilere sahiptir.

Kullanıcı bir malzeme listesi girdiğinde, bu listeye uygun yoğurtla ilgili tariflerin veritabanında aranması sağlanır.

Yanıt Üretme (Generation):

Arama sonucunda elde edilen bağlam (yani, tarifle ilgili metinler) bir GPT modeline iletilir. Burada, model, kullanıcının sağladığı malzeme listesine göre yalnızca yoğurt içeren tarifleri önerir ve bu tarifleri basit, kullanıcı dostu bir dilde sunar.

Model, özellikle Türk mutfağına odaklanır ve alternatif malzeme önerileri de yapabilir.

Kapsamlı Bir Çözüm:

RAG mimarisi, veritabanından doğru bilgiyi alırken, aynı zamanda kullanıcıyı anlamak ve önerileri doğal bir şekilde oluşturmak için güçlü dil modellerini kullanır. Bu, bilgiye dayalı bir öneri sisteminin hem doğru hem de kullanıcı odaklı olmasını sağlar.

Kullanılan Teknolojiler:
1. Streamlit
Amaç: Web uygulamaları geliştirmek için kullanılan bir açık kaynak kütüphanedir.
Kullanım: Veri bilimi ve makine öğrenmesi projelerinde hızlı bir şekilde interaktif web uygulamaları geliştirmeye olanak tanır. Projende, kullanıcıların malzemelerini girmesi ve tarif önerileri alması için kullanılan bir arayüz sağlar.

2. LangChain (langchain>=0.1.16)
Amaç: Dil modellerini (LLM) kullanarak veri işleme ve yapay zeka tabanlı uygulamalar geliştirmeye yönelik bir framework'tür.
Kullanım: LangChain, verileri işlemek ve doğal dil işleme (NLP) görevlerini yerine getirmek için çoklu dil modeli entegrasyonlarını sağlar. Projende, bir kullanıcının girdiği malzeme listesini analiz eden ve yoğurtla ilgili tarifler öneren bir yapay zeka asistanı oluşturmak için kullanılır.

3. LangChain Google Generative AI (langchain-google-genai)
Amaç: Google'ın generatif AI teknolojilerini LangChain ile entegre etmek için kullanılan bir kütüphanedir.
Kullanım: Google’ın dil modelleri ve yapay zeka özelliklerinden yararlanarak metin tabanlı uygulamalar geliştirmek için kullanılır. Bu kütüphane, projendeki generatif AI modelinin çalışmasını sağlamak için kullanılır (örneğin, GoogleGenerativeAIEmbeddings ve ChatGoogleGenerativeAI sınıfları).

4. Google Generative AI (google-generativeai)
Amaç: Google’ın generatif yapay zeka API’larını kullanmak için kullanılan bir kütüphanedir.
Kullanım: Projede, kullanıcıya tarif önerileri sunarken, Google’ın yapay zeka dil modelinden yararlanmak için bu API kullanılır. ChatGoogleGenerativeAI sınıfı, bir AI asistanının oluşturulmasında temel rol oynar.

5. ChromaDB (chromadb)
Amaç: Vektör tabanlı veritabanı ve veritabanı yönetim sistemidir. FAISS ve diğer vektör veritabanı çözümleriyle benzer şekilde çalışır.
Kullanım: Projende, kullanıcının verdiği malzemeler ile benzer içerikleri bulmak için vektör veri yapıları kullanılır. ChromaDB, vektör tabanlı arama ve benzerlik hesaplamaları için kullanılır. Verilerin hızlıca indekslenmesi ve sorgulanması için bu sistem kullanılabilir.

6. Deep-Translator (deep-translator)
Amaç: Çoklu dil desteği sunan bir çeviri API’sıdır.
Kullanım: Projenin çok dilli destek sağlaması için kullanılır. Kullanıcıların girdikleri metinleri farklı dillere çevirmek ve cevabı hedef dile çevirmek için bu kütüphane kullanılır. Örneğin, bir kullanıcı Türkçe yazarsa, yanıtı diğer dillerde sunmak için kullanılır.

7. PyPDF (pypdf)
Amaç: PDF dosyalarını okumak ve işlemek için kullanılan bir kütüphanedir.
Kullanım: Projede, kullanıcılar tarafından yüklenen PDF dosyalarındaki içeriği okumak için kullanılır. PyPDFLoader sınıfı ile PDF dosyasındaki metin verisi çıkarılır ve bu veriler, yoğurtla ilgili tarifleri içerik olarak seçmek için analiz edilir.

8. LangChain Community (langchain-community)
Amaç: LangChain framework’ünün topluluk sürümüdür ve daha fazla özellik, entegrasyon ve araç sağlar.
Kullanım: LangChain'in açık kaynak sürümü olarak, topluluk tarafından geliştirilen ek özellikleri ve güncellemeleri kullanmanıza olanak tanır. Bu, projedeki temel LangChain işlevselliklerini güçlendirir ve daha fazla araç ve veri kaynağını entegre etmeye yardımcı olur.

9. Python Dotenv (python-dotenv)
Amaç: Ortam değişkenlerini yüklemek için kullanılan bir kütüphanedir.
Kullanım: API anahtarları gibi hassas bilgilerin güvenli bir şekilde yönetilmesi için kullanılır. .env dosyasından ortam değişkenlerini okuyarak, bu bilgilerin kodun içinde sert bir şekilde yazılmasını engeller. Google API anahtarları gibi bilgiler, .env dosyasından yüklenir.

10. FAISS (faiss-cpu)
Amaç: Facebook AI Research tarafından geliştirilen, yüksek performanslı vektör arama kütüphanesidir.
Kullanım: Projende, benzer belgeleri ve metinleri hızlıca aramak ve sıralamak için kullanılacak. FAISS, veritabanındaki vektörleri hızlıca indeksleyip sorgulamak için kullanılacak.

Veri toplama süreci, aşağıdaki adımlarla gerçekleştirilmiştir:

Veri Kaynağı:

Kaynak, Yogurt Uygarlığı Tarifleri başlıklı PDF dosyasını içermektedir. Bu dosyaya, Kültür ve Turizm Bakanlığı'nın e-kitap platformu üzerinden erişilmiştir. Kitap, Türk mutfağındaki yoğurtlu tarifler hakkında kapsamlı bilgiler sunmakta olup, hem geleneksel hem de modern tariflere yer vermektedir.

PDF dosyasına şu bağlantıdan ulaşılabilir: Yogurt Uygarlığı Tarifleri PDF.

Veri Toplama:

PyPDF ve LangChain kullanılarak, bu PDF dosyasındaki metinler işlenmiştir. PyPDFLoader aracılığıyla, PDF dosyasındaki tüm sayfalar yüklenmiş ve içeriği metin formatına dönüştürülmüştür.

Ardından, metinlerde "yoğurt" kelimesi geçmeyen içerikler çıkarılmış ve yalnızca yoğurtla ilgili tarifler içeren bölümler seçilmiştir. Bu filtreleme, yalnızca yoğurtlu tarifleri içeren verilerin kullanılmasını sağlamak amacıyla yapılmıştır.

Veri Hazırlama ve Vektörleştirme:

Elde edilen metinler, ChromaDB gibi bir vektör veritabanına aktarılarak vektörleştirilmiştir. Bu işlem, metinlerin sayısal temsillerinin oluşturulmasını sağlayarak, metinler arası benzerliklerin ve ilişkilerin kolayca analiz edilmesine olanak tanımaktadır.

Google Generative AI Embeddings kullanılarak her bir tarifin içeriği gömme (embedding) vektörlerine dönüştürülmüş ve Chroma veritabanında depolanmıştır. Bu sayede, kullanıcının sorduğu malzemelere uygun tarifler, hızlı bir şekilde ve yüksek doğrulukla bulunabilmektedir.

Veri Kaynağının Geçerliliği:

Kullanılan veri kaynağı, Kültür ve Turizm Bakanlığı tarafından sağlandığı için güvenilir ve resmi bir kaynaktır. Yoğurtlu tariflerin Türk mutfağındaki kültürel ve geleneksel önemi göz önünde bulundurularak, bu veri seti projede önemli bir referans olarak seçilmiştir.

Özellikler / Kullanım Senaryosu
Yogurtluyooo! V1 uygulaması, kullanıcıların malzeme listeleriyle yoğurtlu tarifler aramaları için güçlü bir AI destekli asistan sunmaktadır. Uygulama, LangChain, Google Generative AI, ChromaDB ve deep-translator gibi araçları kullanarak, kullanıcılara özelleştirilmiş tarif önerileri sağlamaktadır. Bu uygulamanın başlıca özellikleri ve kullanım senaryoları aşağıda açıklanmıştır:

1. Multilingual Destek:
Uygulama, Türkçe, İngilizce, Fransızca, Almanca, İspanyolca ve Rusça olmak üzere çoklu dil desteği sunar. Kullanıcılar, tercihlerine göre istediği dili seçebilir ve uygulama, tüm içerikleri seçilen dilde sunar.

Google Translator entegrasyonu sayesinde, verilen içerikler seçilen dile çevrilir ve her dilde doğru tarif önerileri sunulur.

2. Yoğurtlu Tarif Önerileri:
Kullanıcılar, malzemeleri girdiklerinde, sistem yalnızca yoğurtlu tarifler önerir. Bu özellik, sadece yoğurt içeren yemek tarifleriyle sınırlı olup, daha fazla çeşitlilik için malzeme girişi özelleştirilebilir.

Uygulama, kullanıcının verdiği malzemelere uygun tarifleri bulmak için Google Generative AI'yi kullanarak akıllı cevaplar üretir.

3. Tarif Veritabanı (PDF Kaynağı):
Yogurt Uygarlığı Tarifleri adlı PDF kitap, uygulamanın veri kaynağı olarak kullanılır. Kitapta yer alan yoğurtla yapılan tarifler ChromaDB'ye vektörleştirilmiş ve depolanmıştır. Bu sayede, kullanıcıların malzemeleri ile uyumlu tarifler hızlıca bulunur.

PyPDFLoader kullanılarak PDF dosyasındaki metinler işlenmiş ve sadece yoğurt içeren tarifler seçilerek vektör veritabanına aktarılmıştır.

4. Hızlı ve Doğru Yanıtlar:
Uygulama, kullanıcının yazdığı malzeme listesini analiz ederek, doğru tarifleri Retrieval-Augmented Generation (RAG) yöntemini kullanarak bulur. Bu yöntem, LangChain'in RetrievalQA zincirini kullanarak soruları metinlerle eşleştirir ve anlamlı cevaplar oluşturur.

Kullanıcı dostu bir arayüz ve görsel öğelerle (yemek tarifinin adımlarını ve malzemelerini) sonuçlar hızlı ve kullanıcı dostu bir şekilde sunulur.

5. Alternatif Malzeme Önerileri:
Eğer kullanıcının girdiği malzeme tarifin gereksinimleriyle uyumlu değilse, sistem alternatif malzeme önerileri sunar. Bu özellik, özellikle malzemelerin eksik olduğu durumlarda kullanıcılara alternatif seçenekler sunarak tarifin tamamlanmasını sağlar.

Bu özellik, kullanıcıya daha fazla seçenek sunarak tarifin esnekliğini artırır.

Kullanım Senaryoları:
Senaryo 1: Kullanıcı Malzeme Girişi Yapıyor ve Tarif Alıyor
Bir kullanıcı, elinde bulunan malzemeleri (örneğin, yoğurt, domates, peynir) yazarak tarif önerisi almak istiyor. Sistem, verilen malzemelere uygun, yalnızca yoğurt içeren tarifleri sunar. Uygulama, hızlı bir şekilde yemek tarifini ve adımları kullanıcıya gösterir.

Kullanıcı Girişi: "Yoğurt, domates, peynir"

Yanıt: "Yoğurtlu Domates Salatası Tarifi" — Tarifin malzemeleri ve yapılış adımları verilir.

Senaryo 2: Kullanıcı Farklı Dillerde Tarif Arıyor
Bir kullanıcı İngilizce dilinde tarif aramak istiyor. Uygulama, Google Translator entegrasyonu sayesinde, İngilizce olarak yazılan tarifi doğru şekilde çevirir ve önerileri sunar.

Kullanıcı Girişi: "I have yogurt, cucumbers, and garlic."

Yanıt: "Yogurt Cucumber Salad Recipe" — Tarifi ve malzemeleri İngilizce olarak sağlar.

Senaryo 3: Kullanıcı Alternatif Malzeme İstiyor
Bir kullanıcı, tarifte belirtilen bazı malzemeleri temin edememiştir ve alternatif malzemeler arar. Uygulama, bu durumda malzeme alternatifi sunarak tarifin eksiksiz yapılmasını sağlar.

Kullanıcı Girişi: "I have yogurt, but I don't have honey. Can you suggest an alternative?"

Yanıt: "You can use agave syrup or maple syrup as an alternative to honey."

Senaryo 4: Kullanıcı İki Dilde Tarif İstiyor
Bir kullanıcı, bir tarifin hem Türkçe hem İngilizce olarak gösterilmesini ister. Uygulama, her iki dilde doğru ve anlamlı sonuçlar sağlar.

Kullanıcı Girişi: "Yogurt, cucumber, and mint."

Yanıt: Türkçe: "Yoğurtlu Salata Tarifi" / İngilizce: "Yogurt Cucumber Salad Recipe" şeklinde, iki dilde de tarif önerisi yapılır.

1. Python ve Gerekli Araçların Yüklenmesi
Uygulama Python ile yazılmıştır, dolayısıyla Python 3.7 veya daha yeni bir sürümünün bilgisayarınızda yüklü olması gerekmektedir.

Python yüklü değilse, Python'un resmi sitesinden Python'u indirin ve yükleyin.

pip (Python paket yöneticisi) otomatik olarak yüklenecektir, ancak yüklü olup olmadığını kontrol etmek için terminal veya komut satırında şu komutu kullanabilirsiniz:

bash
Kopyala
Düzenle
pip --version
2. Git ile Proje Dosyalarını İndirme
Projeyi GitHub'dan veya başka bir kaynaktan edindiyseniz, projeyi yerel ortamınıza indirmeniz gerekir.

Git ile projeyi indirmek için:

bash
Kopyala
Düzenle
git clone <repo_url>
cd <proje_dizin_adı>
Eğer Git kullanmıyorsanız, projeyi zip dosyası olarak indirip çıkarabilirsiniz.

3. Sanallaştırma Ortamı Kurulumu (Opsiyonel, Ancak Tavsiye Edilir)
Proje bağımlılıkları ile çakışmayı önlemek için sanal bir ortam kullanmak iyi bir uygulamadır. Python venv modülü ile sanal ortam oluşturabilirsiniz.

Sanal ortamı oluşturmak için:

bash
Kopyala
Düzenle
python -m venv env
Sanal ortamı aktifleştirmek için:

Windows:

bash
Kopyala
Düzenle
.\env\Scripts\activate
Mac/Linux:

bash
Kopyala
Düzenle
source env/bin/activate
4. Gerekli Python Paketlerini Yükleme
Projenin requirements.txt dosyasındaki gerekli bağımlılıkları yüklemek için şu komutu kullanabilirsiniz:

bash
Kopyala
Düzenle
pip install -r requirements.txt
5. .env Dosyasını Oluşturma
Projede bazı özel API anahtarları veya ortam değişkenleri kullanılmaktadır. Bunları .env dosyasına eklemeniz gerekecek. Örnek bir .env dosyasını şu şekilde oluşturabilirsiniz:

Proje dizininde yeni bir dosya oluşturun ve adını .env koyun.

.env dosyasına şu satırları ekleyin:

bash
Kopyala
Düzenle
GOOGLE_API_KEY=your_google_api_key_here
Google API Key almak için, Google Cloud Console üzerinden bir proje oluşturup, Google Cloud API'leri için gerekli erişim izinlerini ayarlayabilirsiniz.

6. PDF Dosyasını Proje Dizininize Ekleme
Proje, PDF dosyasını veri kaynağı olarak kullanır. Yogurt Uygarlığı Tarifi adlı PDF dosyasını proje dizininize eklemeniz gerekmektedir. PDF dosyasını şu linkten indirebilirsiniz:

Yogurt Uygarlığı Tarifi PDF

PDF dosyasını projedeki uygun bir dizine (örneğin, pdf_files/) koyun ve pdf_path değişkenini bu dizine göre güncelleyin.

7. Uygulamayı Çalıştırma
Bağımlılıklar ve ayarlar tamamlandığında, Streamlit uygulamanızı başlatmak için şu komutu kullanın:

bash
Kopyala
Düzenle
streamlit run app.py
app.py dosyasının adı, ana uygulama dosyasının adı olmalıdır (bu örnekte ana dosya app.py olarak varsayılmıştır).

8. Uygulamayı Kullanma
Uygulama başarıyla başlatıldıktan sonra, tarayıcınızda localhost:8501 adresine giderek uygulamayı kullanabilirsiniz. Burada:

Dil seçimi yaparak, uygulamanın farklı dillerde çalışmasını sağlayabilirsiniz.

Malzeme listesi girerek, yoğurtlu tariflerinizi alabilirsiniz.

9. Sorun Giderme
Eğer API anahtarınız geçersizse veya başka bir hata alıyorsanız, Google API Key'in doğru olduğundan emin olun.



PDF yükleme veya veritabanı ile ilgili sorunlar yaşarsanız, PDF dosyasının yolunun doğru ve erişilebilir olduğundan emin olun.

Uygulama yükleme sırasında herhangi bir bağımlılık hatası alırsanız, bağımlılıklarınızın düzgün yüklendiğini doğrulamak için pip freeze komutunu kullanarak yüklü paketleri kontrol edebilirsiniz.
