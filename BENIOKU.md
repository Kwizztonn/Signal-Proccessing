# UNV2TDMS – Simcenter Testlab UNV → TDMS (nCode) dönüştürücü

## Kurulum (tek tık)
1. Klasörü bir yere çıkarın (yolda boşluk/Türkçe karakter olmaması önerilir, ör. `C:\Araclar\UNV2TDMS`).
2. **`KUR_VE_EXE_OLUSTUR.bat`** dosyasına çift tıklayın. Betik sırasıyla şunları yapar:
   * Python 3.9+ (64-bit) arar. Bulamazsa Python 3.12'yi winget ile, olmazsa python.org'dan indirip kullanıcıya kurar.
   * `.venv` sanal ortamını kurar ve eksik kütüphaneleri indirir (numpy, npTDMS, matplotlib, pyinstaller).
   * Kendi kendini test eder: sentetik ASCII/binary UNV dosyaları üretir, dönüştürür ve veriyi kontrol eder.
   * **`UNV2TDMS.exe`** dosyasını oluşturup aynı klasöre koyar.
3. Bundan sonra yalnızca `UNV2TDMS.exe` yeterlidir. Tek dosyadır; başka bir bilgisayara da kopyalanabilir.

> İnternet erişimi yalnızca ilk kurulumda gerekir. Şirket proxy'si varsa pip için `HTTPS_PROXY` tanımlı olmalıdır.

## Kullanım
1. **UNV Dosyası Ekle** / **Klasörden Ekle** ile dosyaları listeye ekleyin. Dosyaları exe simgesinin üzerine sürükleyip bırakmak da işe yarar.
2. Bir dosya seçince kanallar listelenir: ad, tip, nokta sayısı, fs, süre, birim ve hassasiyet.
3. Bir kanal seçin:
   * Alt kutuda **tüm başlık alanları** ve UNV'deki ham 11 başlık satırı görünür.
   * Sağda **önizleme** çizilir. Uzun kanallarda tüm sinyal min/max zarfı olarak çizilir, yani tepe değerler kaybolmaz. Altta min / max / ortalama / RMS değerleri yer alır.
   * Büyüteçle bir bölgeye yakınlaşıp **"Görünen aralığı HAM veriyle çiz"** düğmesine basarsanız o aralık tek tek örneklerle çizilir.
   * Ctrl/Shift ile birden fazla kanal seçip birlikte çizebilirsiniz.
4. **TÜMÜNÜ DÖNÜŞTÜR**. Her UNV için bir `.tdms` oluşur. Çıkış klasörü boş bırakılırsa TDMS, UNV dosyasının yanına yazılır.
5. Kontrol için **TDMS Aç** ile oluşan dosyayı listeye ekleyip aynı şekilde önizleyebilirsiniz.

## Veri kaybı olmaması için yapılanlar
* **Hassasiyet korunur:** float32 olarak gelen veri float32, float64 olarak gelen veri float64 yazılır. Ölçekleme ya da yeniden örnekleme yapılmaz. Binary (58b) veri baytı baytına kopyalanır.
* ASCII (58) dosyalarda metindeki anlamlı basamak sayısına bakılır. Tek hassasiyetli bir alanda 7'den fazla basamak varsa veri float64 olarak saklanır.
* **Doğrulama (varsayılan açık):** Yazılan TDMS geri okunur. Her kanalın nokta sayısı, veri tipi ve tüm örneklerin BLAKE2 özeti kaynakla karşılaştırılır. Tek bir bit farkı bile olsa dosya *kaydedilmez* ve hata verilir.
* Yazma işlemi önce `.part` dosyasına yapılır. Dosya ancak doğrulama geçerse asıl adını alır.
* **Başlıklar:** Her kanalın tüm dataset-58 alanları (ID1–ID5, nokta/yön, fonksiyon tipi, eksen etiketleri/birimleri, üs değerleri vb.) `UNV_...` özellikleri olarak saklanır. Ham 11 başlık satırı `UNV_header_raw` özelliğinde durur. Dataset 151 (model/proje bilgisi) ve 164 (birim sistemi) dosya düzeyinde saklanır.
* **nCode için:** `wf_increment`, `wf_start_offset`, `wf_samples`, `wf_start_time`, `unit_string`, `wf_xname`, `wf_xunit_string` ve `NI_ChannelName` yazılır. nCode örnekleme frekansını ve birimi bu alanlardan okur.

## Ayarlar
| Ayar | Açıklama |
|---|---|
| Δt yuvarlama düzeltmesi (varsayılan açık) | UNV formatı Δt'yi yalnızca 6 basamakla yazar (`1.92308E-05`). Bu yüzden 52 000 Hz değeri 51 999.92 Hz olarak okunur. Bu ayar açıkken, fark yazım hassasiyetinin altındaysa fs tam sayıya (52 000 Hz) yuvarlanır. Yalnızca zaman eksenli verilere uygulanır. Orijinal değer `UNV_abscissa_inc_original` özelliğinde saklanır. Örnek değerleri hiç değişmez. |
| Tüm veriyi float64 yaz | İsteğe bağlıdır. Dosya boyutu iki katına çıkar, kayıpsızlık açısından fark yaratmaz. |
| Kanal adı | ID1 (Testlab kanal adı), ID1 + [nokta:yön], nokta:yön veya sıra no + ID1. Aynı adlı kanallara otomatik `_2`, `_3` eklenir. `/` ve `\` karakterleri `_` ile değiştirilir. |
| TDMS grup adı | Boş bırakılırsa UNV dosya adı kullanılır. |

Kompleks veri (FRF vb.) `_Re` / `_Im` kanalları olarak, eşit aralıklı olmayan veri ise ek bir `_X` kanalıyla yazılır.

## Komut satırı (toplu iş)
```
UNV2TDMS.exe --cli D:\olcum\*.unv -o D:\tdms
```
veya `.venv\Scripts\python src\converter.py dosya.unv -o cikis` (seçenekler: `--no-verify --no-snap --float64 --overwrite --naming id1|id1_node|node_dir|index_id1 -g GRUP`).

## Dosyalar
* `src/unv_reader.py`: UNV 58/58b okuyucu (memory-map; GB'lık dosyalarda düşük RAM kullanır)
* `src/converter.py`: TDMS yazma, doğrulama, komut satırı
* `src/app.py`: arayüz
* `src/selftest.py`: otomatik test
* `EXE_OLMADAN_CALISTIR.bat`: kurulumdan sonra programı exe'siz çalıştırır
