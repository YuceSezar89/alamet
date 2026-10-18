# Oturum notu · 18 Ekim 2026

## Yapılanlar

**TradingView doğrulaması (20 sembol, son 4000 bar) — tüm gruplar 0 fark:**
sinyal yöntemleri ve MTF onayı, seçici modları ve gölge kasa, skor/sıralama/ayrışma kuralları,
boyutlandırma, çıkış/takas, sistem korumaları, giriş kalitesi, Devisso, yalnız long/short,
TP/stop/zaman stopu, 1 dk / 5 dk / 15 dk / 30 dk / 1 saat.

**Düzeltilen gerçek Python hataları (varsayılanda etkisizdi):**
1. 75 dk gibi günü bölmeyen HTF mum periyotlarının hizası (`heikin.bucket_id`, gün çapalı).
2. `drOn` kapalıyken Devisso kolaylık puanının sıralamaya eklenmesi (`stages.couple_dr_rank`).

**Öğrenilenler:**
- TradingView bazen aynı dosyayı farklı çalıştırıyor. Fark çıkarsa önce sayfayı yenile (Cmd+R),
  göstergeleri kaldır, yüklenme bitince tek gösterge ekle, tekrar çalıştır. Sonra Python'a bak.
- TradingView 30 dk ve üzeri grafikte geçmiş barlara kendi gününün TAM günlük paketini veriyor
  (15 dk ve altında dünün paketini). Doğrulama aracında `--day-map same`. Canlı botu etkilemez.

**Araçlar:** `alamet/tv_variant.py` (Pine girdi varsayılanlarını ve `--set` karşılığını birlikte üretir),
`verify_system --set / --day-map`, sinyal basamaklarında `--set`.

**Arayüz (http://127.0.0.1:8765, salt okunur):** sol menü; Hoca'nın 5 tablosu (İşlemler, Kasa,
Kayıp/Kazanç sebebi, Kasa motoru, Yöntem × Emtia); Tüm semboller; Grafikler; Ayrışma sekmesi
(piyasa görünümü, bacak grafiği, fiyat grafiği); Ayarlar (Hoca'nın başlık ve gruplarıyla, 166 ayar).

**Commit'ler (HİÇBİRİ push edilmedi):**
- alamet `7578aa1` (hata düzeltmeleri, `tv_variant`), `4ec80ec` (arayüz)
- eski depo (TRader-Panel-ASYNC) `337f378` (46 Pine varyant dosyası)

## Yeniden başlatma (bilgisayar açıldıktan sonra)

```bash
cd /Users/yusuf/Documents/TRader/CascadeProjects/alamet
export PYTHONPATH=$PWD
# demo bot (tüm semboller; anahtarlar .env'de, canlı için ALAMET_ALLOW_LIVE=EVET gerekir, SET ETME)
nohup python3 -u -m alamet.runner --universe data/universe.json --anchor 2023-07-01 \
  --db data/alamet.sqlite --trade demo --max-position 1000 --max-total 5000 \
  --daily-loss 250 --max-orders 200 > data/runner8.log 2>&1 &
# izleme arayüzü
nohup python3 -u -m alamet.webui --data-root data/binance_15m > data/webui.log 2>&1 &
```
Bot açılışta kapalı kalınan barları kendisi onarır ve uzlaştırma yapar. Durum `data/alamet.sqlite`'te kalıcıdır.
Kontrol: `tail data/runner8.log` ve arayüzdeki "Bot" kartı.

## Sıradaki işler

1. **Demo bot hâlâ HİÇ emir göndermedi** (emir, pozisyon, outbox = 0; sistem 686 sanal işlem biriktirdi).
   Nedenine bak: yeni sinyal gelmemesi mi, bir engel mi (örn. mod, sınır, filtre)?
2. **Borsa tarafı koruma planı, kullanıcı kararı bekliyor:**
   - Hoca'da marj = notional / `maxLev` (kasa belirler). Öneri: izole marjin + kaldıraç = `maxLev` (5).
   - Felaket stopu isteğe bağlı (Hoca varsayılanı kapalı): `stopR` × R, varsayılan 4R ≈ 6 ATR. İstenirse motorda
     `use_stop` da açılmalı (yoksa sistem ile borsa ayrışır, uzlaştırma botu durdurur).
   - Binance koşullu emirleri artık `POST /fapi/v1/algoOrder` (`algoType=CONDITIONAL`, `closePosition`,
     `triggerPrice`). Demo hesapta tek sembolde küçük test emriyle doğrulanmadan kod yazılmayacak.
   - Botta `set_leverage` / `set_margin_type` yazılı ama çalışan bot hiç çağırmıyor.
3. `vpTfOn` (TF onayı) kapalı hali TradingView testi (tek dosya, tek tur).
4. Alarm metni karşılaştırması (`DEBUG_S17_LOG.pine` Pine Logs örneği gerekiyor).
5. Hoca'nın sistemi + diğer dört sistemin birleştirilmesi (rol, zaman çerçevesi, kanıt, çıkış soruları).
6. Küçük: arayüzde skor kolonu (saklanmıyor, runner'a yazma gerekir), telefon düzeninin görsel doğrulaması,
   push (üç commit).
