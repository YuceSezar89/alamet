# Alamet

Devis'So DEPV2 sinyal ve kasa sisteminin (TradingView Pine) birebir Python portu ve bunun üzerine
kurulan Binance Futures (USDT-M) botu. Eski TRader Panel projesinden bağımsızdır.

> **Not:** Bu bot, İsmail Selçuk Deveci'nin Pine kodundan Yusuf Türk tarafından Python'a çevrilmiştir.
> Kod, Pine kodunun sahibinin izniyle paylaşılmıştır.

## Lisans ve uyarı
MIT lisansı ([LICENSE](LICENSE)): kopyalayabilir, değiştirebilir ve kullanabilirsiniz. Bu bir yatırım
tavsiyesi değildir; sistem kâr garantisi vermez, gerçek parayla kullanım tamamen sizin
sorumluluğunuzdadır.

## Durum
- Pine mantığı Python'a taşındı ve TradingView ile satır satır doğrulandı (basamak S0–S50).
- İstatistikler, olaylar (7 alert), 5 görünüm tablosu, ayarlanabilir sinyal TF'si (15/5/1 dk).
- Evren seçimi, toplu geçmiş indirme, canlı mum akışı. Henüz yok: emir katmanı, durum kaydı.

## Hızlı başlangıç
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest                                   # ALAMET_GOLDEN=all pytest tests/test_golden.py → 51 basamak
python -m alamet.market --mode top_n --n 20 --out data/universe.json
python -m alamet.bulk_data --universe data/universe.json --start 2023-07-01
python -m alamet.livefeed --universe data/universe.json      # gölge çalışma
```

## Yapı
| Yer | İçerik |
|---|---|
| `alamet/` | Paket: sinyal, sıralama, ayrışma, motor, Kelly, havuz, istatistik, olaylar, görünümler, veri |
| `tests/` | Birim testler + altın referans (`tests/data/golden`, TV'ye çapalı) |
| `pine/` | Pine kopyaları (`debug/` TV doğrulama kopyaları, `original/` hocanın kodu) |
| `docs/` | `BAGIMLILIK.md` (bağımlılık ve doğrulama günlüğü), `GENISLETME_PLANI.md` |
| `data/` | Geçmiş mum önbelleği, evren dosyası (git dışı) |

## Kurallar
- Çekirdek yalnızca numpy, pandas, pyarrow, aiohttp, websockets kullanır; eski sistemin hiçbir modülünü
  içe aktaramaz (`tests/test_independence.py`).
- Birebirlik TradingView CSV'siyle kanıtlanır (`python -m alamet.verify_system`, `verify_stats`).
- API anahtarları yalnızca `.env` içinde durur; koda, loga ve sohbete yazılmaz. Varsayılan ortam testnet/demo.
