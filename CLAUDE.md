# Alamet — Claude Çalışma Kuralları

## Dil ve tarz
- Her zaman Türkçe konuş. Cevaplar orta uzunlukta: kısa açıklama + sonuç.
- Büyük işlerde önce planı göster, onay gelince yaz. Yalnızca istenen şeyi yap.
- Yorum satırı ekleme (gerekmedikçe). Refaktör istenmedikçe çevreyi temizleme.
- Önce ilgili kodu/veriyi oku, sonra öner. Değişimin etkilediği tüm modülleri tara.

## Kod kalitesi
- Her değişiklikten sonra: `black`, `isort`, `pylint`, `autoflake`; duplicate kod varsa belirt.
- Çekirdeğe eski TRader Panel modüllerini (config, database, utils, signals…) ASLA içe aktarma.

## Proje ilkeleri
- Hedef önce birebir port, sonra genişletme. Kâr/zarar yorumu yapma; "%94" gibi kod-kapsamı
  yüzdelerini birebirlik ilerlemesi diye sunma.
- Birebirlik kanıtı: TradingView "Export chart data" CSV'si (grafik en başa kadar kaydırılmış olmalı).
- Altın referans (`ALAMET_GOLDEN=all pytest tests/test_golden.py`) davranış değişikliğinde değişmemeli.
- Orderbook entegrasyonu EN SON iş; yalnızca mimariyi hazır tut.
- Yön kilidi + sinyal onayı incelemesi kullanıcının fikirleriyle, port bittikten sonra yapılır.

## Güvenlik
- Binance API anahtarları yalnızca `.env` (git dışı). Koda/loga/sohbete yazma. Varsayılan testnet/demo.
- Commit/push yalnızca kullanıcı isterse. Silme/üzerine yazma öncesi hedefe bak.
- datetime için `datetime.now()` kullan; timezone/ZoneInfo ile karışık kullanma.
