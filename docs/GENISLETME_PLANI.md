# DEPV2 — Genişletme Noktaları Planı

Amaç: çekirdeğe **yeni sinyal yöntemleri**, **yeni onay/doğrulama mekanizmaları**, yeni giriş filtreleri ve
çarpanlar, ileride orderbook gibi **yeni veri** eklemeyi, doğrulanmış davranışı bozmadan mümkün kılmak.

Güvence: `tests/data/golden` (S0–S49 donmuş çıktı, TV'ye çapalı) + 376 birim test. Her aşamadan sonra
`ALAMET_GOLDEN=all pytest tests/test_golden.py` aynı kalmalı. Varsayılan kapalı bir bileşen
hiçbir sonucu değiştirmemelidir.

## 1. Mevcut dikişler (kodda bugün nasıl)

| Genişletme | Bugün nerede | Sınırlama |
|---|---|---|
| Yeni sinyal yöntemi | `symbol.py` `raw = (trend, momentum, volat, vol)` | Yöntem sayısı 4 sabit |
| Yöntem sayısına bağlı kod | `selector.py` 4× `range(4)`, `shadow.py` `[:, :4]`, 5 sütun (4 + seçili), `kelly.py` kova 11–14 ve 60 sabit, `SelectorParams.enabled` 4'lü, `parse_seq_order` 1–4 | Hepsi 4'e bağlı |
| Yeni onay | `symbol.py` `mtf_confirm` → `run_gate(tf_long, tf_short)` | Tek onay maskesi; birden çoğu için kod gerekir |
| Yeni giriş filtresi | `entry_quality.py` bit maskesi (1…64) + `LABELS` | Güzel bir dikiş; yeni bit eklemek kolay |
| Yeni boyut çarpanı | `engine.try_entry`: `mult = entry_multiplier * kelly * dr` satır içi | Sıra sabit kodlu |
| Yeni skor terimi | `score.combine_score(base, tgt, dr)` | İki sabit bonus |
| Yeni sıralama maskesi | `pool.py` → `rank_all(pool_ok)` | Tek maske |
| Yeni çıkış | `engine.manage_open` (öncelik: stop, tp, kâr geri verme, ters sinyal, süre) | Öncelik sırası kodda |
| Yeni veri | `Universe` yalnızca OHLCV | Ek özellik dizisi yok |
| Yön kilidi / uygunluk | `divergence.py` `lock`, `_eligible` | Kullanıcı bu konuyu ayrıca inceleyecek |

## 2. Tasarım ilkeleri
1. **Davranış korunur:** her adım önce yeniden düzenleme (altın referans değişmez), sonra özellik.
2. **Basit kayıt listesi, eklenti keşfi yok:** bileşenler kodda açık listeyle bağlanır (`SIGNAL_METHODS`
   gibi); dinamik yükleme yok (YAGNI).
3. **Pine uyumu korunur:** varsayılan 4 yöntem ve mevcut kova/mod numaraları aynı kalır; TV debug
   kopyaları çalışmaya devam eder. Yeni bileşenlerin Pine karşılığı yoktur.
4. **Yalnızca ilk gerçek ihtiyaçta genelleştir:** puan terimi, çıkış, kilit gibi alanlar ilk gerçek
   kullanımda ele alınır, şimdi değil.

## 3. Hedef arayüzler

**A. `SignalMethod` (yeni sinyal)** — `name`, parametre `dataclass`'ı, `compute(open, high, low, close, volume, features) -> (state ∈ {−1,0,1}, strength ∈ 0..2)` (dizi). `symbol_signals` kayıtlı
yöntemleri sırayla çalıştırır; sonuç `(bar, sembol, M)` olur. Varsayılan kayıt = bugünkü 4 yöntem.

**B. Yöntem sayısı M'yi genelleştirme:** `selector` (`range(M)`), `shadow` (M + 1 sütun), `SelectorParams.enabled`
(M'lik), `parse_seq_order` (1..M), tek-yöntem modları (`SINGLE(m)`), `kelly` kova yerleşimi `BucketLayout(M)`
ile türetilir; **M = 4 için indeksler bugünküyle birebir aynıdır** (altın referans bunu kanıtlar). Mod
numaralarını (4 conf, 5 sıralı, 6 otomatik) adlandırılmış sabitlere çevirmek gerekir.

**C. `Confirmation` (yeni onay)** — `masks(bars...) -> (ok_long, ok_short)`; hepsi VE ile birleşir ve
`run_gate`'in `tf_long/tf_short` girdisine verilir. MTF Heikin-Ashi ilk onaydır. Ayrıca *seçici düzeyi*
onay (çoklu yöntem uyumu, `confN`) ayrı bir kavramdır ve olduğu gibi kalır.

**D. `EntryFilter`** — `bit`, `label`, `flags(ctx) -> bool dizisi`; mevcut 7 filtre bu arayüze taşınır,
yeni filtre yeni bit alır (64'ten sonra). `describe()` etiketleri otomatik tamamlar.

**E. `SizingFactor` zinciri** — `multiplier(ctx) -> (çarpan, blok)`; dinamik kasa, Kelly, Devisso aynı
sırayla listeye taşınır; yeni çarpan listeye eklenir.

**F. `Universe.features`** — `dict[str, ndarray(bar, sembol)]` (isteğe bağlı, varsayılan boş): orderbook
dengesi, funding, açık pozisyon vb. `SignalMethod`, `Confirmation`, `EntryFilter` bağlamında okunur.
Orderbook hazırlığı (hafızadaki üç kural) bunun parçasıdır; piyasa verisi arayüzü ve kayma arayüzü
emir katmanında (E1) kurulur.

## 4. Yeni bileşenlerin doğrulaması (TV karşılığı yok)
1. **Nedensellik testi (bu planın en önemli parçası):** bileşeni `[0, T]` verisiyle ve `[0, T+k]` verisiyle
   çalıştır; `≤ T` çıktıları **birebir aynı** olmalı. Geleceğe bakmayı (look-ahead) otomatik yakalar.
   Mevcut panel için de yazılır.
2. **Varsayılan kapalı = sıfır etki:** altın referans değişmez.
3. **Birim testler** (sentetik veri, bilinen sonuç) ve **belirlenimcilik** testi.
4. **Etki ölçümü:** ablation (bileşen açık/kapalı) + kontrol (zaman kaydırılmış/ters girdi) ile; kâr yorumu
   çok başlangıçlı ve placebo ile birlikte yapılır, tek koşuyla değil.
5. Doğrulanan davranışı altın referansa yeni basamak olarak ekle (`golden update`).

## 5. Aşamalar (her biri ayrı onay ve ayrı commit)

| Aşama | İş | Risk | Güvence |
|---|---|---|---|
| G0 | Nedensellik testi aracı + mevcut panele uygula | düşük | yeni test |
| G1 | `M` genelleştirme (selector, shadow, kelly yerleşimi, mod adları), M=4 | **yüksek** (en hassas kod) | altın referans 50 basamak |
| G2 | `SignalMethod` kaydı; mevcut 4 yöntemi kayda taşı | orta | altın referans |
| G3 | `Confirmation` listesi (tek elemanlı: MTF) | düşük | altın referans |
| G4 | `EntryFilter` kaydı (mevcut 7 filtreyi taşı) | düşük | altın referans |
| G5 | `SizingFactor` zinciri | orta | altın referans |
| G6 | `Universe.features` (boş varsayılan) + bağlam | düşük | altın referans |
| G7 | Örnek 5. yöntem (kanıtlayıcı) ve onay ile uçtan uca nedensellik + ablation | orta | G0 testi |

**Önerilen sıra:** G0 → G1 → G2 → G7 (ilk gerçek yeni sinyal gelene kadar G3–G6 beklemeye alınabilir;
ihtiyaç doğunca yapılır).

## 6. Bilerek yapılmayanlar
Eklenti keşfi/dinamik yükleme, çıkış ve kilit politikalarının genelleştirilmesi (ilk ihtiyaçta),
panel/arayüz, orderbook kodu (en son iş).

## 7. Açık sorular (senin kararın)
1. İlk eklenecek sinyal ya da onay hangisi? (G7'nin içeriği ve G1'in gerçekten gerekli olup olmadığı buna bağlı; yeni onay ise G1 gerekmeyebilir.)
2. Yön kilidi ve sinyal onayı incelemesi (senin fikirlerin) G1–G3 ile birlikte mi, ayrı mı yapılsın?
3. Yeni bileşenlerin parametreleri `stages.py` benzeri basamaklarda mı tanımlansın?
