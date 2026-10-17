# Devis'So Otomatik Bot Eğitim Paneli V2 (DEPV2) — Döküman

Kaynak: `Devis'So Otomatik Bot Eğitim Paneli.txt` (Pine v6, 2473 satır, yazar: engineerofmoney).
Tür: `indicator()` — strateji DEĞİL. Tüm "işlemler" scriptin içindeki sanal (gölge) kasada simüle edilir; TradingView hiçbir hesapta emir açmaz. Dışarıya tek çıkış `alert()` metinleridir.

---

## 1. ÖZET (tek sayfa)

**Ne yapar?** 20 perpetual sembolü (varsayılan MEXC) 15 dk "Sinyal TF"sinde tarar. Her sembol için 4 bağımsız yöntemden (Trend, Momentum, Volatilite, Hacim) yön sinyali üretir, MTF Heikin Ashi ile onaylar, sonra 20 sembolü "kendi ATR'sine göre yüzde değişim" skoruyla sıralar. Sıralamada ilk N'e giren, kendi geçmiş hareketlerinden ve marketten ayrışan semboller için LONG/SHORT sanal işlem açar. Pozisyon büyüklüğünü çok katmanlı bir risk zinciri belirler (Kelly motoru, Devisso ratio kolaylığı, dinamik kasa, hard kap). Her şeyi tablolarla gösterir ve AÇ/KAPAT alarmı üretir.

**Akış (6 aşama):**
1. Sembol başına 4 yöntem → ham yön (+1/−1/0)
2. Onay makinesi (`f_gate`): bekleyen sinyal → MTF HA onayı → tetik (direkt veya "2. sinyal")
3. Mod seçimi: tek yöntem / aynı anda conf / sıralı conf / otomatik (emtia başına en kârlı yöntem)
4. Skor + sıralama: `yön × yüzde_değişim / ATR%` + hedef mesafesi puanı + Devisso kolaylık puanı; havuz ve "kendi içinde ayrışma" kapısından geçenler sıralanır
5. Uygunluk: ilk N, aktif ayrışma, **çift ayrışma**, yön kilidi (ödül/ceza), giriş kalitesi filtreleri
6. Slot + boyutlandırma + koruma katmanları → sanal işlem → `alert("AÇ|...")` / `alert("KAPAT|...")`

**Varsayılan davranış (önemli):**
- Sinyal modu: **Otomatik** (emtia başına gölge kasada en iyi yöntem; hiçbiri pozitif değilse o emtiaya işlem açılmaz)
- Giriş fiyatı: sinyal barının **kapanışı**; slippage varsayılan 0, komisyon %0.02 + %0.02
- **TP kapalı, felaket STOP kapalı, zaman stopu kapalı.** Açık olan çıkışlar: kâr geri verme kilidi (1.5R zirve sonrası %50 geri verme), onaylı ters sinyal, sıralama değişimi (swap), ve sistem seviyesi kapatmalar (sert DD, günlük limit, kasa kâr kilidi)
- "1R" = 1.5 × ATR(10) — yalnızca boyutlandırma birimi, stop değil
- Her yönde 4 slot (LONG 4 + SHORT 4 = 8 eşzamanlı), toplam açık notional ≤ 3× kasa
- Baz risk 1R başına kasanın %0.5'i; çarpanlar büyütse de HARD KAP 1R kaybını kasanın %1'inde tutar
- Kasa 1000 USDT sanal

**Kritik teknik not:** 40 `request.security` kullanır (20 × sinyal TF + 20 × günlük). Bu TradingView sınırının tam üstünde; yeni istek eklenemez (satır 615'teki not bunu söylüyor).

---

## 2. GİRDİ GRUPLARI (satır 7–296)

| Grup | İçerik |
|---|---|
| Semboller (9–28) | 20 sembol, `MEXC:XXXUSDT.P`. Sembol 20 `BTWUSDT.P` (varlığı doğrulanmadı) |
| Sinyal Yöntemi | mod seçimi, yöntem aç/kapa (useM0–3), confN, seqOrder/seqN/seqWin, autoMinN/autoMinNet/autoBlock |
| Yöntem parametreleri | tdLen/tdThr, moLen/moZLen/moThr, vaSlow/vaFast/vaThr/vaCalm, hvLen/hvBand/hvThr/hvFlow |
| Mod | LONG+SHORT / sadece LONG / sadece SHORT |
| Secret Signal | ssTf="15", enforceTf, dOpenHr (günlük open UTC saati), atrPeriod=10 |
| MTF Heikin Ashi | çarpanlar 2/3/4/6 (→ 30/45/60/90 dk), ağırlıklar 2/3/4/6 |
| Sinyal | vpCool=10 (cooldown), vpMinPow=0 |
| TF Onayı | vpTfOn, vpTfNeed=3/4, vpConfBars=1, vpPendMax=12 |
| Ayrışma | gapMin, nTop=2, rankMode (ATR cinsinden), maxDiv=10 |
| Hedef Mesafesi | tgtW=0.25, tgtCap=8 ATR, tgtMode |
| Çift Ayrışma | zMin=1σ, minPrev=2, dblPosOnly |
| Kendi İçinde Ayrışma | useOwnFirst, ownMode, ownMult, ownN=8, ownLatch, ownAllowNoHist |
| Ödül/Ceza | minScore, minShare=%50 |
| Slot | needDbl=true, reEntry=true, maxLongSlots=4, swapEdge=0.3, stayResetBars=3 |
| Giriş Kalitesi | useExt/maxExt=4 ATR, rank2Mult=0.5, coolBars=12, useDOp, useWOp, useMOp, maxDOpDist |
| Kâr Kilidi | useGive, giveArm=1.5R, giveFrac=0.5 |
| Koruma | useStop(false)/stopR=4, useTimeStop(false)/maxHold=96 |
| İşlem Motoru | rUnit=1.5, useTp(false)/tpR=2, logN=8 |
| Kasa | startCap=1000, sizeMode, riskPct=0.5, maxLev=5, fixMargin, leverage, feeIn/feeOut=0.02, slipPct=0 |
| Hard Kap (NS22) | capOn, capPct=1, capMode |
| Devisso Ratio (NS22) | drOn, drRsiLen=3, drLook=60, drMinM=0.4, drMaxM=2.0, drPow, drBlockOn(false), drSrc="Karma", drRankOn/W, drBlockPct=90 |
| Kayıp Analizi | holdShort=3, holdLong=12 |
| Dinamik Kasa | dynMinN, dynDecay, dynGain, min/maxMult, useKill/killN/killR, ddThrottle/ddMult/ddHalt, streak, maxExp=3 |
| DD Koruması (NS13) | ddHard %15, günlük kayıp %3 |
| Kasa Koruması (NS19) | lockArm %3, lockFrac 0.4, negMult 0.6 |
| Kasa Motoru (NS21) | engKelly=0.25, engPrior=4, engZ=0.5, engMinN=6, engExplore=0.5, engMin/Max, engBlock=0.15, engClip=3, engDecay |
| Havuz | poolSize=8, poolMinN=4, poolDecay vb. |
| Tablolar | 5 tablo, konum/boyut seçenekleri |

Girdi doğrulama (301–320): grafik TF'si sinyal TF'sinden büyükse hata; minPrev/maxDiv uyumu; en az bir yöntem aktif; drMaxM ≥ drMinM; seqOrder 1–4 arası sayılar.

---

## 3. ZAMANLAMA VE REPAINT KORUMASI (322–345)

- `ssNew = timeframe.change(ssTf)`: yeni 15 dk sinyal barı. `ssIdx` bunları sayar; tüm "bar sayısı" parametreleri (cooldown, bekleme, sönümleme) SS bar cinsindendir.
- `ssLag`: grafik TF'si < sinyal TF'si ise 1, değilse 0. `f_lg(x, lag)` ile bir önceki **kapanmış** SS barı kullanılır.
- Tüm işlem mantığı `barstate.isconfirmed and ssNew` içinde çalışır (`f_shadow`, `f_blockA`, `f_blockB`, `f_peak`, `f_ownUpdate`). Seçim güncellemesi (`f_selUpdate`) her onaylı grafik barında çalışır.
- `request.security(..., lookahead_on)` kullanılır; repaint'i önleyen unsur ssLag + yalnızca onaylı barda hesap.
- `calc_bars_count = 4000`: geçmiş hesabı en fazla 4000 bar. **Tüm öğrenen yapılar (Kelly kovaları, havuz, gölge kasa, sanal kasa) her yüklemede bu pencereden sıfırdan yeniden oynatılır; kalıcı hafıza yoktur.**
- `dayChg` / `dayPendA`: UTC günü (veya dOpenHr) değişince günlük kayıp limiti için gün başı kasa kaydı tetiklenir.

---

## 4. ÜRETİM KATMANI: `f_sym` (483–586), sembol başına çalışır

### 4.1 MTF Heikin Ashi gücü (`f_haScore`, 377–401)
Sinyal TF barlarını 2/3/4/6 katı TF'ye birleştirir (kendi OHLC'sini biriktirir), Heikin Ashi hesaplar, skor = `(haC − haO)/(haH − haL) × 100` (−100…+100). `finalPct` = ağırlıklı ortalama (2,3,4,6) → "MTF güç".
**TF onayı:** `nBullTf` = skoru >0 olan HA sayısı. `tfOkL = nBullTf ≥ vpTfNeed` (3/4); SHORT için `<0`.

### 4.2 Dört dik yöntem (her biri farklı girdiden)
| Yöntem | Girdi | Hesap | Durum makinesi | Güç kodu |
|---|---|---|---|---|
| ① TREND | fiyat seviyesi | `z = (close − SMA(hlc3,24)) / ATR` | z≥0.5 → +1; z≤−0.5 → −1; zıt tarafa geçince 0 | \|z\| ≥3·thr→2, ≥2·thr→1 |
| ② MOMENTUM | zamanın türevi | `z = ln(c/c[12]) / (σ_1bar(48) × √12)` | ±1.0, zıt işarette sıfırla | \|z\| ≥3·thr→2, ≥2·thr→1 |
| ③ VOLATİLİTE | bar aralığı | `X = TR / ATR_slow[1]` (12); yön = genişleme barının gövdesi | X≥1.3 ve yeşil → +1, kırmızı → −1; hızlı/yavaş ATR<0.8 → 0 | X≥1.5·thr→2, ≥1.2·thr→1 |
| ④ HACİM | katılım | `relV = hacim/SMA`; `hvShare` = son 12 barda fiyat bölgesinde (±0.5 ATR) işlem gören hacim payı; `ratio = relV / max(hvShare, 0.1)`; yön = CLV delta akışı | ratio≥2 ve \|flow\|≥0.1 | ratio≥1.5·thr→2, ≥1.2·thr→1 |

### 4.3 Onay makinesi `f_gate` (411–470) — her yöntem için ayrı örnek
- Ham sinyal oluşunca (cooldown 10 bar geçmişse) **bekleyen** (`pendDir`) olur.
- Ham durum bozulursa veya `vpPendMax` (12 bar) dolarsa bekleyen iptal.
- Bekleyen yönde TF onayı `vpConfBars` kez üst üste gelirse **TETİK**: `confDir` güncellenir (son ONAYLI yön).
- Onay aynı barda gelirse "direkt", sonraki barlarda gelirse güce +3 eklenir → **② = 2. sinyal**.
- Çıktı: `[yön, başlangıç fiyatı (LONG→low, SHORT→high), güç kodu, bekleyen yön, son onay bar'ı]`.

### 4.4 Günlük open
Son günlük open (dOpenHr=0 ise UTC 00:00) saklanır; giriş filtresinde kullanılır.

### 4.5 Çıktı dizisi (27 değer)
`[0]close [1]high [2]low [3]atr [4]günlük open [5]open [6]MTF güç` + 4 yöntem × 5 değer (`7 + m·5 + o`: yön, başlangıç, güç, bekleyen, onay bar'ı).

---

## 5. GÜNLÜK PAKET: `f_dayPack` (639–665) — 20 değer / sembol

- `f_per`: hafta (W), ay (M), yıl (12M) için **bu periyodun O/H/L'si** + **önceki periyodun H/L/C'si** (18 değer; kolon 0–5 hafta, 6–11 ay, 12–17 yıl).
- **Devisso Ratio** (kolon 18–19):
  - `drTot` = kümülatif % değişim toplamı. Yükselişte `+Δ/close[1]×100`, düşüşte `−(close[1]−close)/close×100`. **Not:** düşüş dalında payda `close[1]` değil `close` (yükseliş dalıyla asimetrik).
  - `drRsi = RSI(drTot, 3)`; `drRaw = Δrsi / Δtot`; `drK = |drRaw|` (Δtot≠0 iken)
  - `drB = drK[1]` (bir önceki KAPANMIŞ gün — repaint yok), `drPr` = drB'nin önceki 60 günün ratio'ları içindeki yüzdeliği (0 = en küçük = en kolay; min 10 geçerli gün)
- Emtialar arası yüzdelik (`f_drXsAll`): bugünkü ratio'nun 20 emtia içindeki sırası (≥3 emtia gerekir).
- `f_drPr(k)`: kaynağa göre kendi / emtialar arası / karma (ortalama; biri yoksa diğeri).

---

## 6. SEÇİM KATMANI (791–947)

`mSig` (20×27 matris) sütunlara ayrılır: `cls, highs, lows, atrs, dops, opens, pows`.

**`f_selUpdate`** (yalnız onaylı grafik barında):
- Tek yöntem modları (0–3): o yöntemin yönü.
- **Otomatik (6):** her emtia için, `shCnt ≥ autoMinN (3)` olan aktif yöntemler arasında sönümlü net K/Z (`shNetD`) `> autoMinNet` olan **en yüksek**. Yoksa: veri var ama hiçbiri pozitif değilse ve `autoBlock` → o emtiada yön 0 (işlem yok); veri hiç yoksa ilk aktif yönteme düş.
- **Aynı anda conf (4):** en az `min(confN, aktif)` yöntem aynı yönde ve karşı yönden fazla; temsilci = en yeni onaylı yöntem.
- **Sıralı conf (5):** `seqOrder`'daki ilk `seqN` aktif yöntem aynı yönde, onay barları sıra bozulmadan artmalı, ilk–son onay arası ≤ `seqWin`.
- Sonuç `cbDir/cbM/cbSt/cbStr` (sticky): yön, hangi yöntem, başlangıç fiyatı, güç.

**`f_selOut`:** `types` (yön), `sts` (başlangıç), `strs` (güç), `pnds` (bekleyen), `pcts` = `(close − başlangıç)/başlangıç × 100` — sinyal başlangıcından beri % hareket (grafikte çizilen 20 çizgi budur).

---

## 7. GÖLGE KASA (`f_shadow`, 1437–1473)

Her emtia × 5 slot (4 yöntem + seçili mod = 100 kayıt) kendi sinyal yönünü sanal işler:
- Yön değişince önceki pozisyon kapanış fiyatından kapanır, yeni yön açılır.
- Boyut: **başlangıç kasası üzerinden sabit risk** (bileşik değil) → yöntemler birebir kıyaslanabilir.
- Net K/Z (`shNet`) ve her `dynDecayBars`'ta 0.97 ile sönümlenen `shNetD` tutulur. Otomatik mod bu sönümlü değeri kullanır.
- Yöntem × Emtia tablosunun veri kaynağı.

---

## 8. SKOR, HAVUZ, SIRALAMA

### 8.1 Skor (`f_calcSc`, `f_tgtAll`)
- `raw = yön × pcts`; ATR modunda `skor = raw / ATR%` (volatilite normalize).
- **Hedef puanı:** yönündeki en yakın HTF OHLC seviyesine mesafe (ATR cinsinden; yakın = hepsi, büyük = ay+yıl, varsayılan ikisinin ortalaması), `tgtW × min(mesafe, 8)` skora eklenir. Yalnız kendi-içi kapıyı geçenlere.
- **Devisso kolaylık puanı:** `drRankW × (1 − yüzdelik/100)` — en kolay emtia +1, en zor 0.

### 8.2 Kazanan havuzu (`f_poolCalc`)
`poolNet` (emtia başına gerçek işlem net K/Z, her 48 SS barda ×0.85 sönüm). Yargılanmak için ≥4 işlem; ilk `poolSize=8` ve net > 0 geçer. Denenmemiş/affedilen emtialar (`poolProb`, `poolForgive` %0.2 kasa) deneme olarak girer.

### 8.3 Kendi içinde ayrışma — ÖNCE (`f_ownUpdate/f_ownGate`, 1054–1125)
Her emtia için "bacak" (aynı yönlü sinyal dizisi) başına **zirve hareket** (`legMax`, yön lehine %) halka tamponuna yazılır (8 bacak). Geçerli hareket, bir önceki bacağın zirvesinin `ownMult` katını (veya ort+σ) aşmalı. Aşılmışsa bacak boyunca korunur (`ownLatch`). Geçmişi olmayan emtia varsayılanda geçemez.

### 8.4 1. sıra koruması (`f_top1`)
Havuz dışında olsa bile her yönün en yüksek skorlu emtiası, net K/Z `top1Floor` (−%2 kasa) üstündeyse (veya yargılanmamışsa) havuza geri alınır.

### 8.5 Sıralama (`f_rankAll`)
Havuz + kendi-içi kapıdan geçenler, LONG ve SHORT için ayrı sıralanır. İlk `nTop=2` (ve (N+1). ile fark ≥ gapMin) `topArr`'a girer = **"market hedefi"**.

---

## 9. AYRIŞMA TAKİBİ (Blok A, 1786–1854; `f_upd`, `f_own`, `f_lock`)

- Bir emtia `topArr`'a girince **yeni ayrışma n** başlar: giriş fiyatı kaydedilir. Her barda hareket `mv = (close/giriş − 1)×100` güncellenir, `resMv` halka tamponuna (son 10 ayrışma) yazılır. `topArr`'dan çıkınca veya sinyal yönü zıt döndüğünde ayrışma biter.
- **Çift ayrışma (★):** o anki hareket, önceki ayrışmaların ortalaması + zMin·σ'yı aşmalı (en az minPrev geçmiş). Gerçekleşince `resDb` işaretlenir ve alarm gönderilir.
- **Yön kilidi `f_lock` (ödül/ceza):** son maxDiv ayrışmanın yukarı vs aşağı hareket toplamlarından kazanan taraf (pay ≥ %50, > minScore) hangi yöndeyse işlem yalnız o yönde uygundur; belirsizse (2) hiçbir yönde değil.
- İşlem uygunluğu (`f_elig`): yön ≠ 0, yön izinli, `topArr`, ayrışma aktif, `cntA ≥ tradeMinN`, **çift ayrışma (needDbl)**, `f_lock == yön`. Uygunlar sıralamadaki yerine göre **hedef #1…#N** (`tRank`).

---

## 10. İŞLEM MOTORU

### 10.1 Giriş filtreleri (`f_qual`, 1588–1614)
Boş string = geçti. Engeller: MTF güç yetersiz (vpMinPow), **uzama** (başlangıçtan >4 ATR veya ≤0), **soğuma** (zararla kapanan emtiada 12 SS barı), **günlük open ters taraf** (LONG için fiyat ≤ günlük open), günlük open'dan çok uzak (varsayılan kapalı), haftalık/aylık open (varsayılan kapalı).

### 10.2 Boyutlandırma zinciri (Blok B, 1962–1981)
```
fMult = fMult0 × engM × drM
  fMult0 = (motor açıksa 1, değilse dinamik skor×DD) × seri kısıtı(×0.5) × [kasa<başlangıç ise ×0.6]
  engM   = Kelly motoru çarpanı (0.25–3.0)
  drM    = Devisso kolaylık çarpanı (0.4–2.0)
notional = min( kasa × riskPct × fMult / (1R mesafesi),  kasa × maxLev )
         → min( ... , HARD KAP )          # risk modu: kasa × %1 / mesafe
         → × (hedef #1 ise 1.0, #2+ ise 0.5)
         → min( ... , kalan ısı: kasa×3 − açık notional )
```
Blok (işlem açılmaz): dinamik DD durdurma/kötü kova (BLOK), motor E[R] ≤ −0.15, DR blok (varsayılan kapalı), herhangi bir durdurma aktif, ısı limiti dolu.

### 10.3 Slot ve swap
Her yönde 4 slot. Slot doluysa, hedef listesinde olmayan açık işlemlerden skoru en düşük olan, yeni adayın skoru ondan en az `swapEdge` (0.3) iyiyse kapatılır (`SIRALAMA`) ve yenisi girer.
`reEntry=true`: hedefte kaldığı sürece kapanıp yeniden girilebilir; `stayUsed/stayResetBars` ile kontrol.

### 10.4 Çıkış öncelik sırası (`f_blockA`)
1. Felaket stop (açıksa; aynı barda TP+STOP → STOP; gap'te açılıştan)
2. TP (açıksa)
3. Kâr geri verme: zirve R ≥ 1.5 iken R ≤ zirve × 0.5
4. Onaylı **ters sinyal**
5. Zaman stopu (açıksa)
Ek: SIRALAMA swap'ı (Blok B), sistem kapatmaları (aşağıda).

PnL net: `d·(px/ent − 1)·100 − feeIn − feeOut·(px/ent)` (slippage girdileri komisyona eklenir). Fonlama oranı modellenmemiştir.

### 10.5 Sistem koruma katmanları
| Katman | Tetik | Etki |
|---|---|---|
| Dinamik kasa DD durdurma | zirveden ≥%10 | 48 SS barı yeni işlem yok |
| Sert DD (NS13) | zirveden ≥%15 | TÜM açıkları kapat + 96 bar bekle + alarm |
| Günlük kayıp | gün başı kasadan ≥%3 | gün sonuna yeni işlem yok (dayLossClose açıksa açıkları da kapat) |
| Kasa kâr kilidi (NS19) | zirve kâr ≥ başlangıcın %3'ü ve zirve kârın %40'ı geri verilince | TÜM açıkları kapat + 48 bar bekle |
| Negatif kasa | kasa < başlangıç | risk ×0.6 |
| Kayıp serisi | ≥3 ardışık kayıp | risk ×0.5 |
| Isı limiti | toplam açık notional ≤ kasa×3 | boyut kırpılır / işlem açılmaz |

---

## 11. ÖĞRENEN KATMANLAR

### 11.1 Dinamik Kasa (f_dyn, 1520–1536)
12 klasik kovadan (yön, hedef sırası, ayrışma grubu, tutma, günlük open) giriş için 3 kova skorlanır: `sc = Σ sr/(ne+dynMinN)/3`; çarpan `clamp(1 + 1.5·sc, 0.25, 2.0)`. Kova (yön hariç) `ne ≥ 6` ve ort R ≤ −0.5 ise BLOK. Kovalar 96 barda ×0.97 sönümlenir. Motor açık + `engReplace` iken çarpan motora devredilir (çift sayım önlenir).

### 11.2 Kasa Motoru (NS21, f_engine, 1638–1691)
60 kova (sebep tablosuyla aynı 56 + 4 Devisso dilimi). Bir işlem adayı için **9 özellik kovası**: yön, yöntem, sinyal gücü, hedef sırası, ayrışma no grubu, giriş tipi (direkt/2.), günlük open yönü, DR kolaylık dilimi, emtia.
- Her kapanan işlemin net R'si (±3'e kırpılmış) tüm 9 kovaya yazılır (n, Σr, Σr²).
- `E[R]` = kovaların shrinkage ortalamalarının güven (n/(n+4)) ağırlıklı birleşimi; `nEff` = emtia kovası hariç kovaların ortalama n'i.
- `σ` = yön kovalarındaki R dağılımı std (min 0.5).
- `nEff ≥ 6` ise: `muS = E[R] − 0.5·σ/√nEff`; **Kelly çarpanı = 0.25 × (muS/σ²) / (riskPct/100)**, `[0.25, 3.0]`; `E[R] ≤ −0.15` → BLOK. Değilse "keşif" çarpanı 0.5.
- Drawdown'da çarpan `×max(0.25, 1 − dd/%15)`.
- Kovalar 96 barda ×0.98 sönümlenir.

### 11.3 Devisso kolaylık dilimi
`pr ≤25` çok kolay (56), `≤50` kolay (57), `≤75` orta (58), `>75` zor (59), veri yoksa 58. Hem çarpanı (`drM = 0.4 + 1.6·ease^drPow`) hem motor kovasını besler; yani "kolay işlemler gerçekten kazandırıyor mu" sorusunu script kendi kendine öğrenmeye çalışır.

---

## 12. ÇIKTILAR

### 12.1 Tablolar (5)
1. **İşlemler** (son 8 işlem, durum, R, PNL, K/Z, toplamlar)
2. **Kasa** (kasa, net, açık K/Z, zirve, DD, W/L, PF, L/S K/Z, en iyi/kötü, kayıp serisi)
3. **Kayıp/Kazanç sebep analizi** (56 kova, 12 bölüm: çıkış sebebi, yön, yöntem, güç, hedef sırası, ayrışma no, tutma, işlem seyri, giriş tipi, günlük open, komisyon kaynaklı kayıp, emtia; 🔥 en büyük kayıp / 🏆 en büyük kazanç)
4. **Kasa motoru** (kova n, ham R, E[R], Kelly x)
5. **Yöntem × Emtia** (gölge net $, işlem, win %, 🏅 en iyi yöntem, gerçek motor sonucu)
+ 20 çizginin ucunda etiket (sıra, ●top, ★çift ayrışma, 🎯T hedef sırası, yöntem baş harfi, güç, ② ikinci sinyal, ⏳ bekleyen, DR yüzdelik).

### 12.2 Alarmlar (hepsi `alert()`, `freq_once_per_bar_close`, serbest metin — JSON değil)
| Olay | Biçim |
|---|---|
| Çift ayrışma | `SYM · n. AYRIŞMA ★ ÇİFT (kendi içinde de) +x%` |
| n≥2 ayrışma | `SYM · n. AYRIŞMA ▲L/▼S` |
| **AÇ** | `AÇ \| SYM \| LONG/SHORT (mod · yöntem) güç \| Hedef #r \| Giriş px \| TP px \| Büyüklük USDT \| 1R risk % \| Dinamik x(...)` |
| **KAPAT** | `KAPAT \| SYM \| LONG/SHORT \| TP/KİLİT/SIRALAMA/STOP-ZAMAN-DD/TERS SİNYAL \| R \| PNL \| K/Z USDT` |
| Sert DD | `⛔ SERT DD ACİL ÇIKIŞ \| ...` |
| Günlük limit | `⛔ GÜNLÜK KAYIP LİMİTİ \| ...` |
| Kasa kâr kilidi | `🔒 KASA KÂR KİLİDİ \| ...` |

Sembol adı `f_nm` ile `MEXC:XXXUSDT.P` → `XXXUSDT` biçimine indirgenir.

---

## 13. GERÇEK HESAPLA KULLANIM İÇİN NOTLAR (gözlem, eleştiri değil)

1. **Sanal durum ≠ gerçek hesap.** Slot, kasa, DD, Kelly kovaları scriptin kendi sanal kasasından beslenir. Gerçek hesapta bir alarm kaçarsa, emir reddedilirse veya bakiye farklıysa script ile hesap ayrışır; script bunu bilmez.
2. **Kalıcı hafıza yok.** `calc_bars_count=4000` penceresinden her yüklemede yeniden oynatılır; sayfa yenilemesi/alarm yeniden oluşturma durumu değiştirebilir (özellikle öğrenen kovalar ve havuz).
3. **Varsayılan stop yok.** Gerçek hesapta koruyucu emir (borsa tarafı stop) ayrıca düşünülmeli; scriptin kendisi TP/STOP seviyesi vermez (TP/stop kapalıyken alarmda `TP -`).
4. **Giriş kapanış fiyatından**, alarm bar kapanışında gelir; gerçek dolum fiyatı farklı olur (slippage 0 varsayımı).
5. **Alarm metni serbest biçim**; webhook köprüsü için ayrıştırma ya da JSON'a çevirme gerekir.
6. **Sembol evreni elle seçilmiş 20 sembol**; MEXC → Binance eşlemesi ve sembol varlığı (özellikle LIT, BTW) doğrulanmalı.
7. **40 request.security sınırı dolu**; sembol sayısını artıramaz, yöntem eklenemez.
8. **Devisso ratio formülü**: düşüş dalındaki payda asimetrisi (`/close` vs `/close[1]`) panelimizdeki porttaki karşılığıyla karşılaştırılmalı.
