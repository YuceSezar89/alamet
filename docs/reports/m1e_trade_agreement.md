# M1e — İşlem düzeyinde başlangıç duyarlılığı (S17, 20 sembol, 15 dk)

Son 4000 barda (≈41 gün) açılan işlemler; anahtar = (sembol, mutlak açılış barı, yön).
Referans: 30000 bar ısınmalı koşu (16 işlem). Maliyet: "original". Kelly ve havuz AÇIK (S17).

| Isınma | Koşudaki işlem | Referansla ortak | Yalnız referans | Yalnız bu koşu |
|---:|---:|---:|---:|---:|
| 0 | 25 | 0 | 16 | 25 |
| 1000 | 20 | 3 | 13 | 17 |
| 4000 | 9 | 2 | 14 | 7 |
| 8000 | 6 | 1 | 15 | 5 |
| 16000 | 11 | 7 | 9 | 4 |
| 20000 | 1 | 0 | 16 | 1 |
| 40000 | 23 | 14 | 2 | 9 |
| 70000 | 5 | 2 | 14 | 3 |

Sınırlar: tek pencere, 20 sembol, küçük örneklem; kâr/zarar ölçülmedi. Aynı pencerede
sinyal hücre uyumu (m1d) %97'ye varırken işlem listeleri bile uzun ısınmalar arasında tutmuyor.
