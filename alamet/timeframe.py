"""
Sinyal zaman dilimi: tek yapılandırma değeri ve Pine'ın `ssTf` / `ssLag` / `enforceTf` kuralları.

Kaynak Pine: `ssTf` (78), `enforceTf` + `runtime.error` (80, 303), `ssLag` ve `ssNew` (326-327),
günlük open saati notu (`dOpenHr`, 81). Pine'da bütün işlem mantığı yalnızca sinyal TF'sinin barı
kapandığında çalışır; süre parametreleri (tutma, bekleme, sönümleme) bu TF'nin BAR sayısıdır.
Bu yüzden 15 dk'da 96 bar 24 saat, 5 dk'da 8 saat, 1 dk'da 1,6 saat demektir.

Bot ayrı bir "grafik periyodu" taşımaz; veriyi sinyal TF'sinde alır (ince veriyi önce yeniden
örnekler). `ssLag` yalnızca TradingView karşılaştırmasında anlamlıdır (grafik < sinyal TF).
"""

from __future__ import annotations

from dataclasses import dataclass

SECONDS_PER_DAY = 86400
DEFAULT_SECONDS = 900
INTERVALS = {
    60: "1m", 180: "3m", 300: "5m", 900: "15m", 1800: "30m", 3600: "1h", 7200: "2h",
    14400: "4h", 21600: "6h", 28800: "8h", 43200: "12h", 86400: "1d",
}  # fmt: skip
DAILY_OPEN_MAX_SECONDS = 3600


@dataclass(frozen=True)
class SignalTf:
    """Sinyal TF'si (saniye). Binance'in desteklediği ve günü tam bölen süreler geçerlidir."""

    seconds: int = DEFAULT_SECONDS

    def __post_init__(self) -> None:
        if self.seconds not in INTERVALS:
            raise ValueError(f"Desteklenmeyen sinyal TF'si: {self.seconds} sn")

    @property
    def interval(self) -> str:
        return INTERVALS[self.seconds]

    @property
    def millis(self) -> int:
        return self.seconds * 1000

    @property
    def bars_per_day(self) -> int:
        return SECONDS_PER_DAY // self.seconds

    @property
    def bars_per_year(self) -> int:
        return self.bars_per_day * 365

    @property
    def last_bar_offset(self) -> int:
        """Günün son barının gün başlangıcına uzaklığı (sn)."""
        return SECONDS_PER_DAY - self.seconds


def from_interval(text: str) -> SignalTf:
    """ "5m" gibi Binance aralığından."""
    for seconds, name in INTERVALS.items():
        if name == text:
            return SignalTf(seconds)
    raise ValueError(f"Bilinmeyen aralık: {text}")


def ss_lag(chart_seconds: int, signal_seconds: int) -> int:
    """Pine `ssLag`: grafik sinyal TF'sinden küçükse 1 (sinyal bir üst-TF barı gecikmeli okunur)."""
    return 1 if chart_seconds < signal_seconds else 0


def enforce_chart_tf(chart_seconds: int, signal_seconds: int, enforce: bool = True) -> None:
    """Pine `enforceTf`: grafik periyodu sinyal TF'sinden BÜYÜKSE hata."""
    if enforce and chart_seconds > signal_seconds:
        raise ValueError(
            f"Grafik periyodu sinyal zaman diliminden ({signal_seconds} sn) BÜYÜK olamaz"
        )


def check_daily_open(daily_open_hour: int, signal_seconds: int) -> bool:
    """`dOpenHr` ≠ 0 yalnızca sinyal TF ≤ 60 dk iken doğru çalışır (Pine açıklaması); Pine hata
    vermez, burada yalnızca durumu bildiririz."""
    return daily_open_hour == 0 or signal_seconds <= DAILY_OPEN_MAX_SECONDS
