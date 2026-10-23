from __future__ import annotations

import pandas as pd


def wilder_atr(ohlc: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = ohlc["high"], ohlc["low"], ohlc["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


class BaseTimingStrategy:

    name: str = "base"

    def signal(self, ohlc: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    @staticmethod
    def _frame(close: pd.Series, ohlc: pd.DataFrame | None) -> pd.DataFrame:
        return ohlc if ohlc is not None else close.to_frame("close")

    def prepare(self, close: pd.Series, ohlc: pd.DataFrame | None = None) -> None:
        self._sig = self.signal(self._frame(close, ohlc))

    def decide(self, ctx) -> float:
        sig = getattr(self, "_sig", None)
        if sig is not None and ctx.date in sig.index:
            v = sig.loc[ctx.date]
        else:
            frame = ctx.ohlc if ctx.ohlc is not None else ctx.history.to_frame("close")
            s = self.signal(frame)
            v = s.iloc[-1] if len(s) else 0
        return float(v) if pd.notna(v) else 0.0


class BuyAndHold(BaseTimingStrategy):
    name = "buy_and_hold"

    def signal(self, ohlc: pd.DataFrame) -> pd.Series:
        return pd.Series(1, index=ohlc.index)


class SMACross(BaseTimingStrategy):
    name = "sma_cross"

    def __init__(self, short: int = 20, long: int = 60):
        self.short, self.long = short, long

    def signal(self, ohlc: pd.DataFrame) -> pd.Series:
        close = ohlc["close"]
        s = close.rolling(self.short, min_periods=self.short).mean()
        l = close.rolling(self.long, min_periods=self.long).mean()
        sig = (s > l).astype(int)
        sig[l.isna()] = 0
        return sig


class WMACross(BaseTimingStrategy):
    name = "wma_cross"

    def __init__(self, short: int = 20, long: int = 60):
        self.short, self.long = short, long

    @staticmethod
    def _wma(close: pd.Series, n: int) -> pd.Series:
        w = pd.Series(range(1, n + 1), dtype=float)
        return close.rolling(n).apply(lambda x: (x * w.values).sum() / w.sum(), raw=True)

    def signal(self, ohlc: pd.DataFrame) -> pd.Series:
        close = ohlc["close"]
        s = self._wma(close, self.short)
        l = self._wma(close, self.long)
        sig = (s > l).astype(int)
        sig[l.isna()] = 0
        return sig


class BollingerBands(BaseTimingStrategy):

    name = "bollinger_bands"

    def __init__(self, window: int = 20, k: float = 2.0):
        self.window, self.k = window, k

    def signal(self, ohlc: pd.DataFrame) -> pd.Series:
        close = ohlc["close"]
        ma = close.rolling(self.window, min_periods=self.window).mean()
        sd = close.rolling(self.window, min_periods=self.window).std(ddof=0)
        lower = ma - self.k * sd
        sig, holding = [], 0
        for px, lo, mid, ok in zip(close, lower, ma, ma.notna()):
            if not ok:
                sig.append(0)
                continue
            if holding == 0 and px < lo:
                holding = 1
            elif holding == 1 and px > mid:
                holding = 0
            sig.append(holding)
        return pd.Series(sig, index=close.index)


class TurnOfTheMonth(BaseTimingStrategy):

    name = "turn_of_the_month"

    def __init__(self, tail: int = 1, head: int = 3):
        self.tail, self.head = tail, head

    def signal(self, ohlc: pd.DataFrame) -> pd.Series:
        d = pd.Series(pd.DatetimeIndex(ohlc.index), index=ohlc.index)
        ym = d.dt.to_period("M")
        pos_from_start = d.groupby(ym).cumcount()
        pos_from_end = d.groupby(ym).cumcount(ascending=False)
        sig = ((pos_from_start < self.head) | (pos_from_end < self.tail)).astype(int)
        return pd.Series(sig.to_numpy(), index=ohlc.index)


class ATRBand(BaseTimingStrategy):

    name = "atr_band"

    def __init__(self, atr_period: int = 14, multiplier: float = 1.5):
        self.atr_period, self.multiplier = atr_period, multiplier

    def signal(self, ohlc: pd.DataFrame) -> pd.Series:
        sma = ohlc["close"].rolling(self.atr_period, min_periods=self.atr_period).mean()
        atr = wilder_atr(ohlc, self.atr_period)
        upper = sma + atr * self.multiplier
        lower = sma - atr * self.multiplier
        low, high = ohlc["low"].to_numpy(), ohlc["high"].to_numpy()
        lo_b, up_b = lower.to_numpy(), upper.to_numpy()
        ok = sma.notna().to_numpy()
        sig, holding = [], 0
        for i in range(len(ohlc)):
            if not ok[i] or i == 0 or not ok[i - 1]:
                sig.append(holding if ok[i] else 0)
                continue
            cross_below_lower = low[i] <= lo_b[i] and low[i - 1] > lo_b[i - 1]
            cross_above_upper = high[i] >= up_b[i] and high[i - 1] < up_b[i - 1]
            if holding == 0 and cross_below_lower:
                holding = 1
            elif holding == 1 and cross_above_upper:
                holding = 0
            sig.append(holding)
        return pd.Series(sig, index=ohlc.index)


class TrendFollowing(BaseTimingStrategy):

    name = "trend_following"

    def __init__(self, period: int = 20, atr_period: int = 10, leverage: float = 0.95):
        self.period, self.atr_period, self.leverage = period, atr_period, leverage

    def signal(self, ohlc: pd.DataFrame) -> pd.Series:
        close = ohlc["close"]
        highest = close.rolling(self.period, min_periods=self.period).max()
        std = close.rolling(self.period, min_periods=self.period).std(ddof=0)
        atr = wilder_atr(ohlc, self.atr_period)
        c = close.to_numpy()
        hi = highest.to_numpy()
        sd = std.to_numpy()
        a = atr.to_numpy()
        ok = (highest.notna() & atr.notna()).to_numpy()
        sig, holding = [], 0
        for i in range(len(ohlc)):
            if not ok[i]:
                sig.append(0)
                continue
            if holding == 0 and c[i] >= hi[i] and sd[i] > 0:
                holding = 1
            elif holding == 1 and c[i] < hi[i] - 2.0 * a[i]:
                holding = 0
            sig.append(self.leverage if holding else 0.0)
        return pd.Series(sig, index=ohlc.index, dtype=float)


from baselines.predictors import PREDICTOR_STRATEGIES

class MACDCross(BaseTimingStrategy):

    name = "macd"

    def __init__(self, fast: int = 12, slow: int = 26, signal_span: int = 9):
        self.fast, self.slow, self.signal_span = fast, slow, signal_span

    def signal(self, ohlc: pd.DataFrame) -> pd.Series:
        close = ohlc["close"]
        macd = close.ewm(span=self.fast, adjust=False).mean() - close.ewm(span=self.slow, adjust=False).mean()
        signal = macd.ewm(span=self.signal_span, adjust=False).mean()
        sig = (macd > signal).astype(int)
        sig.iloc[: self.slow] = 0
        return sig


STRATEGIES = {
    cls.name: cls
    for cls in (
        BuyAndHold,
        SMACross,
        WMACross,
        MACDCross,
        BollingerBands,
        TurnOfTheMonth,
        ATRBand,
        TrendFollowing,
    )
}
STRATEGIES.update(PREDICTOR_STRATEGIES)
