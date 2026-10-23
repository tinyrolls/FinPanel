from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from data import api

_LOOKBACK_DAYS = 460


def _ohlc_frame(ticker: str, as_of: str) -> pd.DataFrame:
    start = (datetime.fromisoformat(str(as_of)[:10]) - timedelta(days=_LOOKBACK_DAYS)).date().isoformat()
    df = api.get_price_data(ticker, start, str(as_of)[:10])
    if df is None or len(df) == 0 or "close" not in getattr(df, "columns", []):
        return pd.DataFrame()
    idx = pd.to_datetime(df["date"]) if "date" in df.columns else pd.to_datetime(df.index)
    out = pd.DataFrame(index=idx)
    out["close"] = pd.Series(df["close"].to_numpy(dtype=float), index=idx)
    out["high"] = pd.Series(df["high"].to_numpy(dtype=float), index=idx) if "high" in df.columns else out["close"]
    out["low"] = pd.Series(df["low"].to_numpy(dtype=float), index=idx) if "low" in df.columns else out["close"]
    out["volume"] = pd.Series(df["volume"].to_numpy(dtype=float), index=idx) if "volume" in df.columns else 0.0
    out = out.sort_index()
    return out[~out.index.duplicated(keep="last")]


def _fmt_pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def _fmt_vol(x: float) -> str:
    if x >= 1e9:
        return f"{x / 1e9:.2f}B"
    if x >= 1e6:
        return f"{x / 1e6:.1f}M"
    if x >= 1e3:
        return f"{x / 1e3:.1f}K"
    return f"{x:.0f}"


def compute_technical_brief(ticker: str, as_of: str) -> str:
    frame = _ohlc_frame(ticker, as_of)
    if len(frame) < 26:
        return ""
    c = frame["close"]

    last = float(c.iloc[-1])
    lines: list[str] = []

    rets = []
    if len(c) >= 6:
        rets.append(f"1w {_fmt_pct(last / float(c.iloc[-6]) - 1.0)}")
    if len(c) >= 21:
        rets.append(f"1m {_fmt_pct(last / float(c.iloc[-21]) - 1.0)}")
    if rets:
        lines.append("Return: " + ", ".join(rets))

    v = frame["volume"]
    v_last = float(v.iloc[-1])
    if v_last > 0 and len(v) >= 20:
        v_avg = float(v.rolling(20).mean().iloc[-1])
        if pd.notna(v_avg) and v_avg > 0:
            lines.append(f"Volume: {_fmt_vol(v_last)} vs 20d avg {_fmt_vol(v_avg)} ({v_last / v_avg:.2f}x)")

    ma = {n: (c.rolling(n).mean().iloc[-1] if len(c) >= n else None) for n in (10, 20, 50, 200)}
    above = [f"{n}d" for n in (10, 20, 50, 200) if ma[n] is not None and pd.notna(ma[n]) and last >= ma[n]]
    have = [n for n in (10, 20, 50, 200) if ma[n] is not None and pd.notna(ma[n])]
    if have:
        rel = "above " + "/".join(above) + " MA" if above else "below all MAs"
        trend = ""
        if ma[50] is not None and pd.notna(ma[50]) and ma[20] is not None and pd.notna(ma[20]):
            trend = f"; 20d MA {'above' if ma[20] >= ma[50] else 'below'} 50d MA"
        lines.append(f"MAs: price {rel}{trend}")

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = float(macd.iloc[-1] - signal.iloc[-1])
    lines.append(f"MACD(12,26,9): line {float(macd.iloc[-1]):+.2f}, signal {float(signal.iloc[-1]):+.2f}, hist {hist:+.2f}")

    if len(c) >= 15:
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
        loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
        if pd.notna(gain) and pd.notna(loss):
            rsi = 100.0 if loss == 0 else 100.0 - 100.0 / (1.0 + gain / loss)
            lines.append(f"RSI(14): {rsi:.0f}")

    if len(c) >= 20:
        ma = c.rolling(20).mean()
        sd = c.rolling(20).std(ddof=0)
        upper = float(ma.iloc[-1] + 2 * sd.iloc[-1])
        lower = float(ma.iloc[-1] - 2 * sd.iloc[-1])
        if upper > lower:
            pct_b = (last - lower) / (upper - lower)
            lines.append(f"Bollinger(20,2): %B {pct_b:.2f} (band {lower:.2f}–{upper:.2f})")

    if len(frame) >= 9:
        low9 = frame["low"].rolling(9).min()
        high9 = frame["high"].rolling(9).max()
        rng = (high9 - low9).replace(0, pd.NA)
        rsv = ((frame["close"] - low9) / rng * 100.0).fillna(50.0)
        k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
        d = k.ewm(alpha=1 / 3, adjust=False).mean()
        kv, dv = float(k.iloc[-1]), float(d.iloc[-1])
        jv = 3 * kv - 2 * dv
        lines.append(f"KDJ(9,3,3): K{kv:.0f}/D{dv:.0f}/J{jv:.0f}")

    if not lines:
        return ""
    return "\n".join(lines)
