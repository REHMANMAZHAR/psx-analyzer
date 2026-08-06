import io
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="PSX Analyzer", page_icon="📈", layout="wide")


@dataclass
class Level:
    kind: str
    price: float
    touches: int
    strength: int
    sources: str


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Yahoo/CSV OHLCV data into unique one-dimensional columns."""
    if df is None or df.empty:
        raise ValueError("No price data was returned.")

    df = df.copy()

    # yfinance may return MultiIndex columns such as ("Close", "SAZEW.KA").
    if isinstance(df.columns, pd.MultiIndex):
        price_names = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        flattened = []
        for col in df.columns:
            parts = [str(x).strip() for x in col if str(x).strip()]
            chosen = next((x for x in parts if x.title() in price_names), parts[0] if parts else "")
            flattened.append(chosen)
        df.columns = flattened

    df.columns = [str(c).strip().title() for c in df.columns]
    df = df.rename(columns={"Vol": "Volume"})

    # Keep the genuine Close column. Use Adj Close only when Close is absent.
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})
    elif "Close" in df.columns and "Adj Close" in df.columns:
        df = df.drop(columns=["Adj Close"])

    # Guard against any duplicate names returned by a provider.
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date")

    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]

    # Force every required field to a one-dimensional numeric Series.
    clean = pd.DataFrame(index=df.index)
    for c in needed:
        value = df[c]
        if isinstance(value, pd.DataFrame):
            value = value.iloc[:, 0]
        clean[c] = pd.to_numeric(value, errors="coerce")

    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(subset=["Open", "High", "Low", "Close"])
    clean["Volume"] = clean["Volume"].fillna(0)
    return clean.sort_index()


@st.cache_data(ttl=900, show_spinner=False)
def download_data(symbol: str, period: str) -> tuple[pd.DataFrame, str]:
    ticker = symbol.upper().strip()
    if not ticker.endswith(".KA"):
        ticker += ".KA"
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}")
    return normalize_columns(df), f"Yahoo Finance ({ticker})"


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for n in (20, 50, 200):
        d[f"EMA{n}"] = d["Close"].ewm(span=n, adjust=False).mean()
    prev_close = d["Close"].shift(1)
    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - prev_close).abs(),
        (d["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    d["ATR14"] = tr.rolling(14).mean()
    d["VOL20"] = d["Volume"].rolling(20).mean()
    d["RSI14"] = 100 - 100 / (1 + d["Close"].diff().clip(lower=0).rolling(14).mean() /
                                    (-d["Close"].diff().clip(upper=0).rolling(14).mean()))
    return d


def pivot_points(df: pd.DataFrame, window: int = 5) -> tuple[list[float], list[float]]:
    highs, lows = [], []
    h, l = df["High"], df["Low"]
    for i in range(window, len(df) - window):
        if h.iloc[i] >= h.iloc[i-window:i+window+1].max():
            highs.append(float(h.iloc[i]))
        if l.iloc[i] <= l.iloc[i-window:i+window+1].min():
            lows.append(float(l.iloc[i]))
    return highs, lows


def cluster_prices(prices: list[float], tolerance: float) -> list[tuple[float, int]]:
    if not prices:
        return []
    prices = sorted(prices)
    clusters: list[list[float]] = [[prices[0]]]
    for p in prices[1:]:
        center = float(np.mean(clusters[-1]))
        if abs(p - center) <= tolerance:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [(float(np.mean(c)), len(c)) for c in clusters]


def calculate_levels(df: pd.DataFrame) -> tuple[list[Level], float]:
    d = add_indicators(df)
    current = float(d["Close"].iloc[-1])
    atr = float(d["ATR14"].iloc[-1]) if pd.notna(d["ATR14"].iloc[-1]) else current * 0.02
    tolerance = max(atr * 0.45, current * 0.006)
    highs, lows = pivot_points(d.tail(min(len(d), 500)), 5)

    # Add widely watched dynamic/statistical levels as candidates.
    candidates: list[tuple[float, str, int]] = []
    for price, touches in cluster_prices(lows, tolerance):
        candidates.append((price, "Swing lows", min(55, 20 + touches * 9)))
    for price, touches in cluster_prices(highs, tolerance):
        candidates.append((price, "Swing highs", min(55, 20 + touches * 9)))

    last = d.iloc[-1]
    for n, weight in [(20, 12), (50, 18), (200, 25)]:
        val = float(last[f"EMA{n}"])
        candidates.append((val, f"EMA{n}", weight))
    for lookback, weight, label in [(20, 12, "20-day range"), (60, 18, "60-day range"), (252, 25, "52-week range")]:
        sample = d.tail(min(lookback, len(d)))
        candidates.append((float(sample["Low"].min()), label, weight))
        candidates.append((float(sample["High"].max()), label, weight))

    # Merge candidates that overlap into confluence zones.
    candidates.sort(key=lambda x: x[0])
    merged: list[list[tuple[float, str, int]]] = []
    for item in candidates:
        if not merged or abs(item[0] - np.mean([x[0] for x in merged[-1]])) > tolerance:
            merged.append([item])
        else:
            merged[-1].append(item)

    levels: list[Level] = []
    for group in merged:
        price = float(np.average([x[0] for x in group], weights=[max(x[2], 1) for x in group]))
        sources = sorted(set(x[1] for x in group))
        touch_proxy = sum(1 for x in group if "Swing" in x[1]) + len(group)
        score = min(100, sum(x[2] for x in group) + min(20, len(sources) * 4))
        if abs(price - current) < tolerance * 0.35:
            continue
        levels.append(Level("Support" if price < current else "Resistance", price, touch_proxy, score, ", ".join(sources)))

    supports = sorted([x for x in levels if x.kind == "Support"], key=lambda x: current - x.price)[:3]
    resistances = sorted([x for x in levels if x.kind == "Resistance"], key=lambda x: x.price - current)[:3]
    return supports + resistances, atr


def trend_label(d: pd.DataFrame) -> tuple[str, int]:
    last = d.iloc[-1]
    score = 0
    score += 1 if last.Close > last.EMA20 else -1
    score += 1 if last.EMA20 > last.EMA50 else -1
    score += 1 if last.EMA50 > last.EMA200 else -1
    if score == 3:
        return "Strong Uptrend", score
    if score >= 1:
        return "Positive / Developing", score
    if score == -3:
        return "Strong Downtrend", score
    return "Mixed / Range", score


def chart(df: pd.DataFrame, levels: list[Level], symbol: str) -> go.Figure:
    d = add_indicators(df).tail(180)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low, close=d.Close, name=symbol))
    for n in (20, 50, 200):
        fig.add_trace(go.Scatter(x=d.index, y=d[f"EMA{n}"], mode="lines", name=f"EMA{n}", line={"width": 1.2}))
    for level in levels:
        fig.add_hline(y=level.price, line_dash="dash", annotation_text=f"{level.kind[0]} {level.price:,.2f}", annotation_position="right")
    fig.update_layout(height=650, xaxis_rangeslider_visible=False, margin=dict(l=10, r=20, t=40, b=10), legend_orientation="h")
    return fig


st.title("PSX Analyzer — Phase 1")
st.caption("Type a PSX symbol. The app finds support/resistance zones, trend, volatility, stop-loss and targets. Levels are probabilities—not guarantees.")

with st.sidebar:
    st.header("Stock Search")
    symbol = st.text_input("PSX symbol", value="SAZEW", help="Examples: LUCK, OGDC, INDU, EFERT, SAZEW")
    period = st.selectbox("History", ["1y", "2y", "5y", "10y"], index=2)
    uploaded = st.file_uploader("Optional OHLCV CSV", type=["csv"], help="Columns: Date, Open, High, Low, Close, Volume")
    analyze = st.button("Analyze Stock", type="primary", use_container_width=True)
    st.divider()
    st.caption("Free online data may be delayed or occasionally incomplete. The app displays the source and last trading date so stale data is obvious.")

if analyze or "ran" not in st.session_state:
    st.session_state.ran = True
    try:
        with st.spinner("Loading prices and calculating levels..."):
            if uploaded is not None:
                df = normalize_columns(pd.read_csv(uploaded))
                source = "Uploaded CSV"
            else:
                df, source = download_data(symbol, period)
            if len(df) < 60:
                raise ValueError("At least 60 daily candles are required; 200+ is preferred.")
            d = add_indicators(df)
            levels, atr = calculate_levels(df)
            last = d.iloc[-1]
            trend, trend_score = trend_label(d)
            current = float(last.Close)
            supports = [x for x in levels if x.kind == "Support"]
            resistances = [x for x in levels if x.kind == "Resistance"]
            s1 = supports[0].price if supports else current - atr
            r1 = resistances[0].price if resistances else current + atr

        st.success(f"Loaded {len(df):,} sessions from {source}. Last candle: {df.index[-1].date().isoformat()}")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Last Price", f"Rs {current:,.2f}")
        c2.metric("Trend", trend)
        c3.metric("ATR (14)", f"Rs {atr:,.2f}", f"{atr/current:.2%} volatility")
        c4.metric("Nearest Support", f"Rs {s1:,.2f}", f"{(s1/current-1):.2%}")
        c5.metric("Nearest Resistance", f"Rs {r1:,.2f}", f"{(r1/current-1):.2%}")

        tab1, tab2, tab3 = st.tabs(["Chart & Levels", "Trade Plan", "Data & Method"])
        with tab1:
            st.plotly_chart(chart(df, levels, symbol.upper()), use_container_width=True)
            table = pd.DataFrame([{
                "Type": x.kind,
                "Level": round(x.price, 2),
                "Zone Low": round(x.price - atr * 0.25, 2),
                "Zone High": round(x.price + atr * 0.25, 2),
                "Distance": f"{x.price/current-1:.2%}",
                "Strength / 100": x.strength,
                "Evidence": x.sources,
            } for x in levels])
            st.dataframe(table, use_container_width=True, hide_index=True)

        with tab2:
            entry_low, entry_high = s1 - atr * 0.25, s1 + atr * 0.25
            stop = s1 - atr * 0.75
            target1 = r1
            target2 = resistances[1].price if len(resistances) > 1 else r1 + atr
            risk = max(entry_high - stop, 0.01)
            reward = max(target1 - entry_high, 0)
            a, b = st.columns(2)
            with a:
                st.subheader("Pullback Plan")
                st.write(f"**Watch zone:** Rs {entry_low:,.2f}–{entry_high:,.2f}")
                st.write("**Trigger:** bullish rejection candle and close back above the zone; preferably with improving volume.")
                st.write(f"**Indicative invalidation:** below Rs {stop:,.2f}")
            with b:
                st.subheader("Targets")
                st.write(f"**Target 1:** Rs {target1:,.2f}")
                st.write(f"**Target 2:** Rs {target2:,.2f}")
                st.write(f"**Approx. reward/risk to Target 1:** {reward/risk:.2f}×")
            if reward/risk < 2:
                st.warning("The nearest setup offers less than 2:1 reward/risk. Waiting for a better entry or a confirmed breakout may be wiser.")
            else:
                st.info("The numerical reward/risk is acceptable, but entry still requires price confirmation and market/sector alignment.")

        with tab3:
            st.write(f"**Source:** {source}")
            st.write(f"**Last available candle:** {df.index[-1]}")
            st.write("**Method:** confirmed five-bar swing highs/lows are clustered using an ATR-based tolerance, then combined with EMA20/50/200 and 20-day, 60-day and 52-week range levels. Overlapping evidence receives a higher strength score.")
            st.write("**Important:** free Yahoo data is convenient but is not an official PSX real-time feed. For execution decisions, compare the last price with your broker terminal or the PSX Data Portal.")
            csv = df.reset_index().to_csv(index=False).encode("utf-8")
            st.download_button("Download loaded OHLCV", csv, file_name=f"{symbol.upper()}_ohlcv.csv", mime="text/csv")

    except Exception as exc:
        st.error(f"Could not analyze {symbol.upper()}: {exc}")
        st.info("Try the exact PSX symbol, or upload a CSV containing Date, Open, High, Low, Close and Volume.")
