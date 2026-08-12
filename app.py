import io
from dataclasses import dataclass
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

st.set_page_config(page_title="PSX Analyzer", page_icon="📈", layout="wide")

KARACHI_TZ = ZoneInfo("Asia/Karachi")
PSX_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/html,application/json",
}


@dataclass
class Level:
    kind: str
    price: float
    touches: int
    strength: int
    sources: str


# --------------------------------------------------------------------------------------
# Live PSX data (PSX Data Portal — dps.psx.com.pk). This is PSX's own free public feed,
# the same one that powers the "Market Watch" page on their site. It is delayed by a few
# minutes during trading hours (PSX marks it "delayed 5 minutes unless otherwise
# indicated"), not a paid tick-by-tick feed, but it is far fresher than a prior-day close.
# --------------------------------------------------------------------------------------

def is_market_open(now: datetime | None = None) -> bool:
    """Rough PSX regular-market session check (Mon-Fri, 09:15-15:30 PKT).
    This is an approximation for display purposes only — always confirm with your
    broker terminal or the PSX site near market open/close and on holidays."""
    now = now or datetime.now(KARACHI_TZ)
    if now.weekday() >= 5:
        return False
    return dtime(9, 15) <= now.time() <= dtime(15, 30)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_quote(symbol: str) -> dict:
    """Scrape the current row for `symbol` from the PSX Data Portal market-watch table."""
    target = symbol.upper().strip()
    resp = requests.get("https://dps.psx.com.pk/market-watch", headers=PSX_HEADERS, timeout=12)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    for tr in soup.find_all("tr"):
        link = tr.find("a")
        if not link:
            continue
        if link.get_text(strip=True).upper() != target:
            continue
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        nums = []
        for c in cells:
            cleaned = c.replace(",", "").replace("%", "").replace("+", "").strip()
            try:
                nums.append(float(cleaned))
            except ValueError:
                continue
        if len(nums) < 8:
            raise ValueError(f"Could not parse the market-watch row for {target}.")
        ldcp, open_, high, low, current, change, change_pct, volume = nums[-8:]
        return {
            "symbol": target,
            "ldcp": ldcp,
            "open": open_,
            "high": high,
            "low": low,
            "current": current,
            "change": change,
            "change_pct": change_pct,
            "volume": volume,
            "fetched_at": datetime.now(KARACHI_TZ),
        }
    raise ValueError(f"'{target}' was not found on the PSX market-watch page. Check the symbol spelling.")


@st.cache_data(ttl=45, show_spinner=False)
def fetch_intraday_ticks(symbol: str) -> pd.DataFrame:
    """Pull today's intraday tick series from the PSX Data Portal timeseries API."""
    target = symbol.upper().strip()
    resp = requests.post(f"https://dps.psx.com.pk/timeseries/int/{target}", headers=PSX_HEADERS, timeout=12)
    resp.raise_for_status()
    payload = resp.json()
    rows = payload.get("data", [])
    if not rows:
        raise ValueError("No intraday ticks returned yet for today.")
    df = pd.DataFrame(rows, columns=["ts", "price", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(KARACHI_TZ)
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def merge_live_into_daily(df: pd.DataFrame, live: dict) -> pd.DataFrame:
    """Fold the live PSX quote into the daily OHLCV history as today's (in-progress) candle,
    so EMAs, ATR and support/resistance reflect the live price rather than yesterday's close."""
    d = df.copy()
    today = pd.Timestamp(datetime.now(KARACHI_TZ).date())
    last_date = d.index[-1].normalize()

    row = {
        "Open": live["open"] if live["open"] else live["current"],
        "High": max(live["high"], live["current"]) if live["high"] else live["current"],
        "Low": min(live["low"], live["current"]) if live["low"] else live["current"],
        "Close": live["current"],
        "Volume": live["volume"],
    }

    if last_date == today:
        for k, v in row.items():
            d.loc[d.index[-1], k] = v
    else:
        d.loc[today] = row
    return d.sort_index()


# --------------------------------------------------------------------------------------
# Historical backbone (Yahoo Finance). Used for the longer EMA/support-resistance history
# that PSX's free portal does not provide; the live quote above then refreshes today's bar.
# --------------------------------------------------------------------------------------

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


def intraday_chart(ticks: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ticks["ts"], y=ticks["price"], mode="lines", name=symbol,
                              line={"width": 1.6, "color": "#1f77b4"}, fill="tozeroy",
                              fillcolor="rgba(31,119,180,0.08)"))
    fig.update_layout(height=320, margin=dict(l=10, r=20, t=30, b=10),
                       yaxis_title="Price (Rs)", showlegend=False)
    return fig


st.title("PSX Analyzer — Phase 1")
st.caption("Type a PSX symbol. The app pulls a live quote from the PSX Data Portal, finds support/resistance zones, "
           "trend, volatility, stop-loss and targets. Levels are probabilities—not guarantees.")

with st.sidebar:
    st.header("Stock Search")
    symbol = st.text_input("PSX symbol", value="SAZEW", help="Examples: LUCK, OGDC, INDU, EFERT, SAZEW")
    period = st.selectbox("History", ["1y", "2y", "5y", "10y"], index=2)
    use_live = st.checkbox("Use live PSX quote", value=True,
                            help="Pulls the current price from the PSX Data Portal (dps.psx.com.pk) and folds it "
                                 "into today's candle. Turn off to analyze on the last daily close only.")
    auto_refresh = False
    if use_live:
        auto_refresh = st.checkbox("Auto-refresh while open", value=False)
        if auto_refresh:
            refresh_secs = st.select_slider("Refresh every", options=[30, 60, 120, 300], value=60,
                                             format_func=lambda s: f"{s}s")
            if HAS_AUTOREFRESH:
                st_autorefresh(interval=refresh_secs * 1000, key="live_refresh")
            else:
                st.caption("Install `streamlit-autorefresh` to enable timed auto-refresh; use the button below meanwhile.")
    uploaded = st.file_uploader("Optional OHLCV CSV", type=["csv"], help="Columns: Date, Open, High, Low, Close, Volume")
    analyze = st.button("Analyze Stock", type="primary", use_container_width=True)
    st.divider()
    now_pk = datetime.now(KARACHI_TZ)
    status = "🟢 Market Open" if is_market_open(now_pk) else "🔴 Market Closed"
    st.caption(f"{status} · {now_pk.strftime('%Y-%m-%d %H:%M')} PKT (approximate session window)")
    st.caption("Live quotes come from the PSX Data Portal (dps.psx.com.pk), which the exchange itself marks as "
               "delayed a few minutes during trading. For execution decisions, always confirm with your broker terminal.")

if analyze or "ran" not in st.session_state:
    st.session_state.ran = True
    try:
        with st.spinner("Loading prices and calculating levels..."):
            if uploaded is not None:
                df = normalize_columns(pd.read_csv(uploaded))
                source = "Uploaded CSV"
                live = None
                live_error = None
            else:
                df, source = download_data(symbol, period)
                live, live_error = None, None
                if use_live:
                    try:
                        live = fetch_live_quote(symbol)
                        df = merge_live_into_daily(df, live)
                        source = f"{source} + PSX Data Portal live quote"
                    except Exception as exc:
                        live_error = str(exc)

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

        if live:
            as_of = live["fetched_at"].strftime("%Y-%m-%d %H:%M:%S %Z")
            st.success(f"Live PSX price loaded. As of {as_of} · Change {live['change']:+.2f} "
                       f"({live['change_pct']:+.2f}%) · Volume {live['volume']:,.0f}")
        else:
            st.success(f"Loaded {len(df):,} sessions from {source}. Last candle: {df.index[-1].date().isoformat()}")
            if use_live and live_error and uploaded is None:
                st.warning(f"Live PSX quote unavailable ({live_error}) — showing the last daily close instead.")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Last Price", f"Rs {current:,.2f}", f"{live['change_pct']:+.2f}%" if live else None)
        c2.metric("Trend", trend)
        c3.metric("ATR (14)", f"Rs {atr:,.2f}", f"{atr/current:.2%} volatility")
        c4.metric("Nearest Support", f"Rs {s1:,.2f}", f"{(s1/current-1):.2%}")
        c5.metric("Nearest Resistance", f"Rs {r1:,.2f}", f"{(r1/current-1):.2%}")

        tab1, tab2, tab3 = st.tabs(["Chart & Levels", "Trade Plan", "Data & Method"])
        with tab1:
            st.plotly_chart(chart(df, levels, symbol.upper()), use_container_width=True)

            if use_live and uploaded is None:
                st.markdown("**Today's intraday tape**")
                try:
                    ticks = fetch_intraday_ticks(symbol)
                    st.plotly_chart(intraday_chart(ticks, symbol.upper()), use_container_width=True)
                    st.caption(f"{len(ticks):,} ticks from the PSX Data Portal intraday feed, "
                               f"{ticks['ts'].iloc[0].strftime('%H:%M')}–{ticks['ts'].iloc[-1].strftime('%H:%M')} PKT.")
                except Exception as exc:
                    st.caption(f"Intraday tick chart unavailable right now ({exc}).")

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
            if live:
                st.write(f"**Live quote as of:** {live['fetched_at'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
                st.write(f"**LDCP (prior close):** Rs {live['ldcp']:,.2f}")
            st.write("**Method:** confirmed five-bar swing highs/lows are clustered using an ATR-based tolerance, then combined with EMA20/50/200 and 20-day, 60-day and 52-week range levels. Overlapping evidence receives a higher strength score. When live mode is on, today's PSX quote replaces or extends the most recent daily candle before indicators are recalculated.")
            st.write("**Important:** the live quote comes from the free PSX Data Portal (dps.psx.com.pk market-watch page), which the exchange marks as delayed by a few minutes during trading — it is not a paid real-time/tick feed. Yahoo Finance supplies the longer daily history used for EMAs and range levels. For execution decisions, always compare with your broker terminal or the PSX Data Portal directly.")
            csv = df.reset_index().to_csv(index=False).encode("utf-8")
            st.download_button("Download loaded OHLCV", csv, file_name=f"{symbol.upper()}_ohlcv.csv", mime="text/csv")

    except Exception as exc:
        st.error(f"Could not analyze {symbol.upper()}: {exc}")
        st.info("Try the exact PSX symbol, or upload a CSV containing Date, Open, High, Low, Close and Volume.")
