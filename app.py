
App · PY
import io
import time
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
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False
 
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False
 
st.set_page_config(page_title="PSX Analyzer", page_icon="📈", layout="wide")
 
KARACHI_TZ = ZoneInfo("Asia/Karachi")
PSX_BASE = "https://dps.psx.com.pk"
PSX_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": f"{PSX_BASE}/market-watch",
    "Origin": PSX_BASE,
    "X-Requested-With": "XMLHttpRequest",
    "Connection": "keep-alive",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "cors",
}
 
 
@st.cache_resource(show_spinner=False)
def _psx_session() -> requests.Session:
    """A shared, browser-like session. PSX's server appears to block bare scripted
    requests (no cookies/referer) from datacenter IPs — visiting the homepage first to
    pick up cookies, and sending the same headers its own page JS would send, gets past
    that without needing anything fancier."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": PSX_HEADERS["User-Agent"],
        "Accept-Language": "en-US,en;q=0.9",
    })
    try:
        s.get(PSX_BASE + "/", timeout=15)
        s.get(PSX_BASE + "/market-watch", timeout=15)
    except Exception:
        pass  # best-effort cookie warm-up; the real request below still tries regardless
    return s
 
 
def _psx_request(method: str, url: str, retries: int = 2, **kwargs) -> requests.Response:
    """requests.get/post through the shared PSX session, with a couple of quick retries
    for the transient connection resets PSX's edge occasionally throws at scripted clients."""
    session = _psx_session()
    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = session.request(method, url, headers=PSX_HEADERS, timeout=kwargs.pop("timeout", 15), **kwargs)
            if resp.status_code == 403 or resp.status_code >= 500 or resp.status_code == 469:
                last_exc = ValueError(f"PSX blocked the request (HTTP {resp.status_code})")
                time.sleep(0.6 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_exc = exc
            time.sleep(0.6 * (attempt + 1))
    raise last_exc
 
 
@dataclass
class Level:
    kind: str
    price: float
    touches: int
    strength: int
    sources: str
 
 
# --------------------------------------------------------------------------------------
# Buy-call logic. This is a deterministic, rules-based technical screen — not personalized
# investment advice. It flags a "BUY CALL" only when trend, entry proximity to support,
# reward:risk and RSI all line up; otherwise it explains what's missing.
# --------------------------------------------------------------------------------------
 
def evaluate_buy_call(current: float, atr: float, levels: list, rsi: float | None, trend_score: int) -> dict:
    supports = sorted([l for l in levels if l.kind == "Support"], key=lambda x: current - x.price)
    resistances = sorted([l for l in levels if l.kind == "Resistance"], key=lambda x: x.price - current)
 
    reasons = []
    if not supports or not resistances:
        return {"call": "NO SETUP", "entry_low": None, "entry_high": None, "stop": None,
                "target1": None, "target2": None, "rr": None,
                "reasons": ["Not enough confirmed support/resistance levels yet."]}
 
    s1, r1 = supports[0], resistances[0]
    entry_low, entry_high = s1.price - atr * 0.25, s1.price + atr * 0.25
    stop = s1.price - atr * 0.75
    target1 = r1.price
    target2 = resistances[1].price if len(resistances) > 1 else r1.price + atr
    risk = max(entry_high - stop, 0.01)
    reward = max(target1 - entry_high, 0)
    rr = reward / risk
 
    in_zone = entry_low - atr * 0.15 <= current <= entry_high + atr * 1.0
    trend_ok = trend_score >= 1
    rr_ok = rr >= 1.5
    rsi_ok = rsi is None or rsi < 70
 
    if not trend_ok:
        reasons.append("Trend not supportive (EMA20/50/200 not stacked bullishly)")
    if not in_zone:
        reasons.append(f"Price is not yet near the support/entry zone (Rs {entry_low:,.2f}–{entry_high:,.2f})")
    if not rr_ok:
        reasons.append(f"Reward:risk to Target 1 is only {rr:.2f}× (need ≥1.5×)")
    if not rsi_ok:
        reasons.append(f"RSI14 is overbought ({rsi:.0f})")
 
    if trend_ok and in_zone and rr_ok and rsi_ok:
        call = "BUY CALL"
    elif trend_ok and rr_ok and rsi_ok:
        call = "WATCH"
    else:
        call = "NO SETUP"
 
    return {"call": call, "entry_low": entry_low, "entry_high": entry_high, "stop": stop,
            "target1": target1, "target2": target2, "rr": rr, "reasons": reasons}
 
 
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
    resp = _psx_request("GET", f"{PSX_BASE}/market-watch", timeout=15)
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
    resp = _psx_request("POST", f"{PSX_BASE}/timeseries/int/{target}", timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    rows = payload.get("data", [])
    if not rows:
        raise ValueError("No intraday ticks returned yet for today.")
    df = pd.DataFrame(rows, columns=["ts", "price", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(KARACHI_TZ)
    df = df.sort_values("ts").reset_index(drop=True)
    return df
 
 
@st.cache_data(ttl=900, show_spinner=False)
def fetch_psx_eod(symbol: str) -> tuple[pd.DataFrame, bool]:
    """Daily history straight from the PSX Data Portal's own end-of-day timeseries feed.
    Returns (dataframe, has_full_ohlc). PSX's free feed sometimes only carries close+volume;
    when that happens the caller blends it onto a Yahoo OHLC backbone rather than faking
    high/low, which would silently wreck pivot/ATR-based levels and candlesticks."""
    target = symbol.upper().strip()
    resp = _psx_request("POST", f"{PSX_BASE}/timeseries/eod/{target}", timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    rows = payload.get("data", [])
    if not rows:
        raise ValueError("PSX Data Portal returned no end-of-day history for this symbol.")
 
    width = len(rows[0])
    if width >= 6:
        df = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close", "Volume"][:width])
        has_full_ohlc = True
    else:
        df = pd.DataFrame(rows, columns=["ts", "Close", "Volume"][:width])
        df["Open"] = df["Close"]
        df["High"] = df["Close"]
        df["Low"] = df["Close"]
        has_full_ohlc = False
 
    df.index = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert(KARACHI_TZ).dt.normalize().dt.tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Close"]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df, has_full_ohlc
 
 
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
# News. Pulled from Google News' free RSS search (no API key needed) — a headline
# aggregator, not a paid news feed. Only titles/links/sources are shown, never full
# article text. Keyword flags are a rough heuristic to help you scan faster, not a
# sentiment or impact score.
# --------------------------------------------------------------------------------------
 
NEWS_KEYWORDS = [
    "dividend", "bonus", "right issue", "rights issue", "results", "earnings", "profit",
    "loss", "secp", "psx notice", "credit rating", "downgrade", "upgrade", "acquisition",
    "merger", "stake", "investigation", "default", "circular debt", "tariff", "imf",
    "subsidy", "strike", "fire", "shutdown", "expansion", "buyback", "delisting",
    "management", "board meeting", "notification",
]
 
 
@st.cache_data(ttl=900, show_spinner=False)
def fetch_news(symbol: str, company_hint: str = "", max_items: int = 8) -> list[dict]:
    """Recent headlines for a PSX symbol via Google News RSS."""
    if not HAS_FEEDPARSER:
        raise RuntimeError("feedparser is not installed")
    query_text = f'"{symbol}" PSX stock'
    if company_hint:
        query_text = f'"{symbol}" OR "{company_hint}" PSX stock'
    query = requests.utils.quote(query_text)
    url = f"https://news.google.com/rss/search?q={query}&hl=en-PK&gl=PK&ceid=PK:en"
    feed = feedparser.parse(url)
    if getattr(feed, "bozo", 0) and not feed.entries:
        raise RuntimeError("news feed could not be parsed")
 
    items = []
    for entry in feed.entries[:max_items]:
        title = entry.get("title", "").strip()
        source = None
        if " - " in title:
            title, source = title.rsplit(" - ", 1)
        published = entry.get("published", "")
        if getattr(entry, "published_parsed", None):
            try:
                pub_dt = datetime(*entry.published_parsed[:6], tzinfo=ZoneInfo("UTC")).astimezone(KARACHI_TZ)
                published = pub_dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                pass
        matched = [kw for kw in NEWS_KEYWORDS if kw in title.lower()]
        items.append({
            "title": title,
            "source": source or "Google News",
            "published": published,
            "link": entry.get("link", ""),
            "keywords": matched,
        })
    return items
 
 
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
 
 
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_valid_symbols() -> set[str]:
    """The full list of tradeable PSX symbols, scraped once a day from the same market-watch
    table the live quote uses. Lets us reject a typo'd ticker in milliseconds instead of
    waiting on two slow, possibly-blocked network calls only to fail with a confusing error."""
    resp = _psx_request("GET", f"{PSX_BASE}/market-watch", timeout=20)
    soup = BeautifulSoup(resp.text, "html.parser")
    symbols = set()
    for tr in soup.find_all("tr"):
        link = tr.find("a")
        if link:
            txt = link.get_text(strip=True).upper()
            if txt:
                symbols.add(txt)
    return symbols
 
 
def load_historical(symbol: str, period: str, prefer_psx: bool) -> tuple[pd.DataFrame, str, str | None]:
    """Load daily OHLCV history, preferring PSX's own EOD feed when it carries full OHLC.
    If PSX only returns close+volume (no daily high/low), the Yahoo OHLC shape is kept and
    PSX's own close prices are substituted in wherever dates overlap, so the exchange's own
    print is used without breaking candlesticks or ATR/pivot-based support-resistance."""
    note = None
    target = symbol.upper().strip()
    try:
        valid = fetch_valid_symbols()
        if valid and target not in valid:
            raise ValueError(f"'{target}' was not found on PSX's own symbol list. Check the ticker spelling.")
    except ValueError:
        raise
    except Exception:
        pass  # validation call itself failed (e.g. PSX blocked it) — don't block the real fetch on that
 
    if not prefer_psx:
        df, source = download_data(symbol, period)
        return df, source, note
 
    try:
        psx_df, has_ohlc = fetch_psx_eod(symbol)
    except Exception as exc:
        df, source = download_data(symbol, period)
        note = f"PSX Data Portal history unavailable ({exc}) — used Yahoo Finance instead."
        return df, source, note
 
    if has_ohlc and len(psx_df) >= 60:
        return psx_df, "PSX Data Portal (EOD)", note
 
    try:
        yf_df, yf_source = download_data(symbol, period)
    except Exception as exc:
        if len(psx_df) >= 60:
            note = ("PSX's free EOD feed only provides close price + volume for this symbol "
                     "(no daily high/low); Yahoo Finance was also unavailable, so support/resistance "
                     "and ATR are based on close-only bars and will be less precise.")
            return psx_df, "PSX Data Portal (close-only)", note
        raise
 
    blended = yf_df.copy()
    overlap = blended.index.intersection(psx_df.index)
    if len(overlap) > 0:
        blended.loc[overlap, "Close"] = psx_df.loc[overlap, "Close"]
    note = ("PSX's free EOD feed only provides close price + volume (no daily high/low), so full "
            "OHLC history still comes from Yahoo Finance; PSX's own close prices are substituted in "
            "wherever the two overlap.")
    return blended, f"{yf_source} (OHLC) + PSX close prices", note
 
 
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
 
 
# --------------------------------------------------------------------------------------
# Breakout detection. Donchian channel break (classic turtle-trading signal) confirmed by
# volume, MACD histogram momentum and ADX/DI trend strength; plus a Bollinger-Bandwidth
# "squeeze" state that flags coiling volatility before a breakout happens.
# --------------------------------------------------------------------------------------
 
def add_breakout_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
 
    # Donchian channels use the PRIOR N bars (shifted), so "breakout" means today's close
    # actually cleared the highest high / lowest low of the preceding window, not itself.
    d["Donch20_Upper"] = d["High"].rolling(20).max().shift(1)
    d["Donch20_Lower"] = d["Low"].rolling(20).min().shift(1)
    d["Donch55_Upper"] = d["High"].rolling(55).max().shift(1)
    d["Donch55_Lower"] = d["Low"].rolling(55).min().shift(1)
 
    # Bollinger Bands (20, 2 sigma) + bandwidth percentile as a volatility-squeeze detector.
    mid = d["Close"].rolling(20).mean()
    std = d["Close"].rolling(20).std()
    d["BB_Mid"], d["BB_Upper"], d["BB_Lower"] = mid, mid + 2 * std, mid - 2 * std
    d["BB_Bandwidth"] = (d["BB_Upper"] - d["BB_Lower"]) / mid
    d["BB_Bandwidth_Pctile"] = d["BB_Bandwidth"].rolling(120).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
 
    # MACD (12, 26, 9)
    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_Hist"] = d["MACD"] - d["MACD_Signal"]
 
    # ADX(14) with +DI/-DI (Wilder-style smoothing via an equivalent EWM alpha).
    up_move = d["High"].diff()
    down_move = -d["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    prev_close = d["Close"].shift(1)
    tr = pd.concat([
        d["High"] - d["Low"], (d["High"] - prev_close).abs(), (d["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_w = tr.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=d.index).ewm(alpha=1 / 14, adjust=False).mean() / atr_w
    minus_di = 100 * pd.Series(minus_dm, index=d.index).ewm(alpha=1 / 14, adjust=False).mean() / atr_w
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    d["Plus_DI"], d["Minus_DI"] = plus_di, minus_di
    d["ADX14"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    return d
 
 
def evaluate_breakout(d: pd.DataFrame) -> dict:
    if len(d) < 60 or pd.isna(d["Donch20_Upper"].iloc[-1]):
        return {"signal": "NOT ENOUGH DATA", "donch20_upper": None, "donch20_lower": None,
                "volume_ratio": None, "adx": None, "macd_hist": None, "bb_bandwidth_pctile": None,
                "reasons": ["Need more price history to compute breakout indicators."]}
 
    last = d.iloc[-1]
    prev_hist = d["MACD_Hist"].iloc[-2] if len(d) > 1 else np.nan
    vol_ratio = float(last["Volume"] / last["VOL20"]) if last.get("VOL20") else None
    adx_now = float(last["ADX14"]) if pd.notna(last["ADX14"]) else None
    adx_prev = float(d["ADX14"].iloc[-6]) if len(d) > 6 and pd.notna(d["ADX14"].iloc[-6]) else None
    macd_hist = float(last["MACD_Hist"]) if pd.notna(last["MACD_Hist"]) else None
    bb_pctile = float(last["BB_Bandwidth_Pctile"]) if pd.notna(last["BB_Bandwidth_Pctile"]) else None
 
    momentum_up = (macd_hist is not None and macd_hist > 0 and (pd.isna(prev_hist) or macd_hist > prev_hist)) or \
                  (adx_now is not None and adx_prev is not None and adx_now > adx_prev and adx_now > 20
                   and last["Plus_DI"] > last["Minus_DI"])
    momentum_down = (macd_hist is not None and macd_hist < 0 and (pd.isna(prev_hist) or macd_hist < prev_hist)) or \
                     (adx_now is not None and adx_prev is not None and adx_now > adx_prev and adx_now > 20
                      and last["Minus_DI"] > last["Plus_DI"])
    vol_ok = vol_ratio is not None and vol_ratio >= 1.3
 
    reasons, signal = [], "NONE"
    if last["Close"] > last["Donch20_Upper"]:
        if vol_ok and momentum_up:
            signal = "BREAKOUT UP"
        else:
            signal = "BREAKOUT UP (unconfirmed)"
            reasons.append(f"Volume is {vol_ratio:.1f}× the 20-day average (want ≥1.3×)" if vol_ratio else "Volume data unavailable")
            if not momentum_up:
                reasons.append("MACD/ADX momentum isn't confirming the move yet")
    elif last["Close"] < last["Donch20_Lower"]:
        if vol_ok and momentum_down:
            signal = "BREAKDOWN"
        else:
            signal = "BREAKDOWN (unconfirmed)"
            reasons.append(f"Volume is {vol_ratio:.1f}× the 20-day average (want ≥1.3×)" if vol_ratio else "Volume data unavailable")
            if not momentum_down:
                reasons.append("MACD/ADX momentum isn't confirming the move yet")
    elif bb_pctile is not None and bb_pctile <= 20:
        signal = "SQUEEZE (WATCH)"
        reasons.append(f"Bollinger bandwidth is in the tightest {bb_pctile:.0f}% of the last ~120 sessions — volatility is coiling.")
    else:
        reasons.append("Price is inside its 20-day Donchian channel and volatility isn't unusually tight.")
 
    return {"signal": signal, "donch20_upper": float(last["Donch20_Upper"]), "donch20_lower": float(last["Donch20_Lower"]),
            "volume_ratio": vol_ratio, "adx": adx_now, "macd_hist": macd_hist, "bb_bandwidth_pctile": bb_pctile,
            "reasons": reasons}
 
 
def chart(d_full: pd.DataFrame, levels: list[Level], symbol: str) -> go.Figure:
    d = d_full.tail(180)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low, close=d.Close, name=symbol))
    for n in (20, 50, 200):
        fig.add_trace(go.Scatter(x=d.index, y=d[f"EMA{n}"], mode="lines", name=f"EMA{n}", line={"width": 1.2}))
    if "Donch20_Upper" in d.columns:
        fig.add_trace(go.Scatter(x=d.index, y=d["Donch20_Upper"], mode="lines", name="20D High (Donchian)",
                                  line={"width": 1, "dash": "dot", "color": "#888888"}))
        fig.add_trace(go.Scatter(x=d.index, y=d["Donch20_Lower"], mode="lines", name="20D Low (Donchian)",
                                  line={"width": 1, "dash": "dot", "color": "#888888"}))
    for level in levels:
        fig.add_hline(y=level.price, line_dash="dash", annotation_text=f"{level.kind[0]} {level.price:,.2f}", annotation_position="right")
    fig.update_layout(height=650, xaxis_rangeslider_visible=False, margin=dict(l=10, r=20, t=40, b=10), legend_orientation="h")
    return fig
 
 
def macd_chart(d_full: pd.DataFrame) -> go.Figure:
    d = d_full.tail(180)
    colors = ["#16a765" if v >= 0 else "#e66550" for v in d["MACD_Hist"].fillna(0)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d.index, y=d["MACD_Hist"], name="MACD Histogram", marker_color=colors))
    fig.add_trace(go.Scatter(x=d.index, y=d["MACD"], name="MACD", line={"width": 1.2, "color": "#4a86e8"}))
    fig.add_trace(go.Scatter(x=d.index, y=d["MACD_Signal"], name="Signal", line={"width": 1.2, "color": "#cc3a21"}))
    fig.update_layout(height=230, margin=dict(l=10, r=20, t=25, b=10), legend_orientation="h")
    return fig
 
 
def adx_chart(d_full: pd.DataFrame) -> go.Figure:
    d = d_full.tail(180)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["ADX14"], name="ADX14", line={"width": 1.6, "color": "#8e63ce"}))
    fig.add_trace(go.Scatter(x=d.index, y=d["Plus_DI"], name="+DI", line={"width": 1, "color": "#16a765"}))
    fig.add_trace(go.Scatter(x=d.index, y=d["Minus_DI"], name="-DI", line={"width": 1, "color": "#e66550"}))
    fig.add_hline(y=20, line_dash="dot", annotation_text="Trend threshold (20)")
    fig.update_layout(height=230, margin=dict(l=10, r=20, t=25, b=10), legend_orientation="h")
    return fig
 
 
def intraday_chart(ticks: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ticks["ts"], y=ticks["price"], mode="lines", name=symbol,
                              line={"width": 1.6, "color": "#1f77b4"}, fill="tozeroy",
                              fillcolor="rgba(31,119,180,0.08)"))
    fig.update_layout(height=320, margin=dict(l=10, r=20, t=30, b=10),
                       yaxis_title="Price (Rs)", showlegend=False)
    return fig
 
 
def screen_symbol(symbol: str, period: str, use_live: bool, prefer_psx_history: bool = True, check_news: bool = False) -> dict:
    df, _source, _note = load_historical(symbol, period, prefer_psx_history)
    live = None
    if use_live:
        try:
            live = fetch_live_quote(symbol)
            df = merge_live_into_daily(df, live)
        except Exception:
            live = None
 
    if len(df) < 60:
        raise ValueError("not enough price history")
 
    d = add_breakout_indicators(add_indicators(df))
    levels, atr = calculate_levels(df)
    current = float(d["Close"].iloc[-1])
    trend, trend_score = trend_label(d)
    rsi = float(d["RSI14"].iloc[-1]) if pd.notna(d["RSI14"].iloc[-1]) else None
    verdict = evaluate_buy_call(current, atr, levels, rsi, trend_score)
    breakout = evaluate_breakout(d)
 
    news_flag = ""
    if check_news:
        try:
            news_items = fetch_news(symbol, max_items=5)
            flagged = [n for n in news_items if n["keywords"]]
            if flagged:
                kw = sorted({k for n in flagged for k in n["keywords"]})
                news_flag = "⚠ " + ", ".join(kw[:3])
            elif news_items:
                news_flag = "No notable headlines"
            else:
                news_flag = "—"
        except Exception:
            news_flag = "n/a"
 
    row = {
        "Symbol": symbol.upper().strip(),
        "Price": round(current, 2),
        "Trend": trend,
        "Call": verdict["call"],
        "Breakout": breakout["signal"],
        "Entry Zone": f"{verdict['entry_low']:.2f}–{verdict['entry_high']:.2f}" if verdict["entry_low"] else "—",
        "Stop": round(verdict["stop"], 2) if verdict["stop"] else None,
        "Target 1": round(verdict["target1"], 2) if verdict["target1"] else None,
        "Target 2": round(verdict["target2"], 2) if verdict["target2"] else None,
        "R:R": round(verdict["rr"], 2) if verdict["rr"] else None,
        "RSI14": round(rsi, 1) if rsi is not None else None,
        "Live": "Yes" if live else "No",
        "Notes": "; ".join(verdict["reasons"]) if verdict["reasons"] else "All criteria met",
    }
    if check_news:
        row["News"] = news_flag
    return row
 
 
# --------------------------------------------------------------------------------------
# Backtesting. Walk-forward test of the Breakout rule and a simplified (rolling-low-based)
# version of the Buy Call rule against this stock's own history. Entries fire the day AFTER
# a signal is confirmed (using that day's open), so there's no look-ahead; exits are the
# first of stop, target, or a fixed holding horizon. The Buy Call backtest intentionally
# uses a fast rolling-low proxy for "support" instead of the full multi-source confluence
# levels in the live Trade Plan — recomputing those for every historical day would be far
# too slow for a web request — so treat results as indicative, not identical to what the
# live badge would have said on any given historical day.
# --------------------------------------------------------------------------------------
 
def backtest_signals(df: pd.DataFrame, horizon: int = 10, stop_atr: float = 1.5, target_atr: float = 3.0) -> dict:
    d = add_breakout_indicators(add_indicators(df))
    d["RollingLow20"] = d["Low"].rolling(20).min()
    n = len(d)
 
    def run(entry_mask: pd.Series, label: str) -> dict:
        trades = []
        prev_active = False
        for i in range(n):
            active = bool(entry_mask.iloc[i]) if pd.notna(entry_mask.iloc[i]) else False
            if not active:
                prev_active = False
                continue
            if prev_active:
                continue  # only count fresh transitions into a signal, not every day it persists
            prev_active = True
            entry_idx = i + 1
            if entry_idx >= n:
                continue
            entry_price = float(d["Open"].iloc[entry_idx])
            atr = d["ATR14"].iloc[i]
            if pd.isna(atr) or atr <= 0 or pd.isna(entry_price):
                continue
            stop = entry_price - stop_atr * atr
            target = entry_price + target_atr * atr
            exit_price, exit_reason, hold_days = None, "horizon", 0
            for h in range(entry_idx, min(entry_idx + horizon, n)):
                hold_days = h - entry_idx + 1
                if d["Low"].iloc[h] <= stop:
                    exit_price, exit_reason = stop, "stop"
                    break
                if d["High"].iloc[h] >= target:
                    exit_price, exit_reason = target, "target"
                    break
            if exit_price is None:
                last_h = min(entry_idx + horizon, n) - 1
                exit_price = float(d["Close"].iloc[last_h])
                hold_days = last_h - entry_idx + 1
            ret_pct = (exit_price / entry_price - 1) * 100
            trades.append({"Entry Date": d.index[entry_idx].date().isoformat(), "Entry": round(entry_price, 2),
                            "Exit": round(exit_price, 2), "Return %": round(ret_pct, 2),
                            "Exit Reason": exit_reason, "Hold (days)": hold_days})
        if not trades:
            return {"label": label, "trades": 0, "win_rate": None, "avg_return": None,
                    "avg_winner": None, "avg_loser": None, "avg_hold_days": None, "records": trades}
        rets = [t["Return %"] for t in trades]
        wins = [r for r in rets if r > 0]
        losses = [r for r in rets if r <= 0]
        return {
            "label": label, "trades": len(trades),
            "win_rate": len(wins) / len(trades) * 100,
            "avg_return": float(np.mean(rets)),
            "avg_winner": float(np.mean(wins)) if wins else None,
            "avg_loser": float(np.mean(losses)) if losses else None,
            "avg_hold_days": float(np.mean([t["Hold (days)"] for t in trades])),
            "records": trades,
        }
 
    vol_ratio = d["Volume"] / d["VOL20"]
    vol_ok = vol_ratio >= 1.3
    macd_up = (d["MACD_Hist"] > 0) & (d["MACD_Hist"] > d["MACD_Hist"].shift(1))
    adx_up = (d["ADX14"] > 20) & (d["ADX14"] > d["ADX14"].shift(5)) & (d["Plus_DI"] > d["Minus_DI"])
    breakout_entry = (d["Close"] > d["Donch20_Upper"]) & vol_ok & (macd_up | adx_up)
 
    ema_stack_ok = (d["Close"] > d["EMA20"]) | (d["EMA20"] > d["EMA50"])
    near_support = (d["Low"] <= d["RollingLow20"] * 1.02) & (d["Close"] >= d["RollingLow20"] * 0.985)
    rsi_ok = d["RSI14"] < 70
    buycall_entry = ema_stack_ok & near_support & rsi_ok
 
    return {
        "breakout": run(breakout_entry.fillna(False), "Breakout Up"),
        "buy_call": run(buycall_entry.fillna(False), "Buy Call (rolling-low proxy)"),
    }
 
 
# --------------------------------------------------------------------------------------
# Portfolio tracker. Live P&L against user-entered holdings, plus a chandelier-style
# trailing stop (highest close since entry, minus 1.5×ATR) so a held position gets an
# evolving, mechanical stop suggestion rather than a static one that never adapts as a
# stock runs up.
# --------------------------------------------------------------------------------------
 
def evaluate_position(symbol: str, qty: float, entry_price: float, entry_date, period: str,
                       use_live: bool, prefer_psx_history: bool) -> dict:
    df, _source, _note = load_historical(symbol, period, prefer_psx_history)
    live = None
    if use_live:
        try:
            live = fetch_live_quote(symbol)
            df = merge_live_into_daily(df, live)
        except Exception:
            live = None
 
    if len(df) < 60:
        raise ValueError("not enough price history")
 
    d = add_breakout_indicators(add_indicators(df))
    current = float(d["Close"].iloc[-1])
    atr = float(d["ATR14"].iloc[-1]) if pd.notna(d["ATR14"].iloc[-1]) else current * 0.02
    trend, trend_score = trend_label(d)
    rsi = float(d["RSI14"].iloc[-1]) if pd.notna(d["RSI14"].iloc[-1]) else None
    levels, _ = calculate_levels(df)
    verdict = evaluate_buy_call(current, atr, levels, rsi, trend_score)
    breakout = evaluate_breakout(d)
 
    if entry_date is not None and not (isinstance(entry_date, float) and pd.isna(entry_date)):
        since = d[d.index >= pd.Timestamp(entry_date)]
        highest_close = float(since["Close"].max()) if len(since) else current
    else:
        highest_close = float(d.tail(60)["Close"].max())
    suggested_stop = highest_close - 1.5 * atr
 
    pnl = (current - entry_price) * qty
    pnl_pct = (current / entry_price - 1) * 100 if entry_price else None
 
    return {
        "Symbol": symbol.upper().strip(),
        "Quantity": qty,
        "Entry Price": round(entry_price, 2),
        "Current Price": round(current, 2),
        "P&L (Rs)": round(pnl, 2),
        "P&L %": round(pnl_pct, 2) if pnl_pct is not None else None,
        "Trend": trend,
        "Call": verdict["call"],
        "Breakout": breakout["signal"],
        "Trailing Stop": round(suggested_stop, 2),
        "Stop Breached": "⚠ Yes" if current < suggested_stop else "No",
        "Live": "Yes" if live else "No",
    }
 
 
st.title("PSX Analyzer — Phase 1")
st.caption("Type a PSX symbol. The app pulls a live quote from the PSX Data Portal, finds support/resistance zones, "
           "trend, volatility, stop-loss and targets. Levels are probabilities—not guarantees.")
 
with st.sidebar:
    st.header("Stock Search")
    symbol = st.text_input("PSX symbol", value="SAZEW", help="Examples: LUCK, OGDC, INDU, EFERT, SAZEW")
    period = st.selectbox("History", ["1y", "2y", "5y", "10y"], index=2)
    prefer_psx_history = st.checkbox("Prefer PSX Data Portal for daily history", value=True,
                                      help="Tries PSX's own end-of-day feed first. If PSX only returns close+volume "
                                           "(no daily high/low) for this symbol, Yahoo Finance's OHLC shape is kept "
                                           "and PSX's own close prices are used wherever they overlap — you'll see a note either way.")
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
    if "last_live_update" in st.session_state:
        st.caption(f"✅ Last successful PSX live update: {st.session_state.last_live_update}")
    else:
        st.caption("No successful PSX live update yet this session.")
    st.caption("Live quotes and history come from the PSX Data Portal (dps.psx.com.pk), which the exchange itself "
               "marks as delayed a few minutes during trading. For execution decisions, always confirm with your broker terminal.")
 
if analyze or "ran" not in st.session_state:
    st.session_state.ran = True
    try:
        with st.spinner("Loading prices and calculating levels..."):
            if uploaded is not None:
                df = normalize_columns(pd.read_csv(uploaded))
                source = "Uploaded CSV"
                hist_note = None
                live = None
                live_error = None
            else:
                df, source, hist_note = load_historical(symbol, period, prefer_psx_history)
                live, live_error = None, None
                if use_live:
                    try:
                        live = fetch_live_quote(symbol)
                        df = merge_live_into_daily(df, live)
                        source = f"{source} + PSX Data Portal live quote"
                        st.session_state.last_live_update = live["fetched_at"].strftime("%Y-%m-%d %H:%M:%S %Z")
                    except Exception as exc:
                        live_error = str(exc)
 
            if len(df) < 60:
                raise ValueError("At least 60 daily candles are required; 200+ is preferred.")
            d = add_breakout_indicators(add_indicators(df))
            levels, atr = calculate_levels(df)
            last = d.iloc[-1]
            trend, trend_score = trend_label(d)
            breakout = evaluate_breakout(d)
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
        if hist_note:
            st.caption(f"ℹ️ {hist_note}")
 
        if breakout["signal"] == "BREAKOUT UP":
            st.success(f"🚀 **BREAKOUT UP** — closed above the 20-day Donchian high (Rs {breakout['donch20_upper']:,.2f}) "
                       f"on {breakout['volume_ratio']:.1f}× average volume, with MACD/ADX confirming momentum.")
        elif breakout["signal"] == "BREAKOUT UP (unconfirmed)":
            st.info(f"🟡 Price broke above the 20-day high (Rs {breakout['donch20_upper']:,.2f}) but isn't fully "
                    f"confirmed yet: " + "; ".join(breakout["reasons"]))
        elif breakout["signal"] == "BREAKDOWN":
            st.error(f"🔻 **BREAKDOWN** — closed below the 20-day Donchian low (Rs {breakout['donch20_lower']:,.2f}) "
                     f"on {breakout['volume_ratio']:.1f}× average volume.")
        elif breakout["signal"] == "BREAKDOWN (unconfirmed)":
            st.info(f"🟡 Price broke below the 20-day low (Rs {breakout['donch20_lower']:,.2f}) but isn't fully "
                    f"confirmed yet: " + "; ".join(breakout["reasons"]))
        elif breakout["signal"] == "SQUEEZE (WATCH)":
            st.info("🌀 " + "; ".join(breakout["reasons"]))
 
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Last Price", f"Rs {current:,.2f}", f"{live['change_pct']:+.2f}%" if live else None)
        c2.metric("Trend", trend)
        c3.metric("ATR (14)", f"Rs {atr:,.2f}", f"{atr/current:.2%} volatility")
        c4.metric("Nearest Support", f"Rs {s1:,.2f}", f"{(s1/current-1):.2%}")
        c5.metric("Nearest Resistance", f"Rs {r1:,.2f}", f"{(r1/current-1):.2%}")
 
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ["Chart & Levels", "Trade Plan", "Data & Method", "📰 News", "🔎 Buy Call Screener", "🧪 Backtest"])
        with tab1:
            st.plotly_chart(chart(d, levels, symbol.upper()), use_container_width=True)
 
            st.markdown("**Momentum & trend strength (MACD, ADX/DI)**")
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                st.plotly_chart(macd_chart(d), use_container_width=True)
            with mcol2:
                st.plotly_chart(adx_chart(d), use_container_width=True)
            st.caption("MACD histogram shows momentum turning; ADX above 20 with +DI/-DI separation shows a "
                       "genuine trend (vs. a choppy range) — both are used to confirm the breakout signal above.")
 
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
            rsi_now = float(last.RSI14) if pd.notna(last.RSI14) else None
            verdict = evaluate_buy_call(current, atr, levels, rsi_now, trend_score)
 
            if verdict["call"] == "BUY CALL":
                st.success(f"✅ **BUY CALL** — {symbol.upper()} meets the entry, trend, reward:risk and RSI criteria right now.")
            elif verdict["call"] == "WATCH":
                st.info(f"👀 **WATCH** — {symbol.upper()} is trend/reward:risk OK but not yet in the entry zone.")
            else:
                st.warning(f"❌ **NO SETUP** — {symbol.upper()} does not currently meet the buy-call criteria.")
            if verdict["reasons"]:
                st.caption("Why: " + "; ".join(verdict["reasons"]))
 
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
            st.caption("This verdict is a deterministic technical rule-check (trend + support proximity + reward:risk + RSI) — not personalized financial advice.")
 
        with tab3:
            st.write(f"**Source:** {source}")
            st.write(f"**Last available candle:** {df.index[-1]}")
            if live:
                st.write(f"**Live quote as of:** {live['fetched_at'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
                st.write(f"**LDCP (prior close):** Rs {live['ldcp']:,.2f}")
            st.write("**Levels method:** confirmed five-bar swing highs/lows are clustered using an ATR-based tolerance, then combined with EMA20/50/200 and 20-day, 60-day and 52-week range levels. Overlapping evidence receives a higher strength score. When live mode is on, today's PSX quote replaces or extends the most recent daily candle before indicators are recalculated.")
            st.write("**Breakout method:** a 20-day Donchian channel break (yesterday's prior 20-session high/low) confirmed by volume ≥1.3× the 20-day average and either rising MACD histogram momentum or ADX(14) > 20 with +DI/-DI separation. A tight Bollinger-Bandwidth reading (bottom 20th percentile of the last ~120 sessions) is flagged separately as a pre-breakout 'squeeze' to watch.")
            st.write("**Data sourcing:** daily history is pulled from the PSX Data Portal's own end-of-day feed first (dps.psx.com.pk); when that feed only carries close price + volume for a symbol (no daily high/low), Yahoo Finance's OHLC shape is used instead with PSX's own close prices substituted in wherever they overlap — you'll see a note above when that happens. The live quote and today's intraday tape always come from the PSX Data Portal.")
            st.write("**Important:** the PSX Data Portal is the exchange's own free public feed (the same one behind its Market Watch page), which PSX marks as delayed by a few minutes during trading — it is not a paid real-time/tick or licensed historical feed. For execution decisions or bulk/commercial use, always compare with your broker terminal, or contact PSX's Market Data Team for a licensed feed.")
            csv = df.reset_index().to_csv(index=False).encode("utf-8")
            st.download_button("Download loaded OHLCV", csv, file_name=f"{symbol.upper()}_ohlcv.csv", mime="text/csv")
 
        with tab4:
            st.caption("Recent headlines from Google News for this symbol. Aggregated links only — always read the "
                       "full article on the source site before acting. Flags below are keyword matches, not sentiment analysis.")
            try:
                news_items = fetch_news(symbol, max_items=10)
                if not news_items:
                    st.info("No recent headlines found for this symbol.")
                for item in news_items:
                    flags = " ".join(f"`{k}`" for k in item["keywords"])
                    st.markdown(f"**[{item['title']}]({item['link']})**")
                    meta = f"{item['source']} · {item['published']}" if item["published"] else item["source"]
                    st.caption(meta + (f" · ⚠ {flags}" if flags else ""))
                    st.divider()
            except Exception as exc:
                st.caption(f"News unavailable right now ({exc}).")
 
        with tab5:
            st.caption("Scans a list of PSX symbols with the same trend / support-resistance / reward:risk / RSI rules "
                       "used above, and lists every symbol that currently has an active **BUY CALL** and/or a "
                       "**BREAKOUT** signal. This is a deterministic technical screen, not personalized financial advice.")
            default_watchlist = "HUBC, INDU, LOTCHEM, LUCK, NPL, NRL, SPSL, EFERT, OGDC, MTL"
            watchlist_raw = st.text_area("Watchlist (comma or newline separated PSX symbols)",
                                          value=default_watchlist, height=80, key="watchlist_box")
            colA, colB, colC = st.columns([1, 1, 1])
            with colA:
                scan_clicked = st.button("Scan Watchlist", type="primary", use_container_width=True)
            with colB:
                show_all_rows = st.checkbox("Show WATCH / NO SETUP rows too", value=False)
            with colC:
                include_news = st.checkbox("Include news flags (slower)", value=False)
 
            if scan_clicked:
                symbols = [s.strip().upper() for s in watchlist_raw.replace("\n", ",").split(",") if s.strip()]
                symbols = list(dict.fromkeys(symbols))  # de-dupe, keep order
                results, errors = [], []
                progress = st.progress(0.0)
                status = st.empty()
                for i, sym in enumerate(symbols):
                    status.text(f"Scanning {sym} ({i+1}/{len(symbols)})...")
                    try:
                        results.append(screen_symbol(sym, period, use_live, prefer_psx_history, check_news=include_news))
                    except Exception as exc2:
                        errors.append(f"{sym}: {exc2}")
                    progress.progress((i + 1) / len(symbols))
                    time.sleep(0.35)  # spread requests out so a burst scan doesn't trip PSX's anti-bot filter
                progress.empty()
                status.empty()
                st.session_state.screen_results = results
                st.session_state.screen_errors = errors
 
            if "screen_results" in st.session_state and st.session_state.screen_results:
                res_df = pd.DataFrame(st.session_state.screen_results)
                buys = res_df[res_df["Call"] == "BUY CALL"].sort_values("R:R", ascending=False)
                breakouts = res_df[res_df["Breakout"].isin(["BREAKOUT UP", "BREAKOUT UP (unconfirmed)"])]
 
                st.subheader(f"📈 {len(buys)} of {len(res_df)} scanned symbol(s) have an active Buy Call")
                if buys.empty:
                    st.info("No symbol in this watchlist currently meets all buy-call criteria. "
                             "Check the 'Show WATCH / NO SETUP' box below to see what's close.")
                else:
                    st.dataframe(buys.drop(columns=["Live"]), use_container_width=True, hide_index=True)
                    buy_csv = buys.to_csv(index=False).encode("utf-8")
                    st.download_button("Download buy-call list (CSV)", buy_csv, file_name="psx_buy_calls.csv", mime="text/csv")
 
                st.subheader(f"🚀 {len(breakouts)} symbol(s) are breaking out")
                if breakouts.empty:
                    st.info("No symbol in this watchlist is currently breaking out of its 20-day range.")
                else:
                    st.dataframe(breakouts.drop(columns=["Live"]), use_container_width=True, hide_index=True)
 
                if show_all_rows:
                    st.markdown("**Full scan results**")
                    st.dataframe(res_df, use_container_width=True, hide_index=True)
 
                if st.session_state.get("screen_errors"):
                    with st.expander(f"Skipped {len(st.session_state.screen_errors)} symbol(s)"):
                        for e in st.session_state.screen_errors:
                            st.caption(e)
            else:
                st.caption("Enter symbols above and click **Scan Watchlist** to run the screen.")
 
        with tab6:
            st.caption(f"Walk-forward test of the Breakout rule and a simplified Buy Call rule on {symbol.upper()}'s "
                       "own history — entries fire the day after a signal is confirmed, exits are the first of stop, "
                       "target, or a fixed holding window. Past performance on this stock's own history does not "
                       "guarantee future results.")
            with st.expander("Backtest settings"):
                bt_horizon = st.slider("Max holding period (sessions)", 3, 30, 10)
                bt_stop = st.slider("Stop distance (× ATR)", 0.5, 3.0, 1.5, 0.1)
                bt_target = st.slider("Target distance (× ATR)", 1.0, 6.0, 3.0, 0.1)
            run_bt = st.button("Run Backtest", type="primary")
            if run_bt:
                with st.spinner("Running walk-forward backtest..."):
                    bt = backtest_signals(df, horizon=bt_horizon, stop_atr=bt_stop, target_atr=bt_target)
                st.session_state.backtest_result = bt
                st.session_state.backtest_symbol = symbol.upper()
 
            if st.session_state.get("backtest_result") and st.session_state.get("backtest_symbol") == symbol.upper():
                bt = st.session_state.backtest_result
                for key, title in [("breakout", "🚀 Breakout Up"), ("buy_call", "📉 Buy Call (rolling-low proxy)")]:
                    res = bt[key]
                    st.subheader(title)
                    if not res["trades"]:
                        st.info("No historical signals found for this rule on this symbol's available history.")
                        continue
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Signals", res["trades"])
                    m2.metric("Win rate", f"{res['win_rate']:.0f}%")
                    m3.metric("Avg return", f"{res['avg_return']:+.2f}%")
                    m4.metric("Avg winner / loser", f"{res['avg_winner']:+.2f}% / {res['avg_loser']:+.2f}%")
                    m5.metric("Avg hold", f"{res['avg_hold_days']:.1f} sessions")
                    with st.expander(f"{res['trades']} individual trades"):
                        st.dataframe(pd.DataFrame(res["records"]), use_container_width=True, hide_index=True)
            else:
                st.caption("Click **Run Backtest** to test these rules against this symbol's own history.")
 
    except Exception as exc:
        st.error(f"Could not analyze {symbol.upper()}: {exc}")
        st.info("Try the exact PSX symbol, or upload a CSV containing Date, Open, High, Low, Close and Volume.")
 
 
# --------------------------------------------------------------------------------------
# Portfolio tracker — always available, independent of the symbol in the sidebar search.
# Data entered here lives only in this browser session (st.session_state); it resets on
# page reload or redeploy. Use the CSV export/import to carry it between sessions.
# --------------------------------------------------------------------------------------
 
st.divider()
with st.expander("💼 Portfolio Tracker — live P&L and stop-loss alerts for your real holdings", expanded=False):
    st.caption("Entries here only persist for this browser session — they reset on page reload or redeploy. "
               "Download the CSV after entering your holdings, and re-upload it next time to restore them. "
               "The 'Trailing Stop' is a mechanical chandelier-style suggestion (highest close since your entry "
               "date, minus 1.5×ATR) — not personalized advice, and it doesn't know about your broader plan.")
 
    if "portfolio_df" not in st.session_state:
        st.session_state.portfolio_df = pd.DataFrame([{"Symbol": "", "Quantity": 0, "Entry Price": 0.0, "Entry Date": None}])
 
    port_upload = st.file_uploader("Restore portfolio from a previously downloaded CSV", type=["csv"], key="portfolio_upload")
    if port_upload is not None:
        try:
            restored = pd.read_csv(port_upload, parse_dates=["Entry Date"])
            st.session_state.portfolio_df = restored
            st.success(f"Restored {len(restored)} holding(s) from CSV.")
        except Exception as exc:
            st.error(f"Could not read that CSV: {exc}")
 
    edited_portfolio = st.data_editor(
        st.session_state.portfolio_df, num_rows="dynamic", use_container_width=True, key="portfolio_editor",
        column_config={
            "Symbol": st.column_config.TextColumn(required=True),
            "Quantity": st.column_config.NumberColumn(min_value=0, step=1),
            "Entry Price": st.column_config.NumberColumn(min_value=0.0, format="%.2f"),
            "Entry Date": st.column_config.DateColumn(help="Optional — improves the trailing-stop calculation."),
        },
    )
    st.session_state.portfolio_df = edited_portfolio
 
    update_portfolio = st.button("Update Portfolio (fetch live prices)", type="primary")
    if update_portfolio:
        rows = edited_portfolio[(edited_portfolio["Symbol"].astype(str).str.strip() != "") &
                                 (edited_portfolio["Quantity"].fillna(0) > 0)]
        pf_results, pf_errors = [], []
        pf_progress = st.progress(0.0)
        for i, r in enumerate(rows.itertuples(index=False)):
            sym = str(r.Symbol).strip().upper()
            try:
                entry_date_val = r._3 if len(r) > 3 else None
                pf_results.append(evaluate_position(sym, float(r.Quantity), float(r._2), entry_date_val,
                                                      "2y", True, prefer_psx_history))
            except Exception as exc2:
                pf_errors.append(f"{sym}: {exc2}")
            pf_progress.progress((i + 1) / max(len(rows), 1))
            time.sleep(0.35)
        pf_progress.empty()
        st.session_state.portfolio_results = pf_results
        st.session_state.portfolio_errors = pf_errors
 
    if st.session_state.get("portfolio_results"):
        pdf = pd.DataFrame(st.session_state.portfolio_results)
        total_value = float((pdf["Current Price"] * pdf["Quantity"]).sum())
        total_cost = float((pdf["Entry Price"] * pdf["Quantity"]).sum())
        total_pl = total_value - total_cost
        total_pl_pct = (total_pl / total_cost * 100) if total_cost else 0.0
 
        pm1, pm2, pm3 = st.columns(3)
        pm1.metric("Portfolio Value", f"Rs {total_value:,.0f}")
        pm2.metric("Total Cost", f"Rs {total_cost:,.0f}")
        pm3.metric("Unrealized P&L", f"Rs {total_pl:,.0f}", f"{total_pl_pct:+.2f}%")
 
        st.dataframe(pdf.drop(columns=["Live"]), use_container_width=True, hide_index=True)
 
        breached = pdf[pdf["Stop Breached"] == "⚠ Yes"]
        if not breached.empty:
            st.warning(f"⚠ {len(breached)} position(s) have breached their suggested trailing stop: " +
                       ", ".join(breached["Symbol"]))
 
        export_csv = edited_portfolio.to_csv(index=False).encode("utf-8")
        st.download_button("Download portfolio (CSV)", export_csv, file_name="psx_portfolio.csv", mime="text/csv")
 
        if st.session_state.get("portfolio_errors"):
            with st.expander(f"Skipped {len(st.session_state.portfolio_errors)} holding(s)"):
                for e in st.session_state.portfolio_errors:
                    st.caption(e)
    else:
        st.caption("Enter your holdings above and click **Update Portfolio** to fetch live prices and P&L.")
 
