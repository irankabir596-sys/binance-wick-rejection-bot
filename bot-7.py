import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests

# =========================
# CONFIG
# =========================
BINANCE_BASE = os.getenv("BINANCE_BASE", "https://data-api.binance.vision")
TELEGRAM_BASE = "https://api.telegram.org"
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()

TOP_COINS = 200
KLINE_LIMIT = 320
MAX_WORKERS = 16
REQUEST_TIMEOUT = 12
STATE_FILE = "state.json"

# Spot USDT pairs, ranked by Binance 24h quote volume.
EXCLUDED_BASE_ASSETS = {
    "USDT", "USDC", "FDUSD", "TUSD", "DAI", "BUSD", "USDP", "USD1", "PYUSD",
    "USTC", "EUR", "TRY", "BRL", "GBP", "RUB", "UAH", "BIDR", "IDRT", "ZAR",
}
LEVERAGED_SUFFIXES = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")
TRADINGVIEW_EXCHANGE = "BINANCE"

TIMEFRAMES = {
    "1D": "1d",
    "4H": "4h",
    "1H": "1h",
}

# D/4H/1H are checked only during the final <30 minutes of the candle.
CLOSE_WINDOW_SECONDS = 30 * 60

# Support/resistance parameters.
PIVOT_LEFT_RIGHT = 3
CLUSTER_PCT = 0.0045
MIN_PIVOT_DISTANCE_PCT = 0.0010
LEVEL_SEARCH_LIMIT = 12

# Breakout / rejection tolerances.
BREAK_CONFIRM_PCT = 0.0005
TOUCH_TOLERANCE_MIN = 0.0015
TOUCH_ATR_FACTOR = 0.12
TOUCH_TOLERANCE_MAX = 0.008

# Prevent a huge state file.
MAX_STATE_ITEMS = 2500

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "binance-sr-scanner/1.0"})


@dataclass
class Candle:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int


@dataclass
class Level:
    price: float
    kind: str  # support / resistance
    touches: int
    last_touch_index: int
    score: int


def get_json(path: str, params: Optional[dict] = None):
    url = f"{BINANCE_BASE}{path}"
    r = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_server_time_ms() -> int:
    try:
        data = get_json("/api/v3/time")
        return int(data["serverTime"])
    except Exception:
        return int(time.time() * 1000)


def get_top_symbols() -> List[str]:
    exchange = get_json("/api/v3/exchangeInfo")
    allowed = set()

    for s in exchange.get("symbols", []):
        if s.get("status") != "TRADING":
            continue
        if s.get("quoteAsset") != "USDT":
            continue
        if s.get("isSpotTradingAllowed") is False:
            continue
        symbol = s.get("symbol", "")
        base = s.get("baseAsset", "")
        if not symbol or base in EXCLUDED_BASE_ASSETS:
            continue
        if symbol.endswith(LEVERAGED_SUFFIXES):
            continue
        allowed.add(symbol)

    tickers = get_json("/api/v3/ticker/24hr")
    ranked = []
    for t in tickers:
        symbol = t.get("symbol", "")
        if symbol not in allowed:
            continue
        try:
            quote_volume = float(t.get("quoteVolume", 0))
            last_price = float(t.get("lastPrice", 0))
        except (TypeError, ValueError):
            continue
        if quote_volume <= 0 or last_price <= 0:
            continue
        ranked.append((quote_volume, symbol))

    ranked.sort(reverse=True)
    return [symbol for _, symbol in ranked[:TOP_COINS]]


def get_klines(symbol: str, interval: str, limit: int = KLINE_LIMIT) -> List[Candle]:
    raw = get_json(
        "/api/v3/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
    )
    candles = []
    for row in raw:
        candles.append(
            Candle(
                open_time=int(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                close_time=int(row[6]),
            )
        )
    return candles


def atr(candles: List[Candle], period: int = 14) -> float:
    if len(candles) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        c = candles[i]
        p = candles[i - 1]
        tr = max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close))
        trs.append(tr)
    return sum(trs[-period:]) / period


def detect_pivots(candles: List[Candle]) -> List[Tuple[int, str, float]]:
    w = PIVOT_LEFT_RIGHT
    pivots = []
    # Use completed candles only. Caller passes candles[:-1].
    for i in range(w, len(candles) - w):
        hi = candles[i].high
        lo = candles[i].low
        left = candles[i - w:i]
        right = candles[i + 1:i + w + 1]
        is_high = all(hi >= c.high for c in left + right)
        is_low = all(lo <= c.low for c in left + right)
        if is_high:
            pivots.append((i, "resistance", hi))
        if is_low:
            pivots.append((i, "support", lo))
    return pivots


def cluster_levels(candles: List[Candle]) -> List[Level]:
    if len(candles) < 2 * PIVOT_LEFT_RIGHT + 10:
        return []

    pivots = detect_pivots(candles)
    levels: List[dict] = []

    for idx, kind, price in pivots:
        matched = None
        for level in levels:
            if level["kind"] != kind:
                continue
            rel = abs(price - level["price"]) / max(level["price"], 1e-12)
            if rel <= CLUSTER_PCT:
                matched = level
                break
        if matched is None:
            levels.append({
                "price": price,
                "kind": kind,
                "touches": 1,
                "indices": [idx],
            })
        else:
            matched["price"] = (
                matched["price"] * matched["touches"] + price
            ) / (matched["touches"] + 1)
            matched["touches"] += 1
            matched["indices"].append(idx)

    result = []
    for lvl in levels:
        touches = int(lvl["touches"])
        last_touch = max(lvl["indices"])
        # Star strength = number of independent historical touches.
        score = min(5, max(1, touches - 1))
        result.append(
            Level(
                price=float(lvl["price"]),
                kind=lvl["kind"],
                touches=touches,
                last_touch_index=last_touch,
                score=score,
            )
        )

    return result


def is_timeframe_eligible(candle: Candle, now_ms: int, interval: str) -> bool:
    remaining_ms = candle.close_time - now_ms
    # Strictly less than 30 minutes and not after candle close.
    return 0 <= remaining_ms < CLOSE_WINDOW_SECONDS * 1000


def pick_relevant_level(levels: List[Level], price: float, kind: str) -> Optional[Level]:
    candidates = [l for l in levels if l.kind == kind]
    if kind == "resistance":
        candidates = [l for l in candidates if l.price >= price * (1 - CLUSTER_PCT * 2)]
    else:
        candidates = [l for l in candidates if l.price <= price * (1 + CLUSTER_PCT * 2)]
    if not candidates:
        return None

    # Prefer significance first, then closeness.
    candidates.sort(key=lambda l: (-l.score, abs(l.price - price) / max(price, 1e-12)))
    return candidates[0]


def detect_signal(candles: List[Candle], levels: List[Level]) -> Optional[Tuple[str, Level]]:
    if len(candles) < 3 or not levels:
        return None

    prev = candles[-2]
    cur = candles[-1]
    current_price = cur.close
    current_atr = atr(candles)
    tol = max(TOUCH_TOLERANCE_MIN, TOUCH_ATR_FACTOR * current_atr / max(current_price, 1e-12))
    tol = min(TOUCH_TOLERANCE_MAX, tol)

    resistances = sorted([l for l in levels if l.kind == "resistance"], key=lambda x: x.price)
    supports = sorted([l for l in levels if l.kind == "support"], key=lambda x: x.price, reverse=True)

    # 1) Confirmed breakout: previous close was at/below level and current close is above it.
    up_breaks = []
    for level in resistances:
        if prev.close <= level.price * (1 + BREAK_CONFIRM_PCT) and cur.close > level.price * (1 + BREAK_CONFIRM_PCT):
            up_breaks.append(level)

    down_breaks = []
    for level in supports:
        if prev.close >= level.price * (1 - BREAK_CONFIRM_PCT) and cur.close < level.price * (1 - BREAK_CONFIRM_PCT):
            down_breaks.append(level)

    if up_breaks:
        level = sorted(up_breaks, key=lambda l: (-l.score, abs(l.price - cur.close)))[0]
        return "BREAK_UP", level
    if down_breaks:
        level = sorted(down_breaks, key=lambda l: (-l.score, abs(l.price - cur.close)))[0]
        return "BREAK_DOWN", level

    # 2) Rejection / shadow touch.
    resistance_hits = []
    for level in resistances:
        touched = cur.high >= level.price * (1 - tol)
        rejected = cur.close < level.price * (1 - BREAK_CONFIRM_PCT)
        if touched and rejected:
            # Require an upper wick to make it a meaningful shadow rejection.
            upper_wick = cur.high - max(cur.open, cur.close)
            body = abs(cur.close - cur.open)
            if upper_wick >= max(body * 0.35, current_atr * 0.05):
                resistance_hits.append(level)

    support_hits = []
    for level in supports:
        touched = cur.low <= level.price * (1 + tol)
        rejected = cur.close > level.price * (1 + BREAK_CONFIRM_PCT)
        if touched and rejected:
            lower_wick = min(cur.open, cur.close) - cur.low
            body = abs(cur.close - cur.open)
            if lower_wick >= max(body * 0.35, current_atr * 0.05):
                support_hits.append(level)

    if resistance_hits:
        level = sorted(resistance_hits, key=lambda l: (-l.score, abs(l.price - cur.high)))[0]
        return "REJECTION_RESISTANCE", level
    if support_hits:
        level = sorted(support_hits, key=lambda l: (-l.score, abs(l.price - cur.low)))[0]
        return "REJECTION_SUPPORT", level

    return None


def format_price(price: float) -> str:
    if price >= 1000:
        return f"{price:,.2f}"
    if price >= 1:
        return f"{price:,.4f}".rstrip("0").rstrip(".")
    if price >= 0.01:
        return f"{price:.6f}".rstrip("0").rstrip(".")
    # Keep enough precision for small coins.
    return f"{price:.10f}".rstrip("0").rstrip(".")


def build_message(symbol: str, timeframe: str, signal: str, level: Level, current_price: float) -> str:
    coin = symbol.replace("USDT", "")
    stars = "⭐️" * level.score

    if signal == "BREAK_UP":
        icon = "🟢📈"
        action = "Resistance Break"
    elif signal == "BREAK_DOWN":
        icon = "🔴📉"
        action = "Support Break"
    elif signal == "REJECTION_RESISTANCE":
        icon = "🔻↩️"
        action = "Resistance Rejection"
    else:
        icon = "🔺↩️"
        action = "Support Rejection"

    # Compact English-only Telegram message.
    tv_interval = {"1D": "D", "4H": "240", "1H": "60"}.get(timeframe, "60")
    tv_symbol = f"{TRADINGVIEW_EXCHANGE}:{coin}USDT"
    encoded_symbol = tv_symbol.replace(":", "%3A")
    tv_url = f"https://www.tradingview.com/chart/?symbol={encoded_symbol}&interval={tv_interval}"
    return (
        f"{icon} <b>{coin}</b> | <b>{timeframe}</b>\n"
        f"{action}: <b>{format_price(level.price)}</b>\n"
        f"⭐️ Importance: {stars}\n"
        f"💰 Price: {format_price(current_price)}\n"
        f"📊 <a href=\"{tv_url}\">Trading view</a>"
    )


def state_key(symbol: str, timeframe: str, candle_open: int) -> str:
    # One notification maximum per candle, regardless of signal type or level.
    # This prevents the same candle from generating multiple messages across
    # repeated GitHub Actions runs.
    return f"{symbol}|{timeframe}|{candle_open}"


def load_state() -> Dict[str, int]:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_state(state: Dict[str, int]) -> None:
    # Keep only the newest N state entries.
    if len(state) > MAX_STATE_ITEMS:
        items = sorted(state.items(), key=lambda kv: kv[1], reverse=True)[:MAX_STATE_ITEMS]
        state = dict(items)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def send_telegram(text: str) -> bool:
    if not BOT_TOKEN or not CHAT_ID:
        raise RuntimeError("BOT_TOKEN or CHAT_ID is missing")
    url = f"{TELEGRAM_BASE}/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    r = SESSION.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return bool(data.get("ok"))


def fetch_one(symbol: str, timeframe: str, interval: str, now_ms: int):
    try:
        candles = get_klines(symbol, interval)
        if len(candles) < 30:
            return None

        current = candles[-1]
        if not is_timeframe_eligible(current, now_ms, interval):
            return None

        # Build levels from completed candles only, never from the live candle.
        completed = candles[:-1]
        levels = cluster_levels(completed)
        if not levels:
            return None

        signal = detect_signal(candles, levels)
        if not signal:
            return None

        signal_name, level = signal
        key = state_key(symbol, timeframe, current.open_time)
        message = build_message(symbol, timeframe, signal_name, level, current.close)
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "candle_open": current.open_time,
            "signal": signal_name,
            "level": level.price,
            "score": level.score,
            "message": message,
            "state_key": key,
        }
    except Exception as exc:
        return {"error": f"{symbol} {timeframe}: {exc}"}


def main() -> None:
    if not BOT_TOKEN or not CHAT_ID:
        raise RuntimeError("Set BOT_TOKEN and CHAT_ID in GitHub Secrets.")

    now_ms = get_server_time_ms()
    symbols = get_top_symbols()
    print(f"Top symbols: {len(symbols)}")

    jobs = []
    for timeframe, interval in TIMEFRAMES.items():
        for symbol in symbols:
            jobs.append((symbol, timeframe, interval))

    results = []
    errors = []
    # Filter D/4H/1H before making kline calls where possible using the current server clock.
    # We still need the current candle's close time, so eligibility is checked after fetch.
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_map = {
            pool.submit(fetch_one, symbol, timeframe, interval, now_ms): (symbol, timeframe)
            for symbol, timeframe, interval in jobs
        }
        for future in as_completed(future_map):
            result = future.result()
            if not result:
                continue
            if "error" in result:
                errors.append(result["error"])
            else:
                results.append(result)

    state = load_state()
    sent = 0

    # User priority: highest timeframe first, then level strength, then symbol.
    # A candle is sent only once; state_key contains only symbol/timeframe/candle.
    tf_order = {"1D": 3, "4H": 2, "1H": 1}
    results.sort(
        key=lambda x: (
            -tf_order.get(x["timeframe"], 0),
            -x["score"],
            x["symbol"],
        )
    )

    for item in results:
        key = item["state_key"]
        if key in state:
            continue
        send_telegram(item["message"])
        state[key] = int(time.time())
        sent += 1
        # Stay comfortably below Telegram's normal per-chat/global burst limits.
        time.sleep(0.08)

    save_state(state)
    print(f"Signals found: {len(results)} | New messages sent: {sent}")
    if errors:
        print(f"Errors: {len(errors)}")
        for e in errors[:20]:
            print(e)


if __name__ == "__main__":
    main()
