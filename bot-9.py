import json
import math
import os
import time
from urllib.parse import urlencode
from statistics import median
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
# Public GitHub Pages URL for chart.html. Override with CHART_BASE_URL when needed.
# In GitHub Actions, the default is derived from GITHUB_REPOSITORY.
CHART_BASE_URL = os.getenv("CHART_BASE_URL", "").strip()

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
    touch_indices: Tuple[int, ...] = ()
    touch_prices: Tuple[float, ...] = ()


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


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def rolling_median(values: List[float]) -> float:
    clean = [float(v) for v in values if v is not None and v > 0]
    return median(clean) if clean else 0.0


def local_atr(candles: List[Candle], index: int, period: int = 14) -> float:
    start = max(1, index - period + 1)
    trs = []
    for i in range(start, index + 1):
        c = candles[i]
        p = candles[i - 1]
        trs.append(max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close)))
    return sum(trs) / len(trs) if trs else 0.0


def independent_touch_count(indices: List[int]) -> int:
    if not indices:
        return 0
    count = 1
    last = indices[0]
    for idx in indices[1:]:
        if idx - last >= 3:
            count += 1
            last = idx
    return count


def pivot_reaction_strength(kind: str, candles: List[Candle], index: int, lookahead: int = 8) -> float:
    if index >= len(candles) - 1:
        return 0.0
    level_price = candles[index].high if kind == "resistance" else candles[index].low
    end = min(len(candles), index + 1 + lookahead)
    future = candles[index + 1:end]
    if not future:
        return 0.0
    if kind == "resistance":
        favorable = level_price - min(c.low for c in future)
    else:
        favorable = max(c.high for c in future) - level_price
    a = local_atr(candles, index)
    return favorable / a if a > 0 else 0.0


def swing_prominence(kind: str, candles: List[Candle], index: int, radius: int = 5) -> float:
    pivot = candles[index].high if kind == "resistance" else candles[index].low
    left = candles[max(0, index - radius):index]
    right = candles[index + 1:min(len(candles), index + radius + 1)]
    if not left or not right:
        return 0.0
    if kind == "resistance":
        neighbor_low = min([c.low for c in left + right])
        move = pivot - neighbor_low
    else:
        neighbor_high = max([c.high for c in left + right])
        move = neighbor_high - pivot
    a = local_atr(candles, index)
    return move / a if a > 0 else 0.0


def false_break_count(level: "Level", candles: List[Candle], zone_pct: float) -> int:
    """Count historical level sweeps that broke the zone and quickly recovered.

    A sweep is treated as a false break when price crosses beyond the zone and
    returns back inside it within the next few completed candles.
    """
    p = level.price
    zone = max(zone_pct, 0.0025)
    count = 0
    for idx in level.touch_indices:
        start = idx + 1
        end = min(len(candles), idx + 6)
        if start >= end:
            continue
        window = candles[start:end]
        if level.kind == "resistance":
            broke = any(c.high > p * (1 + zone) for c in window)
            recovered = any(c.close < p for c in window if c.high > p * (1 + zone))
        else:
            broke = any(c.low < p * (1 - zone) for c in window)
            recovered = any(c.close > p for c in window if c.low < p * (1 - zone))
        if broke and recovered:
            count += 1
    return count


def build_base_level_score(lvl: dict, candles: List[Candle]) -> int:
    """Structural 0-100 score used to choose between candidate levels.

    This is deliberately based only on completed historical candles. The final
    signal score is calculated later after the actual signal candle and the
    higher-timeframe context are known.
    """
    indices = sorted(set(lvl["indices"]))
    touches = len(indices)
    n = max(len(candles), 1)

    # 15: independent historical touches.
    independent = independent_touch_count(indices)
    touch_score = min(1.0, independent / 5.0) * 15.0

    # 15: strong reactions away from the zone.
    reaction_ratios = [pivot_reaction_strength(lvl["kind"], candles, i) for i in indices]
    reaction_ratios = [r for r in reaction_ratios if r >= 0]
    avg_reaction = sum(reaction_ratios) / len(reaction_ratios) if reaction_ratios else 0.0
    reaction_score = clamp(avg_reaction / 1.6, 0.0, 1.0) * 15.0

    # 10: volume quality at historical touches.
    vr = []
    for idx in indices:
        baseline = rolling_median([c.volume for c in candles[max(0, idx-20):idx] if c.volume > 0])
        if baseline > 0 and candles[idx].volume > 0:
            vr.append(candles[idx].volume / baseline)
    avg_vr = sum(vr) / len(vr) if vr else 1.0
    volume_score = clamp((avg_vr - 0.70) / 1.30, 0.0, 1.0) * 10.0

    # 10: age/freshness.
    age = max(0, n - 1 - max(indices))
    freshness_score = clamp(1.0 - age / max(n * 0.65, 1.0), 0.0, 1.0) * 10.0

    # 10: zone tightness / coherence.
    prices = lvl.get("prices", [lvl["price"]])
    center = max(abs(lvl["price"]), 1e-12)
    dispersion = max(abs(p - lvl["price"]) / center for p in prices) if prices else 0.0
    coherence = clamp(1.0 - dispersion / max(CLUSTER_PCT, 1e-12), 0.0, 1.0)
    coherence_score = coherence * 10.0

    # 5: how well the pivot dominates nearby swings.
    swing_ratios = [swing_prominence(lvl["kind"], candles, i) for i in indices]
    swing = max(swing_ratios) if swing_ratios else 0.0
    swing_score = clamp(swing / 2.0, 0.0, 1.0) * 5.0

    # 5: historical distribution; repeated touches spread across time are better.
    if len(indices) >= 2:
        span = indices[-1] - indices[0]
        spread = clamp(span / max(n * 0.55, 1.0), 0.0, 1.0)
    else:
        spread = 0.0
    spread_score = spread * 5.0

    return int(round(clamp(
        touch_score + reaction_score + volume_score + freshness_score +
        coherence_score + swing_score + spread_score, 0.0, 100.0
    )))


def level_score(lvl: dict, candles: List[Candle]) -> int:
    return build_base_level_score(lvl, candles)


def higher_timeframe_alignment(
    level: Level,
    htf_candles: Optional[List[Candle]],
) -> float:
    """Return a 0-15 confirmation score from the next higher timeframe."""
    if not htf_candles or len(htf_candles) < 30:
        return 0.0

    htf = htf_candles[:-1]
    htf_levels = cluster_levels(htf)
    candidates = [l for l in htf_levels if l.kind == level.kind]
    if not candidates:
        return 0.0

    rels = [(abs(l.price - level.price) / max(level.price, 1e-12), l) for l in candidates]
    rels.sort(key=lambda x: x[0])
    rel, best = rels[0]
    if rel > 0.012:
        return 0.0
    if rel <= 0.003 and best.score >= 75:
        return 15.0
    if rel <= 0.005 and best.score >= 65:
        return 12.0
    if rel <= 0.008 and best.score >= 50:
        return 8.0
    if rel <= 0.012:
        return 4.0
    return 0.0


def signal_candle_score(
    signal: str,
    level: Level,
    prev: Candle,
    cur: Candle,
    current_atr: float,
    baseline_volume: float,
) -> float:
    """Score only the current signal candle, max 10 points."""
    rng = max(cur.high - cur.low, 1e-12)
    body = abs(cur.close - cur.open)
    body_ratio = clamp(body / rng, 0.0, 1.0)
    vol_ratio = cur.volume / baseline_volume if baseline_volume > 0 else 1.0

    if signal == "BREAK_UP":
        close_location = clamp((cur.close - cur.low) / rng, 0.0, 1.0)
        margin_atr = (cur.close - level.price) / current_atr if current_atr > 0 else 0.0
        score = 4.0 * clamp(body_ratio / 0.60, 0.0, 1.0)
        score += 2.5 * clamp((vol_ratio - 0.75) / 1.25, 0.0, 1.0)
        score += 2.0 * clamp((close_location - 0.60) / 0.40, 0.0, 1.0)
        score += 1.5 * clamp(margin_atr / 0.75, 0.0, 1.0)
    elif signal == "BREAK_DOWN":
        close_location = clamp((cur.high - cur.close) / rng, 0.0, 1.0)
        margin_atr = (level.price - cur.close) / current_atr if current_atr > 0 else 0.0
        score = 4.0 * clamp(body_ratio / 0.60, 0.0, 1.0)
        score += 2.5 * clamp((vol_ratio - 0.75) / 1.25, 0.0, 1.0)
        score += 2.0 * clamp((close_location - 0.60) / 0.40, 0.0, 1.0)
        score += 1.5 * clamp(margin_atr / 0.75, 0.0, 1.0)
    elif signal == "REJECTION_RESISTANCE":
        wick = cur.high - max(cur.open, cur.close)
        wick_ratio = clamp(wick / rng, 0.0, 1.0)
        close_away = clamp((level.price - cur.close) / max(current_atr, 1e-12), 0.0, 1.5)
        score = 4.0 * clamp(wick_ratio / 0.45, 0.0, 1.0)
        score += 2.5 * clamp((vol_ratio - 0.70) / 1.30, 0.0, 1.0)
        score += 2.0 * clamp(close_away / 0.75, 0.0, 1.0)
        score += 1.5 * clamp((body_ratio - 0.10) / 0.55, 0.0, 1.0)
    else:  # REJECTION_SUPPORT
        wick = min(cur.open, cur.close) - cur.low
        wick_ratio = clamp(wick / rng, 0.0, 1.0)
        close_away = clamp((cur.close - level.price) / max(current_atr, 1e-12), 0.0, 1.5)
        score = 4.0 * clamp(wick_ratio / 0.45, 0.0, 1.0)
        score += 2.5 * clamp((vol_ratio - 0.70) / 1.30, 0.0, 1.0)
        score += 2.0 * clamp(close_away / 0.75, 0.0, 1.0)
        score += 1.5 * clamp((body_ratio - 0.10) / 0.55, 0.0, 1.0)

    # Penalize a signal candle that dramatically expanded against the level.
    if signal.startswith("BREAK") and prev.close != 0:
        gap_extension = abs(cur.close - level.price) / max(abs(level.price), 1e-12)
        if gap_extension > 0.018:
            score -= 1.5
    return clamp(score, 0.0, 10.0)


def final_level_score(
    level: Level,
    candles: List[Candle],
    current: Candle,
    prev: Candle,
    signal: str,
    htf_candles: Optional[List[Candle]],
) -> int:
    """High-strength 0-100 signal score. Does not filter any signals."""
    indices = list(level.touch_indices)
    if not indices:
        indices = [level.last_touch_index]

    # 15: independent tests of the zone.
    independent = independent_touch_count(sorted(indices))
    touches_score = min(15.0, independent / 5.0 * 15.0)

    # 15: average + best historical reaction strength, ATR-normalized.
    ratios = [pivot_reaction_strength(level.kind, candles, i) for i in indices]
    avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    best_ratio = max(ratios) if ratios else 0.0
    reaction_metric = 0.65 * avg_ratio + 0.35 * best_ratio
    reaction_score = 15.0 * clamp(reaction_metric / 1.75, 0.0, 1.0)

    # 10: historical touch volume quality.
    volume_ratios = []
    for idx in indices:
        base = rolling_median([c.volume for c in candles[max(0, idx - 20):idx] if c.volume > 0])
        if base > 0 and candles[idx].volume > 0:
            volume_ratios.append(candles[idx].volume / base)
    avg_volume_ratio = sum(volume_ratios) / len(volume_ratios) if volume_ratios else 1.0
    volume_score = 10.0 * clamp((avg_volume_ratio - 0.70) / 1.30, 0.0, 1.0)

    # 15: higher timeframe alignment.
    htf_score = higher_timeframe_alignment(level, htf_candles)

    # 10: freshness + recurrence balance. A very old level loses a little,
    # but a well-tested old level is not discarded.
    n = max(len(candles), 1)
    age = max(0, n - 1 - max(indices))
    freshness = clamp(1.0 - age / max(n * 0.70, 1.0), 0.0, 1.0)
    recurrence_bonus = clamp((independent - 1) / 4.0, 0.0, 1.0) * 0.25
    freshness_score = 10.0 * clamp(0.80 * freshness + 0.20 * recurrence_bonus, 0.0, 1.0)

    # 10: zone coherence.
    prices = list(level.touch_prices) or [level.price]
    center = max(abs(level.price), 1e-12)
    dispersion = max(abs(p - level.price) / center for p in prices)
    coherence_score = 10.0 * clamp(1.0 - dispersion / max(CLUSTER_PCT, 1e-12), 0.0, 1.0)

    # 5: current distance/extension relative to ATR.
    current_atr = atr(candles)
    if signal == "BREAK_UP":
        distance = max(0.0, current.close - level.price)
    elif signal == "BREAK_DOWN":
        distance = max(0.0, level.price - current.close)
    elif signal == "REJECTION_RESISTANCE":
        distance = abs(current.high - level.price)
    else:
        distance = abs(level.price - current.low)
    distance_atr = distance / current_atr if current_atr > 0 else 0.0
    proximity_score = 5.0 * clamp(1.0 - distance_atr / 1.50, 0.0, 1.0)

    # 5: swing prominence of the strongest historical touch.
    swing_ratios = [swing_prominence(level.kind, candles, i) for i in indices]
    swing_score = 5.0 * clamp((max(swing_ratios) if swing_ratios else 0.0) / 2.2, 0.0, 1.0)

    # 5: previous failed-break behavior. For breakouts, clean history is
    # preferred. For rejections, prior successful sweeps add confirmation.
    fb = false_break_count(level, candles, max(TOUCH_TOLERANCE_MIN, TOUCH_ATR_FACTOR * current_atr / max(current.close, 1e-12)))
    if signal.startswith("BREAK"):
        false_break_score = 5.0 if fb == 0 else (3.0 if fb == 1 else (1.5 if fb == 2 else 0.0))
    else:
        false_break_score = min(5.0, 1.5 + fb * 1.75) if fb > 0 else 1.0

    # 10: quality of the actual signal candle.
    baseline_volume = rolling_median([c.volume for c in candles[-21:-1] if c.volume > 0])
    signal_score = signal_candle_score(signal, level, prev, current, current_atr, baseline_volume)

    total = (
        touches_score + reaction_score + volume_score + htf_score +
        freshness_score + coherence_score + proximity_score +
        swing_score + false_break_score + signal_score
    )
    return int(round(clamp(total, 0.0, 100.0)))


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
                "prices": [price],
            })
        else:
            matched["price"] = (
                matched["price"] * matched["touches"] + price
            ) / (matched["touches"] + 1)
            matched["touches"] += 1
            matched["indices"].append(idx)
            matched["prices"].append(price)

    result = []
    for lvl in levels:
        touches = int(lvl["touches"])
        last_touch = max(lvl["indices"])
        score = level_score(lvl, candles)
        result.append(
            Level(
                price=float(lvl["price"]),
                kind=lvl["kind"],
                touches=touches,
                last_touch_index=last_touch,
                score=score,
                touch_indices=tuple(sorted(lvl["indices"])),
                touch_prices=tuple(float(x) for x in lvl["prices"]),
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


def score_label(score: int) -> str:
    if score >= 90:
        return "🔥 Very Strong"
    if score >= 80:
        return "🟢 Strong"
    if score >= 70:
        return "🟡 Important"
    if score >= 60:
        return "⚪ Moderate"
    return "⚫ Low"


def get_chart_base_url() -> str:
    """Resolve the public chart.html URL used by Telegram links.

    CHART_BASE_URL can be set explicitly, for example:
    https://your-user.github.io/your-repo/chart.html

    When running in GitHub Actions, GITHUB_REPOSITORY provides owner/repo,
    so a project-pages URL can be derived automatically.
    """
    if CHART_BASE_URL:
        return CHART_BASE_URL.rstrip("/")

    repo = os.getenv("GITHUB_REPOSITORY", "").strip()
    if "/" in repo:
        owner, name = repo.split("/", 1)
        return f"https://{owner}.github.io/{name}/chart.html"

    # Fallback for local/manual runs. This should be overridden before use.
    return "https://YOUR-USERNAME.github.io/YOUR-REPO/chart.html"


def build_message(symbol: str, timeframe: str, signal: str, level: Level, current_price: float) -> str:
    coin = symbol.replace("USDT", "")
    if signal == "BREAK_UP":
        icon = "🟢📈"
        action = "Resistance Break"
        level_kind = "resistance"
    elif signal == "BREAK_DOWN":
        icon = "🔴📉"
        action = "Support Break"
        level_kind = "support"
    elif signal == "REJECTION_RESISTANCE":
        icon = "🔻↩️"
        action = "Resistance Rejection"
        level_kind = "resistance"
    else:
        icon = "🔺↩️"
        action = "Support Rejection"
        level_kind = "support"

    tv_interval = {"1D": "D", "4H": "240", "1H": "60"}.get(timeframe, "60")
    tv_symbol = f"{TRADINGVIEW_EXCHANGE}:{coin}USDT"
    encoded_symbol = tv_symbol.replace(":", "%3A")
    tv_url = f"https://www.tradingview.com/chart/?symbol={encoded_symbol}&interval={tv_interval}"

    # Open our TradingView Lightweight Charts page with the detected S/R zone
    # drawn directly on the chart. The page also contains a button to open the
    # same symbol/timeframe on TradingView.com.
    chart_params = urlencode({
        "symbol": symbol,
        "timeframe": timeframe,
        "level": f"{level.price:.12g}",
        "kind": level_kind,
        "signal": signal,
        "score": str(level.score),
        "price": f"{current_price:.12g}",
    })
    chart_url = f"{get_chart_base_url()}?{chart_params}"

    return (
        f"{icon} <b>{coin}</b> | <b>{timeframe}</b>\n"
        f"{action}: <b>{format_price(level.price)}</b>\n"
        f"🎯 Importance: <b>{level.score}/100</b> {score_label(level.score)}\n"
        f"💰 Price: {format_price(current_price)}\n"
        f"📊 <a href=\"{chart_url}\">Trading view</a>"
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
        prev = candles[-2]

        # Fetch only the next higher timeframe when a signal candidate already
        # exists. This keeps the normal scan cost reasonable while adding true
        # multi-timeframe structural confirmation to the final score.
        htf_interval = {"1H": "4h", "4H": "1d", "1D": "1w"}.get(timeframe)
        htf_candles = None
        if htf_interval:
            try:
                htf_candles = get_klines(symbol, htf_interval, limit=220)
            except Exception as htf_exc:
                print(f"HTF scoring data unavailable for {symbol} {timeframe}: {htf_exc}")

        final_score = final_level_score(
            level=level,
            candles=completed,
            current=current,
            prev=prev,
            signal=signal_name,
            htf_candles=htf_candles,
        )
        level.score = final_score

        key = state_key(symbol, timeframe, current.open_time)
        message = build_message(symbol, timeframe, signal_name, level, current.close)
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "candle_open": current.open_time,
            "signal": signal_name,
            "level": level.price,
            "score": final_score,
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
