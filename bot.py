import os
import json
import time
import requests
from datetime import datetime, timezone

# =========================
# CONFIG
# =========================

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

BINANCE_URL = "https://data-api.binance.vision"

TOP_COINS = 200

TIMEFRAMES = {
    "1D": "1d",
    "4H": "4h",
    "1H": "1h",
    "15M": "15m",
}

LOOKBACK = 100

# نزدیک بودن قیمت به سطح
LEVEL_TOLERANCE = 0.004  # 0.4%

STATE_FILE = "state.json"

# =========================
# TELEGRAM
# =========================

def send_telegram(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=15
        )

        if response.status_code != 200:
            print("Telegram error:", response.text)

    except Exception as e:
        print("Telegram exception:", e)


# =========================
# STATE
# =========================

def load_state():
    if not os.path.exists(STATE_FILE):
        return {}

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# =========================
# BINANCE
# =========================

def get_exchange_info():
    url = f"{BINANCE_URL}/api/v3/exchangeInfo"

    response = requests.get(
        url,
        timeout=20
    )

    response.raise_for_status()

    return response.json()


def get_top_symbols():
    """
    انتخاب 200 جفت USDT برتر بر اساس
    حجم معاملات 24 ساعته
    """

    exchange_info = get_exchange_info()

    valid_symbols = set()

    for symbol in exchange_info["symbols"]:

        if symbol.get("status") != "TRADING":
            continue

        if symbol.get("quoteAsset") != "USDT":
            continue

        if symbol.get("isSpotTradingAllowed") is False:
            continue

        valid_symbols.add(symbol["symbol"])

    ticker_url = f"{BINANCE_URL}/api/v3/ticker/24hr"

    response = requests.get(
        ticker_url,
        timeout=30
    )

    response.raise_for_status()

    tickers = response.json()

    data = []

    for ticker in tickers:

        symbol = ticker.get("symbol")

        if symbol not in valid_symbols:
            continue

        try:
            volume = float(ticker.get("quoteVolume", 0))
        except:
            volume = 0

        data.append(
            (
                symbol,
                volume
            )
        )

    data.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return [
        symbol
        for symbol, _ in data[:TOP_COINS]
    ]


# =========================
# KLINES
# =========================

def get_klines(symbol, interval):
    url = f"{BINANCE_URL}/api/v3/klines"

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": LOOKBACK + 10
    }

    response = requests.get(
        url,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    return response.json()


# =========================
# TIME
# =========================

def candle_remaining_seconds(candle, interval):
    """
    زمان باقی‌مانده تا بسته شدن کندل
    """

    open_time = candle[0] / 1000

    interval_seconds = {
        "1d": 24 * 60 * 60,
        "4h": 4 * 60 * 60,
        "1h": 60 * 60,
        "15m": 15 * 60
    }[interval]

    close_time = open_time + interval_seconds

    now = time.time()

    return max(0, close_time - now)


def should_scan_timeframe(candle, interval):

    remaining = candle_remaining_seconds(
        candle,
        interval
    )

    # 15 دقیقه:
    # در تمام مدت تشکیل کندل بررسی شود
    if interval == "15m":
        return remaining > 0

    # 1H / 4H / 1D:
    # فقط در 30 دقیقه پایانی کندل
    return 0 < remaining <= 30 * 60


# =========================
# PRICE HELPERS
# =========================

def candle_data(candle):

    return {
        "open": float(candle[1]),
        "high": float(candle[2]),
        "low": float(candle[3]),
        "close": float(candle[4]),
        "volume": float(candle[5]),
        "time": int(candle[0])
    }


# =========================
# PIVOT LEVELS
# =========================

def find_resistance_levels(candles, current_index):

    levels = []

    start = 2

    # آخرین دو کندل را برای Pivot کامل کنار می‌گذاریم
    end = current_index - 2

    if end <= start:
        return levels

    for i in range(start, end):

        high = float(candles[i][2])

        left_1 = float(candles[i - 1][2])
        left_2 = float(candles[i - 2][2])

        right_1 = float(candles[i + 1][2])
        right_2 = float(candles[i + 2][2])

        if (
            high > left_1
            and high > left_2
            and high > right_1
            and high > right_2
        ):
            levels.append(high)

    return levels


def find_support_levels(candles, current_index):

    levels = []

    start = 2

    end = current_index - 2

    if end <= start:
        return levels

    for i in range(start, end):

        low = float(candles[i][3])

        left_1 = float(candles[i - 1][3])
        left_2 = float(candles[i - 2][3])

        right_1 = float(candles[i + 1][3])
        right_2 = float(candles[i + 2][3])

        if (
            low < left_1
            and low < left_2
            and low < right_1
            and low < right_2
        ):
            levels.append(low)

    return levels


# =========================
# NEAREST LEVEL
# =========================

def nearest_resistance(
    resistance_levels,
    current_high,
    current_close
):

    candidates = []

    for level in resistance_levels:

        distance = abs(current_high - level) / level

        if distance <= LEVEL_TOLERANCE:

            candidates.append(level)

    if not candidates:
        return None

    # نزدیک‌ترین مقاومت
    return min(
        candidates,
        key=lambda x: abs(current_close - x)
    )


def nearest_support(
    support_levels,
    current_low,
    current_close
):

    candidates = []

    for level in support_levels:

        distance = abs(current_low - level) / level

        if distance <= LEVEL_TOLERANCE:

            candidates.append(level)

    if not candidates:
        return None

    # نزدیک‌ترین حمایت
    return min(
        candidates,
        key=lambda x: abs(current_close - x)
    )


# =========================
# RESISTANCE REJECTION
# =========================

def detect_resistance_rejection(
    candle,
    resistance
):

    o = candle["open"]
    h = candle["high"]
    l = candle["low"]
    c = candle["close"]

    if resistance is None:
        return None

    # قیمت باید مقاومت را لمس/نفوذ کرده باشد
    if h < resistance:
        return None

    # و قیمت فعلی زیر مقاومت باشد
    if c >= resistance:
        return None

    body = abs(c - o)

    upper_wick = h - max(o, c)

    total_range = h - l

    if total_range <= 0:
        return None

    # جلوگیری از تقسیم بر صفر
    if body <= 0:
        wick_body_ratio = float("inf")
    else:
        wick_body_ratio = upper_wick / body

    wick_range_percent = (
        upper_wick / total_range
    ) * 100

    penetration_percent = (
        (h - resistance) / resistance
    ) * 100

    rejection_percent = (
        (resistance - c) / resistance
    ) * 100

    return {
        "type": "RESISTANCE",
        "level": resistance,
        "current_price": c,
        "high": h,
        "low": l,
        "open": o,
        "body": body,
        "wick": upper_wick,
        "wick_body_ratio": wick_body_ratio,
        "wick_range_percent": wick_range_percent,
        "penetration_percent": penetration_percent,
        "rejection_percent": rejection_percent
    }


# =========================
# SUPPORT REJECTION
# =========================

def detect_support_rejection(
    candle,
    support
):

    o = candle["open"]
    h = candle["high"]
    l = candle["low"]
    c = candle["close"]

    if support is None:
        return None

    # قیمت باید حمایت را لمس/نفوذ کرده باشد
    if l > support:
        return None

    # قیمت فعلی باید بالای حمایت باشد
    if c <= support:
        return None

    body = abs(c - o)

    lower_wick = min(o, c) - l

    total_range = h - l

    if total_range <= 0:
        return None

    if body <= 0:
        wick_body_ratio = float("inf")
    else:
        wick_body_ratio = lower_wick / body

    wick_range_percent = (
        lower_wick / total_range
    ) * 100

    penetration_percent = (
        (support - l) / support
    ) * 100

    rejection_percent = (
        (c - support) / support
    ) * 100

    return {
        "type": "SUPPORT",
        "level": support,
        "current_price": c,
        "high": h,
        "low": l,
        "open": o,
        "body": body,
        "wick": lower_wick,
        "wick_body_ratio": wick_body_ratio,
        "wick_range_percent": wick_range_percent,
        "penetration_percent": penetration_percent,
        "rejection_percent": rejection_percent
    }


# =========================
# FORMAT
# =========================

def format_number(value):

    if value >= 1000:
        return f"{value:,.2f}"

    if value >= 1:
        return f"{value:.4f}"

    if value >= 0.01:
        return f"{value:.6f}"

    return f"{value:.8f}"


def format_ratio(value):

    if value == float("inf"):
        return "∞"

    return f"{value:.2f}x"


def format_remaining(seconds):

    seconds = int(seconds)

    minutes = seconds // 60
    secs = seconds % 60

    if minutes >= 60:

        hours = minutes // 60
        minutes = minutes % 60

        return f"{hours}h {minutes}m"

    return f"{minutes}m {secs}s"


# =========================
# TRADINGVIEW
# =========================

def tradingview_url(symbol):

    return (
        f"https://www.tradingview.com/"
        f"symbols/{symbol}/"
        f"?exchange=BINANCE"
    )


# =========================
# MESSAGE
# =========================

def build_message(
    symbol,
    timeframe,
    signal,
    remaining
):

    if signal["type"] == "RESISTANCE":

        title = "🔴 <b>WICK REJECTION از مقاومت</b>"
        direction = "⬇️ نزولی"

    else:

        title = "🟢 <b>WICK REJECTION از حمایت</b>"
        direction = "⬆️ صعودی"

    ratio = format_ratio(
        signal["wick_body_ratio"]
    )

    tv = tradingview_url(symbol)

    message = f"""
{title}

🪙 <b>{symbol}</b>
⏱ <b>{timeframe}</b>

🎯 Level:
<b>{format_number(signal["level"])}</b>

💰 Current:
<b>{format_number(signal["current_price"])}</b>

🔺 High:
{format_number(signal["high"])}

🔻 Low:
{format_number(signal["low"])}

📏 Wick:
<b>{format_number(signal["wick"])}</b>

📐 Wick / Body:
<b>{ratio}</b>

📊 Wick / Range:
<b>{signal["wick_range_percent"]:.2f}%</b>

🎯 Penetration:
<b>{signal["penetration_percent"]:.3f}%</b>

↩️ Rejection:
<b>{signal["rejection_percent"]:.3f}%</b>

📌 Direction:
<b>{direction}</b>

⏳ Time remaining:
<b>{format_remaining(remaining)}</b>

<a href="{tv}">📊 Trading view</a>
""".strip()

    return message


# =========================
# MAIN SCANNER
# =========================

def scan():

    if not BOT_TOKEN:
        print("BOT_TOKEN is missing")
        return

    if not CHAT_ID:
        print("CHAT_ID is missing")
        return

    print("Getting top symbols...")

    try:
        symbols = get_top_symbols()
    except Exception as e:
        print("Could not get symbols:", e)
        return

    print(
        f"Scanning {len(symbols)} symbols..."
    )

    state = load_state()

    signals = []

    for symbol in symbols:

        for timeframe, interval in TIMEFRAMES.items():

            try:

                candles = get_klines(
                    symbol,
                    interval
                )

                if len(candles) < 20:
                    continue

                # آخرین کندل = کندل در حال تشکیل
                current_index = len(candles) - 1

                current_raw = candles[current_index]

                if not should_scan_timeframe(
                    current_raw,
                    interval
                ):
                    continue

                current = candle_data(
                    current_raw
                )

                remaining = candle_remaining_seconds(
                    current_raw,
                    interval
                )

                # =========================
                # LEVELS
                # =========================

                resistance_levels = (
                    find_resistance_levels(
                        candles,
                        current_index
                    )
                )

                support_levels = (
                    find_support_levels(
                        candles,
                        current_index
                    )
                )

                # =========================
                # RESISTANCE
                # =========================

                resistance = nearest_resistance(
                    resistance_levels,
                    current["high"],
                    current["close"]
                )

                resistance_signal = (
                    detect_resistance_rejection(
                        current,
                        resistance
                    )
                )

                if resistance_signal:

                    candle_id = current["time"]

                    state_key = (
                        f"{symbol}_"
                        f"{timeframe}_"
                        f"RESISTANCE_"
                        f"{candle_id}"
                    )

                    if state_key not in state:

                        signals.append({
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "signal": resistance_signal,
                            "remaining": remaining,
                            "state_key": state_key
                        })

                # =========================
                # SUPPORT
                # =========================

                support = nearest_support(
                    support_levels,
                    current["low"],
                    current["close"]
                )

                support_signal = (
                    detect_support_rejection(
                        current,
                        support
                    )
                )

                if support_signal:

                    candle_id = current["time"]

                    state_key = (
                        f"{symbol}_"
                        f"{timeframe}_"
                        f"SUPPORT_"
                        f"{candle_id}"
                    )

                    if state_key not in state:

                        signals.append({
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "signal": support_signal,
                            "remaining": remaining,
                            "state_key": state_key
                        })

            except Exception as e:

                print(
                    f"Error {symbol} "
                    f"{timeframe}: {e}"
                )

    # =========================
    # SORT
    # =========================

    timeframe_order = {
        "1D": 0,
        "4H": 1,
        "1H": 2,
        "15M": 3
    }

    signals.sort(
        key=lambda x: (
            timeframe_order.get(
                x["timeframe"],
                99
            ),
            x["symbol"]
        )
    )

    # =========================
    # SEND
    # =========================

    print(
        f"Found {len(signals)} new signals."
    )

    for item in signals:

        message = build_message(
            item["symbol"],
            item["timeframe"],
            item["signal"],
            item["remaining"]
        )

        print(
            "Sending:",
            item["symbol"],
            item["timeframe"],
            item["signal"]["type"]
        )

        send_telegram(message)

        state[item["state_key"]] = {
            "sent_at": datetime.now(
                timezone.utc
            ).isoformat()
        }

        # کمی فاصله بین پیام‌ها
        time.sleep(0.3)

    save_state(state)


# =========================
# RUN
# =========================

if __name__ == "__main__":
    scan()
