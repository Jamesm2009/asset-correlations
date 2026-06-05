"""
Asset Correlation Dashboard
- 70 ETFs, 21-day and 5-day rolling correlations
- Data: yFinance (91 trading days of daily closes)
- Cache: Upstash Redis (refreshed nightly via cron)
- Deploy: Dokku on Digital Ocean, domain: core.market-dashboard.com
"""

from flask import Flask, render_template, jsonify, request
import requests
import pandas as pd
import numpy as np
import json
import os
import threading
import time
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

app = Flask(__name__)
CT = ZoneInfo("America/Chicago")

REDIS_URL   = os.environ.get("UPSTASH_REDIS_REST_URL", "")
REDIS_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")
REDIS_KEY   = "corr_dashboard_cache"
REDIS_KEY_STATUS = "corr_dashboard_status"

TICKERS = [
    "AIQ","CHIQ","CIBR","CPER","CWB","DBA","DBC","DXJ","EEM","EMB",
    "EMXC","EWC","EWG","EWJ","EWU","EWW","EZA","FEZ","FXE","FXI",
    "FXY","GLD","GRID","HYG","IBIT","IEF","IEI","IJH","IJJ","INDA",
    "ITA","IWD","IWF","IWM","KIE","KRE","KSPY","KWEB","MGK","MOAT",
    "MUB","PAVE","QQQ","SDY","SHY","SLV","SMH","SMIN","SPLV","SPMO",
    "SPY","TLT","USO","UUP","VNM","XHB","XHE","XLB","XLC","XLE",
    "XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY","XRT","XTN"
]
TICKERS = sorted(TICKERS)

cache = {
    "prices": None,
    "last_updated": None,
    "status": "idle",
    "error": None,
}
_lock = threading.Lock()
_started = False


# ── Redis helpers ──────────────────────────────────────────────────────────────

def redis_set(key, value, ex_seconds=90000):
    if not REDIS_URL or not REDIS_TOKEN:
        return False
    try:
        payload = json.dumps(value)
        r = requests.post(
            f"{REDIS_URL}/set/{key}",
            headers={"Authorization": f"Bearer {REDIS_TOKEN}"},
            json={"value": payload, "ex": ex_seconds},
            timeout=10
        )
        return r.status_code == 200
    except Exception as e:
        print(f"Redis SET error: {e}")
        return False


def redis_get(key):
    if not REDIS_URL or not REDIS_TOKEN:
        return None
    try:
        r = requests.get(
            f"{REDIS_URL}/get/{key}",
            headers={"Authorization": f"Bearer {REDIS_TOKEN}"},
            timeout=10
        )
        if r.status_code != 200:
            return None
        result = r.json().get("result")
        if result is None:
            return None
        return json.loads(result)
    except Exception as e:
        print(f"Redis GET error: {e}")
        return None


def save_to_redis(prices_dict, last_updated):
    payload = {
        "prices": prices_dict,
        "last_updated": last_updated,
    }
    ok = redis_set(REDIS_KEY, payload)
    print(f"Redis save: {'OK' if ok else 'FAILED'}")
    return ok


def load_from_redis():
    print("Checking Redis for cached prices...")
    payload = redis_get(REDIS_KEY)
    if not payload:
        print("No Redis cache found.")
        return None, None
    prices_dict  = payload.get("prices")
    last_updated = payload.get("last_updated")
    if prices_dict and last_updated:
        print(f"Redis restored data (updated {last_updated})")
        return prices_dict, last_updated
    return None, None


# ── Data fetch ─────────────────────────────────────────────────────────────────

def fetch_prices():
    """Download 91 trading days of daily closes for all tickers via yfinance."""
    import yfinance as yf

    # Request extra calendar days to guarantee 91 trading days
    # 91 trading days ~ 130 calendar days; request 150 to be safe
    end   = date.today()
    start = end - timedelta(days=150)

    print(f"Downloading {len(TICKERS)} tickers from {start} to {end}...")

    with _lock:
        cache["status"] = "Downloading price data..."

    try:
        raw = yf.download(
            tickers=TICKERS,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        print(f"yfinance download error: {e}")
        return None

    if raw is None or raw.empty:
        print("yfinance returned empty data")
        return None

    # Extract Close prices
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    else:
        closes = raw[["Close"]]

    # Drop weekends (yfinance should already do this, but sanitise)
    closes = closes[closes.index.dayofweek < 5]

    # Forward-fill any gaps (holidays, 3-day weekends)
    closes = closes.ffill()

    # Keep last 91 trading days
    closes = closes.tail(91)

    # Drop tickers with too many nulls (>10%)
    min_valid = int(0.90 * len(closes))
    closes = closes.dropna(axis=1, thresh=min_valid)

    # Fill remaining nulls via forward-fill then back-fill
    closes = closes.ffill().bfill()

    print(f"Price data: {len(closes)} trading days, {len(closes.columns)} tickers")
    return closes


def compute_correlation(prices_df, window):
    """Return correlation matrix for the last `window` trading days."""
    tail = prices_df.tail(window)
    return tail.corr(method="pearson")


def compute_rolling_correlation(prices_df, t1, t2, window=21):
    """Compute rolling correlation between two tickers over the full price history."""
    if t1 not in prices_df.columns or t2 not in prices_df.columns:
        return []
    s1 = prices_df[t1]
    s2 = prices_df[t2]
    rolling = s1.rolling(window).corr(s2).dropna()
    result = []
    for dt, val in rolling.items():
        result.append({
            "date": dt.strftime("%Y-%m-%d"),
            "value": round(float(val), 4) if not np.isnan(val) else None
        })
    return result


def run_update():
    global cache
    with _lock:
        cache["status"] = "Fetching..."
        cache["error"]  = None

    prices = fetch_prices()

    if prices is None or prices.empty:
        with _lock:
            cache["status"] = "error"
            cache["error"]  = "Failed to download price data."
        return

    prices_dict = {}
    for col in prices.columns:
        prices_dict[col] = {
            dt.strftime("%Y-%m-%d"): round(float(v), 6)
            for dt, v in prices[col].items()
            if not np.isnan(v)
        }

    last_updated = datetime.now(CT).strftime("%-m/%-d/%y %H:%M CT")

    with _lock:
        cache["prices"]       = prices_dict
        cache["last_updated"] = last_updated
        cache["status"]       = "ready"
        cache["error"]        = None

    save_to_redis(prices_dict, last_updated)
    print(f"Update complete — {last_updated}")


def trigger_update():
    threading.Thread(target=run_update, daemon=True).start()


def _ensure_started():
    global _started
    if not _started:
        _started = True
        prices_dict, last_updated = load_from_redis()
        if prices_dict:
            # Check if data is from today (or yesterday if weekend)
            now = datetime.now(CT)
            is_stale = True
            if last_updated:
                try:
                    # last_updated format: "M/D/YY HH:MM CT"
                    lu_date_str = last_updated.split(" ")[0]
                    lu_month, lu_day, lu_year = lu_date_str.split("/")
                    lu_date = date(2000 + int(lu_year), int(lu_month), int(lu_day))
                    # Consider fresh if from today or yesterday (for weekends)
                    delta = (date.today() - lu_date).days
                    is_stale = delta > 3
                except Exception:
                    is_stale = True

            with _lock:
                cache["prices"]       = prices_dict
                cache["last_updated"] = last_updated
                cache["status"]       = "ready"

            if is_stale:
                print("Cache is stale, refreshing...")
                trigger_update()
            else:
                print("Cache is fresh, using Redis data.")
        else:
            trigger_update()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    _ensure_started()
    with _lock:
        status       = cache["status"]
        last_updated = cache["last_updated"] or "Loading..."
        error        = cache["error"]
    return render_template("index.html",
        tickers=sorted(TICKERS),
        status=status,
        last_updated=last_updated,
        error=error,
    )


@app.route("/api/matrix")
def api_matrix():
    _ensure_started()
    window = request.args.get("window", "21", type=str)
    try:
        window_int = int(window)
        if window_int not in (5, 21):
            window_int = 21
    except Exception:
        window_int = 21

    with _lock:
        prices_dict  = cache["prices"]
        last_updated = cache["last_updated"]
        status       = cache["status"]

    if not prices_dict:
        return jsonify({"error": "Data not ready", "status": status}), 503

    prices_df = pd.DataFrame(prices_dict)
    prices_df.index = pd.to_datetime(prices_df.index)
    prices_df = prices_df.sort_index()

    corr = compute_correlation(prices_df, window_int)

    tickers_available = sorted(corr.columns.tolist())

    matrix = []
    for t1 in tickers_available:
        row = []
        for t2 in tickers_available:
            val = corr.loc[t1, t2] if (t1 in corr.index and t2 in corr.columns) else None
            row.append(round(float(val), 4) if val is not None and not np.isnan(val) else None)
        matrix.append(row)

    return jsonify({
        "tickers": tickers_available,
        "matrix": matrix,
        "window": window_int,
        "last_updated": last_updated,
    })


@app.route("/api/rolling")
def api_rolling():
    t1 = request.args.get("t1", "").upper().strip()
    t2 = request.args.get("t2", "").upper().strip()

    with _lock:
        prices_dict = cache["prices"]

    if not prices_dict:
        return jsonify({"error": "Data not ready"}), 503

    prices_df = pd.DataFrame(prices_dict)
    prices_df.index = pd.to_datetime(prices_df.index)
    prices_df = prices_df.sort_index()

    data = compute_rolling_correlation(prices_df, t1, t2, window=21)
    return jsonify({"t1": t1, "t2": t2, "data": data})


@app.route("/refresh")
def refresh():
    with _lock:
        cache["prices"]  = None
        cache["status"]  = "idle"
    trigger_update()
    return jsonify({"status": "refresh started"})


@app.route("/status")
def status_route():
    _ensure_started()
    with _lock:
        return jsonify({
            "status":       cache["status"],
            "last_updated": cache["last_updated"],
            "error":        cache["error"],
        })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
