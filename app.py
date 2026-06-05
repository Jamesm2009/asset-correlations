"""
Asset Correlation Dashboard
- 44 ETFs, 21-day and 5-day rolling correlations
- Data: yFinance (91 trading days of daily closes)
- Cache: Upstash Redis (refreshed nightly via cron)
- Deploy: Dokku on Digital Ocean, domain: corr.market-dashboards.com
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
REDIS_KEY   = "corr_dashboard_v2"

# 44 ETFs from the ETF Volume dashboard
TICKERS = sorted([
    "AIQ","CHIQ","CIBR","CPER","CWB","DBA","DBC","DXJ","EEM","EMB",
    "EMXC","EWJ","FEZ","FXI","GLD","HYG","IBIT","IEF","IJH","INDA",
    "ITA","IWD","IWF","IWM","KRE","KWEB","MGK","MOAT","MUB","PAVE",
    "QQQ","SDY","SHY","SLV","SMH","SPY","TLT","USO","UUP","XLB",
    "XLC","XLE","XLF","XLI","XLK","XLP","XLRE","XLU","XLV","XLY",
])
# Trim to 44 -- remove 6 least-interesting for a cleaner matrix
TICKERS = sorted([
    "CHIQ","CIBR","CPER","DBA","DBC","EEM","EMB","FXI","GLD","HYG",
    "IBIT","IEF","IJH","INDA","ITA","IWD","IWF","IWM","KRE","KWEB",
    "MGK","MOAT","MUB","PAVE","QQQ","SDY","SHY","SLV","SMH","SPY",
    "TLT","USO","UUP","XLB","XLC","XLE","XLF","XLI","XLK","XLP",
    "XLRE","XLU","XLV","XLY",
])

cache = {
    "prices": None,
    "last_updated": None,
    "status": "idle",
    "error": None,
}
_lock    = threading.Lock()
_started = False


# ── Redis helpers ──────────────────────────────────────────────────────────────

def redis_set(key, value, ex_seconds=90000):
    if not REDIS_URL or not REDIS_TOKEN:
        return False
    try:
        # Upstash REST API: POST /set/KEY with raw string body
        encoded = json.dumps(value)
        r = requests.post(
            f"{REDIS_URL}/set/{key}",
            headers={
                "Authorization": f"Bearer {REDIS_TOKEN}",
                "Content-Type":  "application/json",
            },
            # Upstash expects: ["SET", key, value, "EX", seconds]
            # Via REST pipeline or direct set endpoint with query params
            params={"ex": ex_seconds},
            data=encoded,
            timeout=15,
        )
        print(f"Redis SET status: {r.status_code} {r.text[:80]}")
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
            timeout=15,
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


def redis_del(key):
    if not REDIS_URL or not REDIS_TOKEN:
        return
    try:
        requests.post(
            f"{REDIS_URL}/del/{key}",
            headers={"Authorization": f"Bearer {REDIS_TOKEN}"},
            timeout=10,
        )
    except Exception:
        pass


def save_to_redis(prices_dict, last_updated):
    payload = {
        "prices":       prices_dict,
        "last_updated": last_updated,
    }
    serialised = json.dumps(payload)
    ok = redis_set(REDIS_KEY, serialised)
    print(f"Redis save: {'OK' if ok else 'FAILED'} ({len(serialised)//1024} KB)")
    return ok


def load_from_redis():
    print("Checking Redis for cached prices...")
    raw = redis_get(REDIS_KEY)
    if not raw:
        print("No Redis cache found.")
        return None, None
    # raw may be a dict (already parsed) or a JSON string (double-encoded)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            print("Redis data malformed.")
            return None, None
    prices_dict  = raw.get("prices")
    last_updated = raw.get("last_updated")
    if prices_dict and last_updated:
        print(f"Redis restored {len(prices_dict)} tickers (updated {last_updated})")
        return prices_dict, last_updated
    return None, None


# ── Data fetch ─────────────────────────────────────────────────────────────────

def fetch_prices():
    import yfinance as yf

    end   = date.today()
    start = end - timedelta(days=150)  # ~91 trading days + buffer

    print(f"Downloading {len(TICKERS)} tickers {start} -> {end}...")
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
        print(f"yfinance error: {e}")
        return None

    if raw is None or raw.empty:
        return None

    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]

    # Sanitise: weekdays only, forward-fill gaps (holidays, long weekends)
    closes = closes[closes.index.dayofweek < 5]
    closes = closes.ffill()
    closes = closes.tail(91)

    # Drop tickers with >10% missing
    closes = closes.dropna(axis=1, thresh=int(0.90 * len(closes)))
    closes = closes.ffill().bfill()

    print(f"Prices ready: {len(closes)} days x {len(closes.columns)} tickers")
    return closes


def compute_correlation(prices_df, window):
    return prices_df.tail(window).corr(method="pearson")


def compute_rolling_corr(prices_df, t1, t2, window=21):
    if t1 not in prices_df.columns or t2 not in prices_df.columns:
        return []
    rolling = prices_df[t1].rolling(window).corr(prices_df[t2]).dropna()
    return [
        {"date": dt.strftime("%Y-%m-%d"), "value": round(float(v), 4)}
        for dt, v in rolling.items()
        if not np.isnan(v)
    ]


def run_update():
    with _lock:
        cache["status"] = "Fetching..."
        cache["error"]  = None

    prices = fetch_prices()

    if prices is None or prices.empty:
        with _lock:
            cache["status"] = "error"
            cache["error"]  = "Failed to download price data."
        return

    prices_dict = {
        col: {
            dt.strftime("%Y-%m-%d"): round(float(v), 6)
            for dt, v in prices[col].items()
            if not np.isnan(v)
        }
        for col in prices.columns
    }
    last_updated = datetime.now(CT).strftime("%-m/%-d/%y %H:%M CT")

    with _lock:
        cache["prices"]       = prices_dict
        cache["last_updated"] = last_updated
        cache["status"]       = "ready"
        cache["error"]        = None

    save_to_redis(prices_dict, last_updated)
    print(f"Update complete {last_updated}")


def trigger_update():
    threading.Thread(target=run_update, daemon=True).start()


def _ensure_started():
    global _started
    if _started:
        return
    _started = True

    prices_dict, last_updated = load_from_redis()
    if prices_dict:
        # Check staleness: refresh if data is >3 days old
        stale = True
        try:
            parts   = last_updated.split(" ")[0].split("/")
            lu_date = date(2000 + int(parts[2]), int(parts[0]), int(parts[1]))
            stale   = (date.today() - lu_date).days > 3
        except Exception:
            pass

        with _lock:
            cache["prices"]       = prices_dict
            cache["last_updated"] = last_updated
            cache["status"]       = "ready"

        if stale:
            print("Cache stale — refreshing in background...")
            trigger_update()
        else:
            print("Cache fresh — serving from Redis.")
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
        tickers=TICKERS,
        status=status,
        last_updated=last_updated,
        error=error,
    )


@app.route("/api/matrix")
def api_matrix():
    _ensure_started()
    window = request.args.get("window", 21, type=int)
    if window not in (5, 21):
        window = 21

    with _lock:
        prices_dict  = cache["prices"]
        last_updated = cache["last_updated"]
        status       = cache["status"]

    if not prices_dict:
        return jsonify({"error": "Data not ready", "status": status}), 503

    prices_df = pd.DataFrame(prices_dict)
    prices_df.index = pd.to_datetime(prices_df.index)
    prices_df = prices_df.sort_index()

    corr    = compute_correlation(prices_df, window)
    tickers = sorted(corr.columns.tolist())
    matrix  = [
        [
            round(float(corr.loc[t1, t2]), 4)
            if (t1 in corr.index and t2 in corr.columns and not np.isnan(corr.loc[t1, t2]))
            else None
            for t2 in tickers
        ]
        for t1 in tickers
    ]

    return jsonify({
        "tickers":      tickers,
        "matrix":       matrix,
        "window":       window,
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

    data = compute_rolling_corr(prices_df, t1, t2)
    return jsonify({"t1": t1, "t2": t2, "data": data})


@app.route("/refresh")
def refresh():
    redis_del(REDIS_KEY)
    with _lock:
        cache["prices"] = None
        cache["status"] = "idle"
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
