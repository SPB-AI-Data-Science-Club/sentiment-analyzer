"""
Market Sentiment Analyzer
VPS proxy + stock data layer.
Sentiment inference forwards to the necron GPU worker when available.
Stock price / news data is served directly via yfinance (always available).
"""
from datetime import datetime, timezone
from flask import Flask, render_template, request, jsonify
import requests as http
import yfinance as yf

app = Flask(__name__)

NECRON          = "http://100.72.210.90:15100"
CONNECT_TIMEOUT = 4
READ_TIMEOUT    = 90

POPULAR_TICKERS = [
    {"sym": "AAPL",  "name": "Apple"},
    {"sym": "TSLA",  "name": "Tesla"},
    {"sym": "NVDA",  "name": "NVIDIA"},
    {"sym": "MSFT",  "name": "Microsoft"},
    {"sym": "AMZN",  "name": "Amazon"},
    {"sym": "META",  "name": "Meta"},
    {"sym": "GOOGL", "name": "Alphabet"},
    {"sym": "NFLX",  "name": "Netflix"},
]

SAMPLE_HEADLINES = [
    "Apple reports record quarterly earnings, beating all analyst expectations",
    "Federal Reserve signals potential interest rate cuts next quarter",
    "Tech stocks tumble amid rising inflation fears and bond yield surge",
    "Tesla delivers fewer vehicles than expected, shares drop 8 percent",
    "New AI chip breakthrough promises 10x efficiency gains over current GPUs",
    "Oil prices stabilize as OPEC reaches production agreement",
    "Retail sales fall for third consecutive month amid consumer slowdown",
    "Microsoft Azure revenue grows 29 percent driven by cloud AI demand",
]


def _parse_news(raw_items: list) -> list:
    """Parse yfinance news items, handling both old and new API formats."""
    out = []
    for item in raw_items:
        content = item.get("content")
        if content:
            # New format (yfinance >= 0.2.x)
            click = content.get("clickThroughUrl") or content.get("canonicalUrl") or {}
            pub_dt = content.get("pubDate") or content.get("displayTime") or ""
            ts = 0
            if pub_dt:
                try:
                    ts = int(datetime.fromisoformat(pub_dt.rstrip("Z")).replace(tzinfo=timezone.utc).timestamp())
                except Exception:
                    pass
            out.append({
                "title":     content.get("title", ""),
                "publisher": (content.get("provider") or {}).get("displayName", ""),
                "link":      click.get("url", ""),
                "time":      ts,
            })
        else:
            # Old format
            out.append({
                "title":     item.get("title", ""),
                "publisher": item.get("publisher", ""),
                "link":      item.get("link", ""),
                "time":      item.get("providerPublishTime", 0),
            })
    return [n for n in out if n["title"]]


def necron_post(path: str, **kwargs):
    return http.post(
        f"{NECRON}{path}",
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        **kwargs,
    )


def gpu_unavailable():
    return jsonify({
        "error": "GPU temporarily unavailable. Please try again in a moment.",
        "gpu_offline": True,
    }), 503


# ── Routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           samples=SAMPLE_HEADLINES,
                           popular=POPULAR_TICKERS)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text) > 1000:
        return jsonify({"error": "Text too long (max 1000 characters)"}), 400
    try:
        resp = necron_post("/sentiment/analyze", json={"text": text})
        return (resp.content, resp.status_code, {"Content-Type": "application/json"})
    except http.exceptions.RequestException:
        return gpu_unavailable()


@app.route("/api/batch", methods=["POST"])
def batch_analyze():
    data = request.get_json(force=True)
    headlines = [h.strip() for h in (data.get("headlines") or []) if h.strip()][:20]
    if not headlines:
        return jsonify({"error": "No valid headlines"}), 400
    try:
        resp = necron_post("/sentiment/batch", json={"headlines": headlines})
        return (resp.content, resp.status_code, {"Content-Type": "application/json"})
    except http.exceptions.RequestException:
        return gpu_unavailable()


@app.route("/api/stock", methods=["POST"])
def stock_data():
    """Return 30-day price history + recent news for a ticker. Always works (no GPU needed)."""
    data   = request.get_json(force=True)
    ticker = (data.get("ticker") or "").strip().upper()
    if not ticker or len(ticker) > 10:
        return jsonify({"error": "Invalid ticker symbol"}), 400

    try:
        t    = yf.Ticker(ticker)
        hist = t.history(period="1mo").dropna(subset=["Close"])
        if hist.empty:
            return jsonify({"error": f"No data found for '{ticker}'. Check the ticker symbol."}), 404

        info  = t.info
        news  = (t.news or [])[:10]

        closes = [round(float(c), 2) for c in hist["Close"]]
        dates  = [d.strftime("%Y-%m-%d") for d in hist.index]
        volumes = [int(v) if v == v else 0 for v in hist["Volume"]]

        start = closes[0]
        end   = closes[-1]
        chg   = round(end - start, 2)
        pct   = round((chg / start * 100) if start else 0, 2)

        return jsonify({
            "ticker":   ticker,
            "name":     info.get("longName") or info.get("shortName") or ticker,
            "sector":   info.get("sector", ""),
            "exchange": info.get("exchange", ""),
            "prices": {
                "dates":   dates,
                "close":   closes,
                "volumes": volumes,
            },
            "stats": {
                "current":  end,
                "change":   chg,
                "pct":      pct,
                "high52":   round(float(info.get("fiftyTwoWeekHigh", 0) or 0), 2),
                "low52":    round(float(info.get("fiftyTwoWeekLow",  0) or 0), 2),
                "mcap":     info.get("marketCap"),
                "currency": info.get("currency", "USD"),
            },
            "news": _parse_news(news),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stock-sentiment", methods=["POST"])
def stock_sentiment():
    """Fetch stock data + run GPU sentiment on news headlines."""
    data   = request.get_json(force=True)
    ticker = (data.get("ticker") or "").strip().upper()
    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400

    try:
        t    = yf.Ticker(ticker)
        hist = t.history(period="1mo").dropna(subset=["Close"])
        if hist.empty:
            return jsonify({"error": f"No data found for '{ticker}'"}), 404
        info = t.info
        news = (t.news or [])[:10]
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    closes  = [round(float(c), 2) for c in hist["Close"]]
    dates   = [d.strftime("%Y-%m-%d") for d in hist.index]
    volumes = [int(v) if v == v else 0 for v in hist["Volume"]]
    start   = closes[0]
    end     = closes[-1]
    chg     = round(end - start, 2)
    pct     = round((chg / start * 100) if start else 0, 2)

    parsed_news = _parse_news(news)
    headlines   = [n["title"] for n in parsed_news if n["title"]]
    sentiment   = None
    gpu_offline = True

    if headlines:
        try:
            resp = necron_post("/sentiment/batch", json={"headlines": headlines})
            if resp.ok:
                sentiment   = resp.json()
                gpu_offline = False
        except http.exceptions.RequestException:
            pass

    return jsonify({
        "ticker":      ticker,
        "name":        info.get("longName") or info.get("shortName") or ticker,
        "sector":      info.get("sector", ""),
        "prices":      {"dates": dates, "close": closes, "volumes": volumes},
        "stats": {
            "current":  end,
            "change":   chg,
            "pct":      pct,
            "high52":   round(float(info.get("fiftyTwoWeekHigh", 0) or 0), 2),
            "low52":    round(float(info.get("fiftyTwoWeekLow",  0) or 0), 2),
            "currency": info.get("currency", "USD"),
        },
        "news":        parsed_news,
        "sentiment":   sentiment,
        "gpu_offline": gpu_offline,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5003)
