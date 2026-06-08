"""
Market Sentiment Analyzer
Classify news headlines as Bullish / Neutral / Bearish using NLP.
"""
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Lazy-load model on first request
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline(
            "text-classification",
            model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
            top_k=None,
        )
    return _pipeline


LABEL_MAP = {
    "positive": ("Bullish",  "#4ade80", "📈"),
    "neutral":  ("Neutral",  "#fbbf24", "➡️"),
    "negative": ("Bearish",  "#f87171", "📉"),
}

SAMPLE_HEADLINES = [
    "Apple reports record quarterly earnings, beating all analyst expectations",
    "Federal Reserve signals potential interest rate cuts next quarter",
    "Tech stocks tumble amid rising inflation fears and bond yield surge",
    "Tesla delivers fewer vehicles than expected, shares drop 8%",
    "New AI chip breakthrough promises 10x efficiency gains over current GPUs",
    "Oil prices stabilize as OPEC reaches production agreement",
]


@app.route("/")
def index():
    return render_template("index.html", samples=SAMPLE_HEADLINES)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    nlp = get_pipeline()
    results = nlp(text[:512])[0]  # truncate to model max

    scores = {r["label"].lower(): r["score"] for r in results}
    top_label = max(scores, key=scores.get)
    label, color, icon = LABEL_MAP.get(top_label, ("Neutral", "#fbbf24", "➡️"))

    return jsonify({
        "label":    label,
        "icon":     icon,
        "color":    color,
        "scores": {
            "bullish": round(scores.get("positive", 0) * 100, 1),
            "neutral":  round(scores.get("neutral", 0) * 100, 1),
            "bearish":  round(scores.get("negative", 0) * 100, 1),
        },
    })


if __name__ == "__main__":
    app.run(debug=True, port=5003)
