import os
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

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
    "positive": ("Bullish",  "#4ade80"),
    "neutral":  ("Neutral",  "#fbbf24"),
    "negative": ("Bearish",  "#f87171"),
}

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


@app.route("/")
def index():
    return render_template("index.html", samples=SAMPLE_HEADLINES)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text) > 1000:
        return jsonify({"error": "Text too long (max 1000 characters)"}), 400

    try:
        nlp     = get_pipeline()
        results = nlp(text[:512])[0]
    except Exception as e:
        return jsonify({"error": f"Model error: {e}"}), 500

    scores = {r["label"].lower(): round(r["score"] * 100, 1) for r in results}
    top    = max(scores, key=scores.get)
    label, color = LABEL_MAP.get(top, ("Neutral", "#fbbf24"))

    return jsonify({
        "label": label,
        "color": color,
        "scores": {
            "Bullish": scores.get("positive", 0),
            "Neutral":  scores.get("neutral",  0),
            "Bearish":  scores.get("negative", 0),
        },
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5003)
