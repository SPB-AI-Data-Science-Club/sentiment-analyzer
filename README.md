# Market Sentiment Analyzer

Live stock data paired with transformer-scored news sentiment.

**Live demo:** [sentiment.spbdatascience.org](https://sentiment.spbdatascience.org)

## Features

- Stock tracker: 30-day price chart, company stats, and 52-week range for any ticker
- Recent headlines scored bullish / neutral / bearish by DistilRoBERTa fine-tuned on financial news
- Aggregate sentiment breakdown per stock
- Single-headline and batch analysis modes
- Degrades gracefully: price data and news always work even when the GPU worker is offline

## Architecture

The web app runs on a CPU VPS and never loads the model itself. Inference requests are proxied to a GPU worker over a private Tailscale network; stock prices and news come from Yahoo Finance via `yfinance` directly on the VPS. The split keeps the public-facing app light while the club GPU server does the heavy lifting.

```
browser -> Flask (VPS) -> GPU worker (club server, Tailscale)
                  \-> yfinance (prices + news)
```

## Stack

Python, Flask, Hugging Face Transformers (DistilRoBERTa), yfinance, Plotly

## Local development

```bash
pip install flask requests yfinance
python app.py
```

Point `NECRON` at your own inference endpoint or rely on the built-in offline fallback.
