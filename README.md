# Market Sentiment Analyzer

NLP tool that classifies financial news headlines as **Bullish**, **Neutral**, or **Bearish** using a fine-tuned DistilRoBERTa model from HuggingFace.

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
# → http://localhost:5003
```

The model (~300 MB) downloads automatically on first run.

## Running on necron for faster inference

```bash
# The RTX 5080 cuts inference time dramatically for batch analysis
# In app.py, the pipeline auto-detects CUDA if available
ssh necron   # via Tailscale
cd ~/projects/sentiment-analyzer
source .venv/bin/activate && python app.py
```

## Tech Stack

Python · Flask · HuggingFace Transformers · DistilRoBERTa
