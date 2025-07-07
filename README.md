# Cryptocurrency Tweet Sentiment Classifier

## Introduction
This model classifies cryptocurrency-related tweets into `Positive`, `Negative`, or `Neutral`.

## Installation
```bash
pip install -r requirements.txt
```

## Inference Example
```bash
python inference.py
```

## Training
Ensure your CSV is in format `text,sentiment` (e.g., `Bitcoin is great!,Positive`)
```bash
python trained.py
```

## API Usage
Run:
```bash
uvicorn app:app --reload
```
Call endpoint:
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"input": "buy the dip"}'
```

## Files
- inference.py: Run prediction
- trained.py: Fine-tune model
- app.py: REST API
- weights/: Contains model weights

## Contact
Email: kunalpatra18@gmail.com
