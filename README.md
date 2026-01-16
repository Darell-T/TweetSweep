# TweetSweep

AI-powered content moderation tool using multi-label classification to detect toxic, hate speech, and profanity in text. Trained on DeBERTa-v3-small with ONNX optimization for fast inference.

## Features

- **Multi-label Classification**: Detects three types of toxic content:
  - Hate Speech
  - Toxicity
  - Profanity
- **Optimized Inference**: 2x faster inference using ONNX Runtime (64ms → 32ms on CPU)
- **Interactive Testing**: Command-line tool to test any text for toxicity

## Tech Stack

**Machine Learning:**
- DeBERTa-v3-small (transformer model)
- PyTorch + Hugging Face Transformers
- ONNX Runtime for optimized inference (2x speedup)
- Multi-label classification (3 labels)

**Backend:**
- FastAPI REST API with ONNX inference
- Optimized for low-latency production deployment

## Project Structure

```
TweetSweep/
├── backend/
│   ├── ml/              # Model training & inference
│   │   ├── data/        # Datasets
│   │   ├── models/      # Trained models & ONNX exports
│   │   ├── train.py     # Model training script
│   │   ├── inference.py # Interactive testing tool
│   │   └── export_onnx.py # ONNX export script
│   └── api/             # FastAPI REST API
│       └── model.py     # Production API endpoint
└── README.md
```

## Setup (Windows)

### 1. Create Virtual Environment

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r backend\requirements.txt
```

### 3. Verify Setup

```powershell
python backend\ml\test_setup.py
```

## Testing Your Text

Use the interactive inference script to test if your tweets or any text contains toxic content:

```powershell
python backend\ml\inference.py
```

The script will:
1. Run test examples automatically
2. Enter interactive mode where you can type any text
3. Display probability scores and flagged status for each category

**Example Output:**
```
Text: You're such an idiot, go away
------------------------------------------------------------
⚠️  toxic        [████████████████████████████░░] 85.2%
   hate_speech  [████████░░░░░░░░░░░░░░░░░░░░░░] 28.1%
   profanity    [████████████████░░░░░░░░░░░░░░] 52.3%
------------------------------------------------------------
🚨 FLAGGED: toxic
```

## API Usage

Start the FastAPI server:

```powershell
cd backend\api
uvicorn model:app --reload
```

**Predict Endpoint:**
```bash
POST /predict
Content-Type: application/json

{
  "tweets": [
    "I love this beautiful day!",
    "You're such an idiot"
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "tweet": "I love this beautiful day!",
      "hate_speech": {"probability": 0.02, "flagged": false},
      "toxic": {"probability": 0.05, "flagged": false},
      "profanity": {"probability": 0.01, "flagged": false}
    },
    {
      "tweet": "You're such an idiot",
      "hate_speech": {"probability": 0.15, "flagged": false},
      "toxic": {"probability": 0.85, "flagged": true},
      "profanity": {"probability": 0.12, "flagged": false}
    }
  ],
  "latency_ms": 32.45
}
```

## Performance

- **Inference Speed**: ~32ms per tweet (ONNX Runtime on CPU)
- **Throughput**: ~31 tweets/second
- **Optimization**: 2x faster than PyTorch (64ms → 32ms)

## Project Status

- [x] Dataset acquisition and preprocessing
- [x] Model training (DeBERTa-v3-small)
- [x] Multi-label classification implementation
- [x] ONNX optimization (2x speedup)
- [x] FastAPI REST API
- [x] Interactive inference tool
- [ ] Automatic tweet deletion (not implemented due to X API rate limits)

## Note on Tweet Deletion

Due to X (Twitter) API rate limits and restrictions, automatic tweet deletion functionality is not implemented. Users can test their tweets using the inference script (`backend/ml/inference.py`) or the REST API to check for toxic content before posting.

## Model Details

- **Architecture**: DeBERTa-v3-small
- **Task**: Multi-label classification
- **Labels**: 3 (hate_speech, toxic, profanity)
- **Max Sequence Length**: 128 tokens
- **Threshold**: 0.5 for flagging

## License

MIT
