# TweetSweep

AI-powered Twitter profile cleanup tool using multi-label classification to detect toxic, unprofessional, and risky content.

## Tech Stack

**Machine Learning:**

- DeBERTa-v3-small (transformer model)
- PyTorch + Hugging Face Transformers
- ONNX Runtime for optimized inference

**Backend (Coming Soon):**

- FastAPI
- Celery + Redis
- PostgreSQL

**Frontend (Coming Soon):**

- Next.js + TypeScript
- Tailwind CSS

## Project Structure

```
TweetSweep/
├── backend/
│   ├── ml/              # Model training & inference
│   │   ├── data/        # Datasets
│   │   ├── models/      # Trained models & checkpoints
│   │   |
│   │   └── *.py         # Training scripts
│   └── api/             # FastAPI (future)
└── frontend/            # Next.js (future)
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

## Model Training (TODO)

Coming soon...

## Current Status

- [x] Project setup
- [ ] Dataset acquisition
- [ ] Data preprocessing
- [ ] Model training
- [ ] ONNX optimization
- [ ] FastAPI backend
- [ ] Frontend dashboard
