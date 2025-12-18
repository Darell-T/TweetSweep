from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import onnxruntime as ort
import numpy as np
from typing import List
from transformers import AutoTokenizer
import logging
import os
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ort_session = None
tokenizer = None
LABELS = ["hate_speech", "toxic", "profanity"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ort_session, tokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("models/final_model", use_fast=False)
    
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = min(4, os.cpu_count() or 4)
    
    ort_session = ort.InferenceSession(
        "models/ONNX/model.onnx",
        sess_options=sess_options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    logger.info("Model and tokenizer loaded")
    yield
    logger.info("Shutting down")


app = FastAPI(title="TweetSweep ML API", lifespan=lifespan)


class PredictionRequest(BaseModel):
    tweets: List[str]
    
    @field_validator('tweets')
    def validate_tweets(cls, v):
        if not v or len(v) > 50:
            raise ValueError("Need 1-50 tweets")
        if any(not t.strip() or len(t) > 512 for t in v):
            raise ValueError("Tweets must be non-empty and under 512 chars")
        return v


class PredictionResponse(BaseModel):
    predictions: List[dict]
    latency_ms: float


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        start = time.perf_counter()
        
        encoded = tokenizer(request.tweets, padding='max_length', truncation=True, 
                           max_length=128, return_tensors='np')
        
        ort_inputs = {
            'input_ids': encoded['input_ids'].astype(np.int64),
            'attention_mask': encoded['attention_mask'].astype(np.int64)
        }
        
        logits = ort_session.run(None, ort_inputs)[0]
        probs = 1 / (1 + np.exp(-logits))  # Sigmoid
        
        predictions = [
            {
                "tweet": tweet,
                **{label: {"probability": round(float(probs[i][j]), 4), "flagged": probs[i][j] > 0.5}
                   for j, label in enumerate(LABELS)}
            }
            for i, tweet in enumerate(request.tweets)
        ]
        
        return PredictionResponse(
            predictions=predictions,
            latency_ms=round((time.perf_counter() - start) * 1000, 2)
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy", "ready": ort_session is not None}


@app.get("/")
async def root():
    return {"name": "TweetSweep ML API", "docs": "/docs"}
