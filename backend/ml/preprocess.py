import sys
sys.stdout.reconfigure(encoding='utf-8')

from datasets import load_dataset, concatenate_datasets
import re
import ftfy
import html
from profanity_check import predict
import numpy as np

print("Loading datasets...")
ds1 = load_dataset("tweets-hate-speech-detection/tweets_hate_speech_detection")
ds2 = load_dataset("Masabanees619/toxic_tweets_and_comments")
ds3 = load_dataset("Om1024/racist-sexist")

def clean_tweets(text):
    if not text or not isinstance(text, str):
        return ""
    
    text = ftfy.fix_text(text)
    text = html.unescape(text)
    text = re.sub(r'http\S+|www\.\S+', '[URL]', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def map_hate_speech_tweets(example):
    tweet_text = example['tweet']
    is_hate = example['label']
    
    return {
        'text': clean_tweets(tweet_text),
        'hate_speech': is_hate,
        'toxic': is_hate
    }

def map_toxic_tweets(example):
    tweet_text = example['clean_text']
    
    return {
        'text': clean_tweets(tweet_text),
        'hate_speech': 0,
        'toxic': 1
    }

def map_racist_sexist_tweets(example):
    tweet_text = example['text']
    
    if example['neutral'] == 1:
        return {
            'text': clean_tweets(tweet_text),
            'hate_speech': 0,
            'toxic': 0
        }
    
    is_hate = example['racist'] == 1 or example['sexist'] == 1
    
    return {
        'text': clean_tweets(tweet_text),
        'hate_speech': 1 if is_hate else 0,
        'toxic': 1 if is_hate else 0
    }

def add_profanity_batch(examples):
    texts = examples['text']
    profanity_results = predict(texts)
    return {'profanity': [int(x) for x in profanity_results]}

print("\nMapping datasets...")

print("1. Hate speech tweets...")
ds1_mapped = ds1['train'].map(
    map_hate_speech_tweets, 
    remove_columns=ds1['train'].column_names
)

print("2. Toxic tweets...")
ds2_mapped = ds2['train'].map(
    map_toxic_tweets, 
    remove_columns=ds2['train'].column_names
)

print("3. Racist/Sexist tweets...")
ds3_mapped = ds3['train'].map(
    map_racist_sexist_tweets, 
    remove_columns=ds3['train'].column_names
)

print("\nCombining datasets...")
combined_ds = concatenate_datasets([ds1_mapped, ds2_mapped, ds3_mapped])

combined_ds = combined_ds.filter(lambda x: len(x['text']) > 3)

print("\nAdding profanity labels (batched for speed)...")
combined_ds = combined_ds.map(add_profanity_batch, batched=True, batch_size=1000)

print(f"\n[OK] Combined dataset size: {len(combined_ds):,}")
print(f"[OK] Columns: {combined_ds.column_names}")

print("\n" + "="*80)
print("LABEL DISTRIBUTION")
print("="*80)

sample_size = min(10000, len(combined_ds))
sample = combined_ds.select(range(sample_size))

hate_count = sum(1 for ex in sample if ex['hate_speech'] == 1)
toxic_count = sum(1 for ex in sample if ex['toxic'] == 1)
prof_count = sum(1 for ex in sample if ex['profanity'] == 1)

print(f"Sample: {sample_size:,} / {len(combined_ds):,}")
print(f"Hate speech: {hate_count:,} ({hate_count/sample_size*100:.1f}%)")
print(f"Toxic: {toxic_count:,} ({toxic_count/sample_size*100:.1f}%)")
print(f"Profanity: {prof_count:,} ({prof_count/sample_size*100:.1f}%)")

print("\n" + "="*80)
print("Saving combined dataset...")
combined_ds.save_to_disk("data/processed/combined_dataset")
print("[OK] Saved to: data/processed/combined_dataset")

print("="*80)
