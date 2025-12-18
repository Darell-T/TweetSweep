import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    DebertaV2Tokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

MODEL_NAME = "microsoft/deberta-v3-small"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

print("Loading dataset...")
train_dataset = load_from_disk("data/processed/combined_dataset")
train_dataset = train_dataset.shuffle(seed=42)

print(f"Total samples: {len(train_dataset):,}")

train_size = int(0.8 * len(train_dataset))
val_size = int(0.1 * len(train_dataset))

train_data = train_dataset.select(range(train_size))
val_data = train_dataset.select(range(train_size, train_size + val_size))
test_data = train_dataset.select(range(train_size + val_size, len(train_dataset)))

print(f"Train: {len(train_data):,}")
print(f"Val: {len(val_data):,}")
print(f"Test: {len(test_data):,}")

print("\nLoading tokenizer and model...")
tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    problem_type="multi_label_classification"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

def preprocess_function(examples):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

    labels = []
    for i in range(len(examples["text"])):
        label = [
            float(examples["hate_speech"][i]),
            float(examples["toxic"][i]),
            float(examples["profanity"][i])
        ]
        labels.append(label)
    
    tokenized["labels"] = labels
    return tokenized

print("\nTokenizing datasets...")
train_data = train_data.map(preprocess_function, batched=True)
val_data = val_data.map(preprocess_function, batched=True)
test_data = test_data.map(preprocess_function, batched=True)

train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
val_data.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
test_data.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    predictions = torch.sigmoid(torch.tensor(logits))
    predictions = (predictions > 0.5).float().numpy()
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average='micro',
        zero_division=0
    )
    
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

training_args = TrainingArguments(
    output_dir="models/checkpoints",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=torch.cuda.is_available(),
    logging_steps=100,
    save_total_limit=2,
    report_to="none",
)

print("\nInitializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics,
)

print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80)

trainer.train()

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)

print("\nEvaluating on test set...")
test_results = trainer.evaluate(test_data)

print("\n" + "="*80)
print("TEST SET RESULTS")
print("="*80)
for key, value in test_results.items():
    print(f"{key}: {value:.4f}")
print("="*80)

print("\nSaving model...")
final_model_path = "models/final_model"
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)

print(f"✓ Model saved to: {final_model_path}")
print("\nTraining complete!")