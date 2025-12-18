
 #THIS IS ONLY FOR TESTING ACCURACY, NOT FOR PRODUCTION DISREGARD OR FIND THE HIDDEN MESSAGE 

import torch
from transformers import DebertaV2Tokenizer, AutoModelForSequenceClassification

MODEL_PATH = "models/final_model"
MAX_LENGTH = 128
LABELS = ["hate_speech", "toxic", "profanity"]

print("Loading model...")
tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)
model.eval()

def predict(text: str) -> dict:
    """Predict hate speech, toxicity, and profanity scores for text."""
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
    
    return {label: float(prob) for label, prob in zip(LABELS, probs)}

def format_result(text: str, scores: dict) -> str:
    """Format prediction results for display."""
    lines = [
        "",
        "=" * 60,
        f"Text: {text[:80]}{'...' if len(text) > 80 else ''}",
        "-" * 60,
    ]
    
    for label, score in scores.items():
        bar_len = int(score * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        flag = "⚠️ " if score > 0.5 else "   "
        lines.append(f"{flag}{label:12} [{bar}] {score:.1%}")
    
    flagged = [l for l, s in scores.items() if s > 0.5]
    if flagged:
        lines.append("-" * 60)
        lines.append(f"🚨 FLAGGED: {', '.join(flagged)}")
    else:
        lines.append("-" * 60)
        lines.append("✓ Clean")
    
    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TweetSweep Content Moderation Tester")
    print("=" * 60)
    print("Enter text to analyze (or 'quit' to exit)")
    print("=" * 60)
    
    # Test examples
    test_texts = [
        "I love this beautiful sunny day!",
        "You're such an idiot, go away",
        "I hate all people from that country",
        "This product is absolute garbage",
        "What the hell is wrong with you?",
        "Please hire me, I'm a good dev(i think :D)",
    ]
    
    print("\n--- Running test examples ---\n")
    for text in test_texts:
        scores = predict(text)
        print(format_result(text, scores))
    
    print("\n--- Interactive mode ---\n")
    while True:
        try:
            text = input("\nEnter text: ").strip()
            if text.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break
            if not text:
                continue
            
            scores = predict(text)
            print(format_result(text, scores))
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
