import sys
sys.stdout.reconfigure(encoding='utf-8')

from datasets import load_from_disk

# Load the saved dataset
ds = load_from_disk('data/processed/combined_dataset')

print('=' * 80)
print(f'COMBINED DATASET - First 100 Samples')
print(f'Total size: {len(ds):,}')
print(f'Columns: {ds.column_names}')
print('=' * 80)

for i, sample in enumerate(ds.select(range(100))):
    print(f'\nSample {i+1}:')
    print(f'  Text: {sample["text"]}')
    print(f'  Hate Speech: {sample["hate_speech"]}')
    print(f'  Toxic: {sample["toxic"]}')
    print(f'  Profanity: {sample["profanity"]}')
    print('-' * 80)
