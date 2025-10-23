"""
BEST PREPROCESSING PIPELINE FOR AI VS HUMAN TEXT DETECTION
===========================================================
- Cleans text thoroughly
- Removes duplicates
- Balances classes
- Creates stratified train/val/test splits
- Saves statistics and ready-to-use CSVs
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

# Configuration
INPUT_FILE = "AI_Human.csv"  # Your dataset
OUTPUT_DIR = "processed_data"
TRAIN_SIZE = 20000
VAL_SIZE = 10000
TEST_SIZE = 10000
RANDOM_SEED = 42

print("=" * 70)
print("AI TEXT DETECTION - COMPREHENSIVE PREPROCESSING")
print("=" * 70)
print()

# Create output directory
import os

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
print(f"Loading dataset from: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"Initial samples: {len(df):,}")
print(f"Columns: {df.columns.tolist()}")
print()

# Rename 'generated' to 'label' for consistency
df = df.rename(columns={'generated': 'label'})

# Display initial label distribution
print("Initial label distribution:")
print(df['label'].value_counts())
print()


# ============= TEXT CLEANING =============
def clean_text(text):
    """
    Comprehensive text cleaning:
    - Convert to string and strip whitespace
    - Remove URLs
    - Remove email addresses
    - Remove extra whitespace
    - Remove special characters (optional)
    - Keep only ASCII characters (optional, comment out if you need Unicode)
    """
    if pd.isna(text):
        return ""

    text = str(text).strip()

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


print("Cleaning text...")
df['text'] = df['text'].apply(clean_text)
print("✓ Text cleaning complete")
print()

# ============= REMOVE EMPTY TEXTS =============
initial_count = len(df)
df = df[df['text'].str.len() > 0]
empty_removed = initial_count - len(df)
print(f"Removed {empty_removed:,} empty texts")
print(f"Remaining samples: {len(df):,}")
print()

# ============= REMOVE DUPLICATES =============
initial_count = len(df)
df = df.drop_duplicates(subset=['text'], keep='first')
duplicates_removed = initial_count - len(df)
print(f"Removed {duplicates_removed:,} duplicate texts")
print(f"Remaining samples: {len(df):,}")
print()

# ============= FILTER VALID LABELS =============
df = df[df['label'].isin([0, 1])]
print(f"Samples after filtering valid labels: {len(df):,}")
print()

# ============= CHECK LABEL BALANCE =============
print("Label distribution after cleaning:")
label_counts = df['label'].value_counts()
print(label_counts)
print()

count_human = (df['label'] == 0).sum()
count_ai = (df['label'] == 1).sum()

# Verify sufficient samples
samples_needed = (TRAIN_SIZE + VAL_SIZE + TEST_SIZE) // 2
print(f"Samples needed per class: {samples_needed:,}")
print(f"Human samples available: {count_human:,}")
print(f"AI samples available: {count_ai:,}")
print()

if count_human < samples_needed:
    print(f"⚠️  WARNING: Not enough human samples! Need {samples_needed:,}, have {count_human:,}")
    print(f"Adjusting to use maximum available: {count_human:,}")
    samples_needed = min(count_human, count_ai)

if count_ai < samples_needed:
    print(f"⚠️  WARNING: Not enough AI samples! Need {samples_needed:,}, have {count_ai:,}")
    print(f"Adjusting to use maximum available: {count_ai:,}")
    samples_needed = min(count_human, count_ai)

print(f"✓ Using {samples_needed:,} samples per class")
print()

# ============= SAMPLE BALANCED DATASET =============
print(f"Sampling balanced dataset...")
human_samples = df[df['label'] == 0].sample(n=samples_needed, random_state=RANDOM_SEED)
ai_samples = df[df['label'] == 1].sample(n=samples_needed, random_state=RANDOM_SEED)

# Combine and shuffle
balanced_df = pd.concat([human_samples, ai_samples])
balanced_df = balanced_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"✓ Balanced dataset created: {len(balanced_df):,} samples")
print(f"  Human: {(balanced_df['label'] == 0).sum():,}")
print(f"  AI: {(balanced_df['label'] == 1).sum():,}")
print()

# ============= CREATE STRATIFIED SPLITS =============
print("Creating stratified train/val/test splits...")

# Calculate split sizes based on available samples
total_samples = len(balanced_df)
train_size = int(0.5 * total_samples)
val_size = int(0.25 * total_samples)
test_size = total_samples - train_size - val_size

# First split: train vs (val+test)
train_df, temp_df = train_test_split(
    balanced_df,
    test_size=(val_size + test_size),
    stratify=balanced_df['label'],
    random_state=RANDOM_SEED
)

# Second split: val vs test
val_df, test_df = train_test_split(
    temp_df,
    test_size=test_size,
    stratify=temp_df['label'],
    random_state=RANDOM_SEED
)

print("✓ Splits created")
print()

# ============= SAVE TO CSV =============
train_path = os.path.join(OUTPUT_DIR, "train.csv")
val_path = os.path.join(OUTPUT_DIR, "validation.csv")
test_path = os.path.join(OUTPUT_DIR, "test.csv")

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

print("=" * 70)
print("PREPROCESSING COMPLETE!")
print("=" * 70)
print()
print("FINAL STATISTICS:")
print()
print(f"TRAIN SET ({train_path}):")
print(f"  Total samples: {len(train_df):,}")
print(f"  Human (label 0): {(train_df['label'] == 0).sum():,}")
print(f"  AI (label 1): {(train_df['label'] == 1).sum():,}")
print()
print(f"VALIDATION SET ({val_path}):")
print(f"  Total samples: {len(val_df):,}")
print(f"  Human (label 0): {(val_df['label'] == 0).sum():,}")
print(f"  AI (label 1): {(val_df['label'] == 1).sum():,}")
print()
print(f"TEST SET ({test_path}):")
print(f"  Total samples: {len(test_df):,}")
print(f"  Human (label 0): {(test_df['label'] == 0).sum():,}")
print(f"  AI (label 1): {(test_df['label'] == 1).sum():,}")
print()
print("=" * 70)
print("Files saved in:", OUTPUT_DIR)
print("=" * 70)
print()
print("Next steps:")
print("1. Review the statistics above to ensure balance")
print("2. (Optional) Add 10,000 humanized AI samples with label=2")
print("3. Run your training script with these files")
