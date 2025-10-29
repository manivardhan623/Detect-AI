"""
AI Text Detection - Comprehensive Data Analysis and Visualization
Generates multiple graphs and charts for the model performance and dataset analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score
)
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import warnings
import os
import sys

warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create results directory if it doesn't exist
os.makedirs('visualizations', exist_ok=True)


def check_required_files():
    """Check if all required files and folders exist"""
    print("\n" + "=" * 70)
    print("CHECKING REQUIRED FILES")
    print("=" * 70)

    required_files = {
        'Model directory': 'ensemble_models/roberta-base',
        'Test data': 'processed_data/test_data.csv',
        'Train data': 'processed_data/train_data.csv',
        'Validation data': 'processed_data/validation_data.csv'
    }

    missing_files = []

    for name, path in required_files.items():
        if os.path.exists(path):
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name}: {path} - NOT FOUND")
            missing_files.append((name, path))

    if missing_files:
        print("\n" + "=" * 70)
        print("ERROR: MISSING REQUIRED FILES")
        print("=" * 70)
        for name, path in missing_files:
            print(f"  - {name}: {path}")

        print("\nPLEASE RUN THE FOLLOWING STEPS FIRST:")
        print("  1. Run: python preprocess.py")
        print("     (This creates processed_data folder with train/val/test CSV files)")
        print("  2. Run: python train.py")
        print("     (This creates the trained model in ensemble_models folder)")
        print("  3. Then run this visualization script again")
        print("=" * 70)
        sys.exit(1)

    print("\n✓ All required files found!")
    return True


def load_model_and_data():
    """Load the trained model and test data"""
    print("\n" + "=" * 70)
    print("LOADING MODEL AND DATA")
    print("=" * 70)

    # Load model
    print("Loading model...")
    model_dir = "ensemble_models/roberta-base"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
        model.to(device)
        model.eval()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nPlease ensure you have trained the model by running: python train.py")
        sys.exit(1)

    # Load test data (UPDATED FILE NAMES)
    print("Loading datasets...")
    try:
        test_df = pd.read_csv('processed_data/test_data.csv')
        train_df = pd.read_csv('processed_data/train_data.csv')
        val_df = pd.read_csv('processed_data/validation_data.csv')
        print(f"✓ Test samples: {len(test_df)}")
        print(f"✓ Train samples: {len(train_df)}")
        print(f"✓ Validation samples: {len(val_df)}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("\nPlease ensure you have preprocessed the data by running: python preprocess.py")
        sys.exit(1)

    return model, tokenizer, test_df, train_df, val_df


def get_predictions_and_probabilities(model, tokenizer, test_df):
    """Get predictions and probability scores for all test samples"""
    print("\n" + "=" * 70)
    print("GENERATING PREDICTIONS")
    print("=" * 70)

    texts = test_df['text'].tolist()
    true_labels = test_df['label'].tolist()

    predictions = []
    probabilities = []

    print("Making predictions on test set...")
    for text in tqdm(texts, desc="Processing"):
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True,
                           truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            prob = probs[0, 1].item()  # Probability of AI class
            pred = 1 if prob >= 0.5 else 0

        predictions.append(pred)
        probabilities.append(prob)

    print("✓ Predictions complete")
    return true_labels, predictions, probabilities


def plot_1_confusion_matrix(y_true, y_pred):
    """Generate confusion matrix heatmap"""
    print("\n[1/15] Generating Confusion Matrix...")

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Human', 'AI'],
                yticklabels=['Human', 'AI'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - AI Text Detection', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)

    # Add accuracy text
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    plt.text(1, -0.3, f'Overall Accuracy: {accuracy:.2%}',
             ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('visualizations/1_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: visualizations/1_confusion_matrix.png")


def plot_2_roc_curve(y_true, y_probs):
    """Generate ROC curve"""
    print("\n[2/15] Generating ROC Curve...")

    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve',
              fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/2_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: visualizations/2_roc_curve.png")


def plot_3_precision_recall_curve(y_true, y_probs):
    """Generate Precision-Recall curve"""
    print("\n[3/15] Generating Precision-Recall Curve...")

    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    avg_precision = average_precision_score(y_true, y_probs)

    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig('visualizations/3_precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: visualizations/3_precision_recall_curve.png")


def plot_4_class_distribution(train_df, val_df, test_df):
    """Generate class distribution across datasets"""
    print("\n[4/15] Generating Class Distribution Chart...")

    datasets = ['Train', 'Validation', 'Test']
    human_counts = [
        (train_df['label'] == 0).sum(),
        (val_df['label'] == 0).sum(),
        (test_df['label'] == 0).sum()
    ]
    ai_counts = [
        (train_df['label'] == 1).sum(),
        (val_df['label'] == 1).sum(),
        (test_df['label'] == 1).sum()
    ]

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 8))
    bars1 = ax.bar(x - width / 2, human_counts, width, label='Human', color='skyblue')
    bars2 = ax.bar(x + width / 2, ai_counts, width, label='AI', color='lightcoral')

    ax.set_xlabel('Dataset Split', fontsize=14)
    ax.set_ylabel('Number of Samples', fontsize=14)
    ax.set_title('Class Distribution Across Dataset Splits', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('visualizations/4_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: visualizations/4_class_distribution.png")


def plot_5_text_length_distribution(train_df, test_df):
    """Generate text length distribution"""
    print("\n[5/15] Generating Text Length Distribution...")

    train_lengths = train_df['text'].str.len()
    test_lengths = test_df['text'].str.len()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Training set
    ax1.hist(train_lengths, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(train_lengths.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {train_lengths.mean():.0f}')
    ax1.axvline(train_lengths.median(), color='green', linestyle='--', linewidth=2,
                label=f'Median: {train_lengths.median():.0f}')
    ax1.set_xlabel('Text Length (characters)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Training Set - Text Length Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Test set
    ax2.hist(test_lengths, bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax2.axvline(test_lengths.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {test_lengths.mean():.0f}')
    ax2.axvline(test_lengths.median(), color='green', linestyle='--', linewidth=2,
                label=f'Median: {test_lengths.median():.0f}')
    ax2.set_xlabel('Text Length (characters)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Test Set - Text Length Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualizations/5_text_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: visualizations/5_text_length_distribution.png")


def plot_6_confidence_distribution(y_probs):
    """Generate prediction confidence distribution"""
    print("\n[6/15] Generating Confidence Distribution...")

    plt.figure(figsize=(12, 6))

    plt.hist(y_probs, bins=50, color='mediumpurple', alpha=0.7, edgecolor='black')
    plt.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
    plt.xlabel('AI Probability Score', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Distribution of Model Confidence Scores', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add statistics
    high_conf = sum(1 for p in y_probs if p > 0.95 or p < 0.05)
    low_conf = sum(1 for p in y_probs if 0.4 < p < 0.6)
    plt.text(0.5, plt.ylim()[1] * 0.9,
             f'High Confidence (>0.95 or <0.05): {high_conf} ({high_conf / len(y_probs) * 100:.1f}%)\n'
             f'Low Confidence (0.4-0.6): {low_conf} ({low_conf / len(y_probs) * 100:.1f}%)',
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('visualizations/6_confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: visualizations/6_confidence_distribution.png")


def plot_7_error_analysis(test_df, y_true, y_pred, y_probs):
    """Analyze errors by text length"""
    print("\n[7/15] Generating Error Analysis by Text Length...")

    test_df_copy = test_df.copy()
    test_df_copy['predicted'] = y_pred
    test_df_copy['probability'] = y_probs
    test_df_copy['text_length'] = test_df_copy['text'].str.len()
    test_df_copy['correct'] = test_df_copy['label'] == test_df_copy['predicted']

    correct_lengths = test_df_copy[test_df_copy['correct']]['text_length']
    incorrect_lengths = test_df_copy[~test_df_copy['correct']]['text_length']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Box plot
    data_to_plot = [correct_lengths, incorrect_lengths]
    bp = ax1.boxplot(data_to_plot, labels=['Correct', 'Incorrect'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax1.set_ylabel('Text Length (characters)', fontsize=12)
    ax1.set_title('Text Length Distribution: Correct vs Incorrect',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add mean lines
    if len(correct_lengths) > 0:
        ax1.axhline(correct_lengths.mean(), color='green', linestyle='--',
                    xmin=0.1, xmax=0.4, label=f'Correct Mean: {correct_lengths.mean():.0f}')
    if len(incorrect_lengths) > 0:
        ax1.axhline(incorrect_lengths.mean(), color='red', linestyle='--',
                    xmin=0.6, xmax=0.9, label=f'Incorrect Mean: {incorrect_lengths.mean():.0f}')
    ax1.legend()

    # Histogram overlay
    ax2.hist(correct_lengths, bins=30, alpha=0.5, label='Correct', color='green', edgecolor='black')
    if len(incorrect_lengths) > 0:
        ax2.hist(incorrect_lengths, bins=30, alpha=0.5, label='Incorrect', color='red', edgecolor='black')
    ax2.set_xlabel('Text Length (characters)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Text Length Histogram: Correct vs Incorrect',
                  fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualizations/7_error_analysis_length.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: visualizations/7_error_analysis_length.png")


def plot_8_performance_metrics_bar(y_true, y_pred):
    """Generate performance metrics bar chart"""
    print("\n[8/15] Generating Performance Metrics Bar Chart...")

    # Calculate metrics for both classes
    metrics_human = {
        'Precision': precision_score(y_true, y_pred, pos_label=0),
        'Recall': recall_score(y_true, y_pred, pos_label=0),
        'F1-Score': f1_score(y_true, y_pred, pos_label=0)
    }

    metrics_ai = {
        'Precision': precision_score(y_true, y_pred, pos_label=1),
        'Recall': recall_score(y_true, y_pred, pos_label=1),
        'F1-Score': f1_score(y_true, y_pred, pos_label=1)
    }

    overall_accuracy = accuracy_score(y_true, y_pred)

    # Plotting
    x = np.arange(len(metrics_human))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 8))
    bars1 = ax.bar(x - width / 2, list(metrics_human.values()), width,
                   label='Human Text', color='skyblue')
    bars2 = ax.bar(x + width / 2, list(metrics_ai.values()), width,
                   label='AI Text', color='lightcoral')

    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Performance Metrics by Class', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics_human.keys()), fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim([0.0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10)

    # Add overall accuracy
    ax.text(0.5, 0.02, f'Overall Accuracy: {overall_accuracy:.4f}',
            transform=ax.transAxes, ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig('visualizations/8_performance_metrics_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: visualizations/8_performance_metrics_bar.png")


def plot_9_false_positives_negatives(y_true, y_pred):
    """Compare false positives and false negatives"""
    print("\n[9/15] Generating False Positives vs False Negatives Chart...")

    cm = confusion_matrix(y_true, y_pred)
    fp = cm[0, 1]  # Human predicted as AI
    fn = cm[1, 0]  # AI predicted as Human
    tp = cm[1, 1]  # Correctly identified AI
    tn = cm[0, 0]  # Correctly identified Human

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Error counts
    errors = ['False Positives\n(Human→AI)', 'False Negatives\n(AI→Human)']
    counts = [fp, fn]
    colors = ['coral', 'skyblue']

    bars = ax1.bar(errors, counts, color=colors, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Count', fontsize=14)
    ax1.set_title('Error Type Comparison', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, count in zip(bars, counts):
        if fp + fn > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                     f'{count}\n({count / (fp + fn) * 100:.1f}%)',
                     ha='center', va='bottom', fontsize=12, fontweight='bold')
        else:
            ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                     f'{count}',
                     ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Pie chart of predictions
    labels = ['True Negatives\n(Human)', 'False Positives\n(Human→AI)',
              'False Negatives\n(AI→Human)', 'True Positives\n(AI)']
    sizes = [tn, fp, fn, tp]
    colors_pie = ['lightgreen', 'coral', 'skyblue', 'gold']
    explode = (0, 0.1, 0.1, 0)

    ax2.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
            autopct='%1.1f%%', shadow=True, startangle=90)
    ax2.set_title('Overall Prediction Distribution', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('visualizations/9_false_positives_negatives.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: visualizations/9_false_positives_negatives.png")


def plot_10_threshold_analysis(y_true, y_probs):
    """Analyze different threshold values"""
    print("\n[10/15] Generating Threshold Analysis...")

    thresholds = np.arange(0.1, 1.0, 0.05)
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for threshold in thresholds:
        y_pred_thresh = [1 if p >= threshold else 0 for p in y_probs]
        accuracies.append(accuracy_score(y_true, y_pred_thresh))
        precisions.append(precision_score(y_true, y_pred_thresh, zero_division=0))
        recalls.append(recall_score(y_true, y_pred_thresh, zero_division=0))
        f1_scores.append(f1_score(y_true, y_pred_thresh, zero_division=0))

    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, accuracies, 'o-', label='Accuracy', linewidth=2, markersize=6)
    plt.plot(thresholds, precisions, 's-', label='Precision', linewidth=2, markersize=6)
    plt.plot(thresholds, recalls, '^-', label='Recall', linewidth=2, markersize=6)
    plt.plot(thresholds, f1_scores, 'd-', label='F1-Score', linewidth=2, markersize=6)

    plt.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Default Threshold (0.5)')
    plt.xlabel('Classification Threshold', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.title('Performance Metrics vs Classification Threshold', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.xlim([0.1, 0.95])
    plt.ylim([0.0, 1.05])

    plt.tight_layout()
    plt.savefig('visualizations/10_threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: visualizations/10_threshold_analysis.png")


def plot_11_word_count_distribution(train_df, test_df):
    """Distribution of word counts"""
    print("\n[11/15] Generating Word Count Distribution...")

    train_word_counts = train_df['text'].str.split().str.len()
    test_word_counts = test_df['text'].str.split().str.len()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Training set histogram
    axes[0, 0].hist(train_word_counts, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(train_word_counts.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {train_word_counts.mean():.0f}')
    axes[0, 0].set_xlabel('Word Count', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Training Set - Word Count Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Test set histogram
    axes[0, 1].hist(test_word_counts, bins=50, color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(test_word_counts.mean(), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {test_word_counts.mean():.0f}')
    axes[0, 1].set_xlabel('Word Count', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Test Set - Word Count Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Box plot comparison
    data_to_plot = [train_word_counts, test_word_counts]
    bp = axes[1, 0].boxplot(data_to_plot, labels=['Train', 'Test'], patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('coral')
    axes[1, 0].set_ylabel('Word Count', fontsize=12)
    axes[1, 0].set_title('Word Count Comparison: Train vs Test', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Statistics table
    stats_data = [
        ['Mean', f'{train_word_counts.mean():.1f}', f'{test_word_counts.mean():.1f}'],
        ['Median', f'{train_word_counts.median():.1f}', f'{test_word_counts.median():.1f}'],
        ['Std Dev', f'{train_word_counts.std():.1f}', f'{test_word_counts.std():.1f}'],
        ['Min', f'{train_word_counts.min():.0f}', f'{test_word_counts.min():.0f}'],
        ['Max', f'{train_word_counts.max():.0f}', f'{test_word_counts.max():.0f}']
    ]

    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    table = axes[1, 1].table(cellText=stats_data,
                             colLabels=['Statistic', 'Train Set', 'Test Set'],
                             cellLoc='center', loc='center',
                             colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    axes[1, 1].set_title('Word Count Statistics', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('visualizations/11_word_count_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: visualizations/11_word_count_distribution.png")


def plot_12_class_wise_confidence(test_df, y_true, y_probs):
    """Confidence distribution by class"""
    print("\n[12/15] Generating Class-wise Confidence Distribution...")

    human_probs = [p for i, p in enumerate(y_probs) if y_true[i] == 0]
    ai_probs = [p for i, p in enumerate(y_probs) if y_true[i] == 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Human texts
    ax1.hist(human_probs, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
    ax1.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
    ax1.set_xlabel('AI Probability Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Confidence Distribution - Human Texts', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add statistics
    below_threshold = sum(1 for p in human_probs if p < 0.5)
    ax1.text(0.5, ax1.get_ylim()[1] * 0.9,
             f'Correctly Classified: {below_threshold} ({below_threshold / len(human_probs) * 100:.1f}%)',
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # AI texts
    ax2.hist(ai_probs, bins=50, color='lightcoral', alpha=0.7, edgecolor='black')
    ax2.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
    ax2.set_xlabel('AI Probability Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Confidence Distribution - AI Texts', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add statistics
    above_threshold = sum(1 for p in ai_probs if p >= 0.5)
    ax2.text(0.5, ax2.get_ylim()[1] * 0.9,
             f'Correctly Classified: {above_threshold} ({above_threshold / len(ai_probs) * 100:.1f}%)',
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    plt.tight_layout()
    plt.savefig('visualizations/12_class_wise_confidence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: visualizations/12_class_wise_confidence.png")


def plot_13_dataset_size_comparison(train_df, val_df, test_df):
    """Dataset split size visualization"""
    print("\n[13/15] Generating Dataset Size Comparison...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Pie chart
    sizes = [len(train_df), len(val_df), len(test_df)]
    labels = [f'Training\n({len(train_df):,})', f'Validation\n({len(val_df):,})', f'Test\n({len(test_df):,})']
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    explode = (0.1, 0, 0)

    ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 12})
    ax1.set_title('Dataset Split Distribution', fontsize=16, fontweight='bold')

    # Bar chart
    splits = ['Training', 'Validation', 'Test']
    bars = ax2.bar(splits, sizes, color=colors, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Number of Samples', fontsize=14)
    ax2.set_title('Dataset Split Sizes', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, size in zip(bars, sizes):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                 f'{size:,}\n({size / sum(sizes) * 100:.0f}%)',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('visualizations/13_dataset_size_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: visualizations/13_dataset_size_comparison.png")


def plot_14_correlation_heatmap(test_df, y_true, y_pred, y_probs):
    """Correlation between various features"""
    print("\n[14/15] Generating Feature Correlation Heatmap...")

    # Create feature dataframe
    feature_df = pd.DataFrame({
        'Text_Length': test_df['text'].str.len(),
        'Word_Count': test_df['text'].str.split().str.len(),
        'Avg_Word_Length': test_df['text'].str.len() / test_df['text'].str.split().str.len(),
        'True_Label': y_true,
        'Predicted_Label': y_pred,
        'AI_Probability': y_probs,
        'Prediction_Correct': [1 if y_true[i] == y_pred[i] else 0 for i in range(len(y_true))]
    })

    # Calculate correlation matrix
    corr_matrix = feature_df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/14_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: visualizations/14_correlation_heatmap.png")


def plot_15_model_performance_summary(y_true, y_pred, y_probs):
    """Comprehensive performance summary dashboard"""
    print("\n[15/15] Generating Model Performance Summary Dashboard...")

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Confusion Matrix (top-left)
    ax1 = fig.add_subplot(gs[0:2, 0])
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')

    # 2. ROC Curve (top-middle)
    ax2 = fig.add_subplot(gs[0, 1])
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.4f}')
    ax2.plot([0, 1], [0, 1], 'k--', lw=1)
    ax2.set_xlabel('False Positive Rate', fontsize=10)
    ax2.set_ylabel('True Positive Rate', fontsize=10)
    ax2.set_title('ROC Curve', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. Precision-Recall Curve (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    avg_precision = average_precision_score(y_true, y_probs)
    ax3.plot(recall, precision, lw=2, label=f'AP = {avg_precision:.4f}')
    ax3.set_xlabel('Recall', fontsize=10)
    ax3.set_ylabel('Precision', fontsize=10)
    ax3.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # 4. Metrics Bar Chart (middle-left merged)
    ax4 = fig.add_subplot(gs[1, 1:])
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'Specificity': cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    }
    bars = ax4.barh(list(metrics.keys()), list(metrics.values()),
                    color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax4.set_xlabel('Score', fontsize=12)
    ax4.set_title('Performance Metrics', fontsize=14, fontweight='bold')
    ax4.set_xlim([0.0, 1.05])
    ax4.grid(True, alpha=0.3, axis='x')

    for bar, value in zip(bars, metrics.values()):
        ax4.text(value, bar.get_y() + bar.get_height() / 2, f'{value:.4f}',
                 va='center', ha='left', fontsize=10, fontweight='bold')

    # 5. Error Distribution (bottom-left)
    ax5 = fig.add_subplot(gs[2, 0])
    fp = cm[0, 1]
    fn = cm[1, 0]
    errors = ['False\nPositives', 'False\nNegatives']
    counts = [fp, fn]
    colors_err = ['coral', 'skyblue']
    bars = ax5.bar(errors, counts, color=colors_err, edgecolor='black')
    ax5.set_ylabel('Count', fontsize=10)
    ax5.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    for bar, count in zip(bars, counts):
        ax5.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                 f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 6. Confidence Distribution (bottom-middle)
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.hist(y_probs, bins=30, color='mediumpurple', alpha=0.7, edgecolor='black')
    ax6.axvline(0.5, color='red', linestyle='--', linewidth=2)
    ax6.set_xlabel('AI Probability', fontsize=10)
    ax6.set_ylabel('Frequency', fontsize=10)
    ax6.set_title('Confidence Distribution', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # 7. Key Statistics (bottom-right)
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    stats_text = f"""
    MODEL PERFORMANCE SUMMARY
    ═══════════════════════════

    Total Samples: {len(y_true):,}
    Correct: {sum([1 for i in range(len(y_true)) if y_true[i] == y_pred[i]]):,}
    Incorrect: {sum([1 for i in range(len(y_true)) if y_true[i] != y_pred[i]]):,}

    Accuracy: {accuracy_score(y_true, y_pred):.4f}
    Error Rate: {1 - accuracy_score(y_true, y_pred):.4f}

    True Positives: {cm[1, 1]:,}
    True Negatives: {cm[0, 0]:,}
    False Positives: {fp:,}
    False Negatives: {fn:,}

    High Confidence: {sum(1 for p in y_probs if p > 0.95 or p < 0.05):,}
    Low Confidence: {sum(1 for p in y_probs if 0.4 < p < 0.6):,}
    """

    ax7.text(0.1, 0.9, stats_text, transform=ax7.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Main title
    fig.suptitle('AI Text Detection - Comprehensive Performance Dashboard',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.savefig('visualizations/15_performance_summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: visualizations/15_performance_summary_dashboard.png")


def main():
    """Main execution function"""
    print("\n" + "=" * 70)
    print(" " * 15 + "AI TEXT DETECTION")
    print(" " * 10 + "DATA ANALYSIS & VISUALIZATION")
    print("=" * 70)

    # Check if required files exist
    check_required_files()

    # Load model and data
    model, tokenizer, test_df, train_df, val_df = load_model_and_data()

    # Get predictions
    y_true, y_pred, y_probs = get_predictions_and_probabilities(model, tokenizer, test_df)

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    # Generate all plots
    plot_1_confusion_matrix(y_true, y_pred)
    plot_2_roc_curve(y_true, y_probs)
    plot_3_precision_recall_curve(y_true, y_probs)
    plot_4_class_distribution(train_df, val_df, test_df)
    plot_5_text_length_distribution(train_df, test_df)
    plot_6_confidence_distribution(y_probs)
    plot_7_error_analysis(test_df, y_true, y_pred, y_probs)
    plot_8_performance_metrics_bar(y_true, y_pred)
    plot_9_false_positives_negatives(y_true, y_pred)
    plot_10_threshold_analysis(y_true, y_probs)
    plot_11_word_count_distribution(train_df, test_df)
    plot_12_class_wise_confidence(test_df, y_true, y_probs)
    plot_13_dataset_size_comparison(train_df, val_df, test_df)
    plot_14_correlation_heatmap(test_df, y_true, y_pred, y_probs)
    plot_15_model_performance_summary(y_true, y_pred, y_probs)

    print("\n" + "=" * 70)
    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n📁 All 15 visualizations saved in: visualizations/")
    print("\nGenerated Files:")
    print("  1. confusion_matrix.png")
    print("  2. roc_curve.png")
    print("  3. precision_recall_curve.png")
    print("  4. class_distribution.png")
    print("  5. text_length_distribution.png")
    print("  6. confidence_distribution.png")
    print("  7. error_analysis_length.png")
    print("  8. performance_metrics_bar.png")
    print("  9. false_positives_negatives.png")
    print(" 10. threshold_analysis.png")
    print(" 11. word_count_distribution.png")
    print(" 12. class_wise_confidence.png")
    print(" 13. dataset_size_comparison.png")
    print(" 14. correlation_heatmap.png")
    print(" 15. performance_summary_dashboard.png")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()