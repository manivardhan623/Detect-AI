import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import time

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def load_model(model_dir="ensemble_models/roberta-base"):
    """Load the trained model"""
    print(f"Loading model from {model_dir}...")

    # Check if model exists
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} not found!")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.to(device)
    model.eval()

    print("Model loaded successfully!")
    return model, tokenizer


def predict_batch(texts, model, tokenizer, batch_size=8):
    """Make predictions on a batch of texts"""
    all_predictions = []
    all_probabilities = []

    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
        batch_texts = texts[i:i + batch_size]

        # Convert each item to string to ensure proper formatting
        batch_texts = [str(text) for text in batch_texts]

        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs[:, 1].cpu().numpy())

    return all_predictions, all_probabilities


def evaluate_model(test_file, model_dir="ensemble_models/roberta-base"):
    """Evaluate model performance on test data"""
    start_time = time.time()

    # Load model
    model, tokenizer = load_model(model_dir)

    # Load test data
    print(f"Loading test data from {test_file}...")
    test_df = pd.read_csv(test_file)
    print(f"Loaded {len(test_df)} test samples.")

    # Check if columns need renaming
    if 'generated' in test_df.columns and 'text' in test_df.columns:
        test_df = test_df.rename(columns={'generated': 'label'})
    elif 'label' not in test_df.columns:
        raise ValueError("Could not find 'label' or 'generated' column in test data!")

    # Convert text column to string type
    test_df['text'] = test_df['text'].astype(str)

    # Check for any missing or NaN values
    if test_df['text'].isnull().any():
        print(f"Warning: Found {test_df['text'].isnull().sum()} null text entries. Filling with empty string.")
        test_df['text'] = test_df['text'].fillna('')

    # Display a few samples for verification
    print("\nSample data for verification:")
    for i, (text, label) in enumerate(zip(test_df['text'].values[:3], test_df['label'].values[:3])):
        print(f"Sample {i + 1} (label={label}):")
        print(f"Text type: {type(text)}")
        print(f"Text preview: {text[:50]}...\n")

    # Make predictions
    print("Running predictions on test data...")
    predictions, probabilities = predict_batch(test_df['text'].values, model, tokenizer)

    # Calculate metrics
    print("\nCalculating evaluation metrics...")
    accuracy = accuracy_score(test_df['label'].values, predictions)
    precision = precision_score(test_df['label'].values, predictions)
    recall = recall_score(test_df['label'].values, predictions)
    f1 = f1_score(test_df['label'].values, predictions)

    # Generate detailed classification report
    report = classification_report(test_df['label'].values, predictions, target_names=['Human', 'AI-Generated'])

    # Create confusion matrix
    cm = confusion_matrix(test_df['label'].values, predictions)

    # Print results
    print("\n" + "=" * 60)
    print(f"{'TEST EVALUATION RESULTS':^60}")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nDetailed Classification Report:")
    print(report)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Human', 'AI-Generated'],
                yticklabels=['Human', 'AI-Generated'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    # Save plot
    plt.savefig('test_confusion_matrix.png')
    print(f"Confusion matrix saved as 'test_confusion_matrix.png'")

    # Save detailed results to file
    with open('test_results.txt', 'w') as f:
        f.write(f"Test Evaluation Results\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of test samples: {len(test_df)}\n\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(report)

    # Save predictions to CSV
    test_df['predicted_label'] = predictions
    test_df['ai_probability'] = probabilities
    test_df.to_csv('test_predictions.csv', index=False)

    print(f"\nDetailed results saved to 'test_results.txt'")
    print(f"Predictions saved to 'test_predictions.csv'")
    print(f"\nEvaluation completed in {time.time() - start_time:.2f} seconds")

    # Analyze errors
    analyze_errors(test_df)

    return accuracy, precision, recall, f1


def analyze_errors(df):
    """Analyze and report on error patterns"""
    print("\nAnalyzing error patterns...")

    # Add error column
    df['error'] = (df['label'] != df['predicted_label']).astype(int)

    # False positives: Human texts classified as AI
    false_positives = df[(df['label'] == 0) & (df['predicted_label'] == 1)]

    # False negatives: AI texts classified as human
    false_negatives = df[(df['label'] == 1) & (df['predicted_label'] == 0)]

    print(f"Total errors: {df['error'].sum()} ({df['error'].mean():.2%} of test data)")
    print(f"False positives (Human classified as AI): {len(false_positives)} ({len(false_positives) / len(df):.2%})")
    print(f"False negatives (AI classified as Human): {len(false_negatives)} ({len(false_negatives) / len(df):.2%})")

    if len(false_positives) > 0 or len(false_negatives) > 0:
        # Save error samples to file
        false_positives.to_csv('false_positives.csv', index=False)
        false_negatives.to_csv('false_negatives.csv', index=False)
        print(f"Error samples saved to 'false_positives.csv' and 'false_negatives.csv'")

    # Calculate average text length for errors vs correct predictions
    df['text_length'] = df['text'].apply(len)

    avg_correct = df[df['error'] == 0]['text_length'].mean()
    avg_error = df[df['error'] == 1]['text_length'].mean()

    print(f"Average text length for correct predictions: {avg_correct:.1f} characters")
    print(f"Average text length for incorrect predictions: {avg_error:.1f} characters")


if __name__ == "__main__":
    # Path to test CSV file
    test_file = "C:/Users/jashw/Desktop/BERT/processed_data/test.csv"  # Update this path

    # Path to trained model
    model_dir = "C:/Users/jashw/Desktop/BERT/ensemble_models/roberta-base"

    # Run evaluation
    evaluate_model(test_file, model_dir)