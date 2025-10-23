import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import os

# Check if NLTK data is available locally
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class AITextDetector:
    def __init__(self, model_dir="ensemble_models/roberta-base"):
        """Initialize the detector with the trained model"""
        print("Loading model from", model_dir)

        # Load transformer model
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
        self.model.to(device)
        self.model.eval()

    def predict(self, text):
        """Make predictions on text input"""
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            prob = probs[0, 1].item()  # Probability of being AI-generated
            prediction = 1 if prob >= 0.5 else 0

        return prediction


def main():
    # Path to your trained model
    model_dir = "ensemble_models/roberta-base"

    # Check if model directory exists
    if not os.path.exists(model_dir):
        print(f"Error: Model directory '{model_dir}' not found.")
        print("Please make sure the trained model is in the correct location.")
        return

    # Initialize detector
    try:
        detector = AITextDetector(model_dir)
        print("AI Text Detector initialized successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure your model files are correctly saved.")
        return

    print("\n" + "=" * 70)
    print(" " * 20 + "AI TEXT DETECTION SYSTEM")
    print("=" * 70)
    print("\nThis system analyzes text to determine if it was written by a human or AI.")
    print("For best results, please enter at least 150 words.")
    print("\nType 'exit' to quit the program.")

    while True:
        print("\n" + "-" * 70)
        text = input("Enter the text to analyze (min 150 words recommended):\n")

        if text.lower() == 'exit':
            print("Exiting program. Goodbye!")
            break

        # Count words instead of characters
        word_count = len(text.split())

        if word_count < 50:
            print(f"Warning: Text is too short for reliable analysis ({word_count} words).")
            print("The model performs best with at least 150 words.")
            continue_anyway = input("Continue anyway? (y/n): ")
            if continue_anyway.lower() != 'y':
                continue

        # Make prediction
        prediction = detector.predict(text)

        print("\nANALYSIS RESULT:")
        print("-" * 70)
        print(f"This text is: {'AI-GENERATED' if prediction == 1 else 'HUMAN-WRITTEN'}")


if __name__ == "__main__":
    main()