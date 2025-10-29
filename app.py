"""
AI Text Detection - Web Service API
Flask application for deployment on Render
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import nltk

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

app = Flask(__name__)
CORS(app)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Global variables for model and tokenizer
model = None
tokenizer = None


def load_model():
    """Load the AI text detection model"""
    global model, tokenizer

    model_dir = "ensemble_models/roberta-base"

    if not os.path.exists(model_dir):
        return False, "Model directory not found"

    try:
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            local_files_only=True
        )
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
        return True, "Model loaded successfully"
    except Exception as e:
        print(f"Error loading model: {e}")
        return False, f"Error loading model: {str(e)}"


def predict_text(text, min_words=50):
    """Predict if text is AI-generated or human-written"""
    if model is None or tokenizer is None:
        return {
            'error': 'Model not loaded',
            'success': False
        }

    # Check word count
    word_count = len(text.split())
    if word_count < min_words:
        return {
            'error': f'Text too short. Minimum {min_words} words required. You provided {word_count} words.',
            'success': False,
            'word_count': word_count
        }

    try:
        # Tokenize
        inputs = tokenizer(
            text,
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
            ai_probability = probs[0, 1].item()
            prediction = 1 if ai_probability >= 0.5 else 0

        result = {
            'success': True,
            'prediction': 'AI-GENERATED' if prediction == 1 else 'HUMAN-WRITTEN',
            'prediction_label': prediction,
            'word_count': word_count,
            'text_length': len(text)
        }

        return result

    except Exception as e:
        return {
            'error': f'Prediction error: {str(e)}',
            'success': False
        }


@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'AI Text Detection API',
        'status': 'running',
        'model_loaded': model is not None,
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/predict': 'POST - Detect AI-generated text'
        }
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Predict endpoint for text detection"""
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided. Please send JSON with "text" field.',
                'success': False
            }), 400

        text = data['text']
        min_words = data.get('min_words', 50)

        result = predict_text(text, min_words)

        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400

    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}',
            'success': False
        }), 500


# Load model on startup
with app.app_context():
    success, message = load_model()
    if not success:
        print(f"WARNING: {message}")
        print("The API will run but predictions will fail until model is available.")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)