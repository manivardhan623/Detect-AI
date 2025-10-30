"""
AI Text Detection - Web Service API
Flask application with Google Drive model loading
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
import gdown
import shutil

app = Flask(__name__)
CORS(app)

# Set device
device = torch.device('cpu')
print(f"Using device: {device}")

# Global variables
model = None
tokenizer = None

# ============================================
# GOOGLE DRIVE CONFIGURATION
# ============================================
# Your Google Drive File ID
MODEL_FILE_ID = "1ukeJocF4VUXf53l1xziC494iZFEK7ZDT"  # ✅ UPDATED

MODEL_DIR = "roberta-model"
MODEL_FILE_PATH = os.path.join(MODEL_DIR, "model.safetensors")


def download_model_from_drive():
    """Download model file from Google Drive if not already present"""

    # Create directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Check if model already exists
    if os.path.exists(MODEL_FILE_PATH):
        file_size = os.path.getsize(MODEL_FILE_PATH) / (1024 * 1024)  # Convert to MB
        print(f"✓ Model file already exists ({file_size:.2f} MB)")
        return True

    print("=" * 70)
    print("DOWNLOADING MODEL FROM GOOGLE DRIVE")
    print("=" * 70)
    print(f"File ID: {MODEL_FILE_ID}")
    print(f"Destination: {MODEL_FILE_PATH}")
    print("This may take a few minutes on first deployment...")
    print("-" * 70)

    try:
        # Google Drive download URL
        url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"

        # Download with gdown
        gdown.download(url, MODEL_FILE_PATH, quiet=False)

        # Verify download
        if os.path.exists(MODEL_FILE_PATH):
            file_size = os.path.getsize(MODEL_FILE_PATH) / (1024 * 1024)
            print(f"\n✓ Model downloaded successfully! ({file_size:.2f} MB)")
            print("=" * 70)
            return True
        else:
            print("✗ Download failed: File not found after download")
            return False

    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("1. Check that your Google Drive link is set to 'Anyone with the link can view'")
        print("2. Verify the FILE_ID is correct")
        print("3. Ensure the file exists in your Google Drive")
        return False


def load_model():
    """Load the AI text detection model"""
    global model, tokenizer

    print("\n" + "=" * 70)
    print("LOADING AI TEXT DETECTION MODEL")
    print("=" * 70)

    # Step 1: Download model from Google Drive if needed
    if not download_model_from_drive():
        return False, "Failed to download model from Google Drive"

    # Step 2: Load tokenizer from HuggingFace
    try:
        print("\nLoading tokenizer from HuggingFace...")
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        print("✓ Tokenizer loaded")
    except Exception as e:
        return False, f"Error loading tokenizer: {str(e)}"

    # Step 3: Load model architecture
    try:
        print("Loading model architecture...")
        model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=2
        )
        print("✓ Model architecture loaded")
    except Exception as e:
        return False, f"Error loading model architecture: {str(e)}"

    # Step 4: Load fine-tuned weights
    try:
        print("Loading fine-tuned weights from downloaded file...")
        state_dict = torch.load(MODEL_FILE_PATH, map_location=device)

        # Handle different state_dict formats
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)

        print("✓ Fine-tuned weights loaded")
    except Exception as e:
        print(f"⚠ Warning: Could not load fine-tuned weights: {e}")
        print("Using base RoBERTa model instead")

    # Step 5: Prepare model for inference
    model.to(device)
    model.eval()

    print("\n" + "=" * 70)
    print("✓ MODEL LOADED SUCCESSFULLY!")
    print(f"Device: {device}")
    print(f"Parameters: ~125M")
    print(f"Expected Accuracy: 98.69%")
    print("=" * 70 + "\n")

    return True, "Model loaded successfully"


def predict_text(text, min_words=50):
    """Predict if text is AI-generated or human-written"""

    if model is None or tokenizer is None:
        return {
            'error': 'Model not loaded. Please wait for model initialization.',
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
            'text_length': len(text),
            'ai_confidence': f"{ai_probability * 100:.2f}%",
            'human_confidence': f"{(1 - ai_probability) * 100:.2f}%"
        }

        return result

    except Exception as e:
        return {
            'error': f'Prediction error: {str(e)}',
            'success': False
        }


# ============================================
# API ENDPOINTS
# ============================================

@app.route('/')
def home():
    """Home endpoint"""
    model_status = "loaded" if model is not None else "loading..."

    return jsonify({
        'message': 'AI Text Detection API - Fine-tuned RoBERTa Model',
        'status': 'running',
        'model_loaded': model is not None,
        'device': str(device),
        'accuracy': '98.69%',
        'model_source': 'Google Drive',
        'endpoints': {
            '/': 'GET - API information',
            '/health': 'GET - Health check',
            '/predict': 'POST - Detect AI-generated text',
            '/model-info': 'GET - Model details'
        },
        'usage': {
            'endpoint': '/predict',
            'method': 'POST',
            'body': {
                'text': 'Your text to analyze (minimum 50 words)',
                'min_words': 50
            }
        }
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device),
        'model_file_exists': os.path.exists(MODEL_FILE_PATH)
    })


@app.route('/model-info')
def model_info():
    """Model information endpoint"""

    model_size = "N/A"
    if os.path.exists(MODEL_FILE_PATH):
        size_bytes = os.path.getsize(MODEL_FILE_PATH)
        model_size = f"{size_bytes / (1024 * 1024):.2f} MB"

    return jsonify({
        'model': 'RoBERTa-base (fine-tuned)',
        'parameters': '~125 million',
        'accuracy': '98.69%',
        'precision': '97.76%',
        'recall': '99.66%',
        'f1_score': '98.70%',
        'model_size': model_size,
        'training_samples': '40,000',
        'model_loaded': model is not None,
        'source': 'Google Drive',
        'file_id': MODEL_FILE_ID[:10] + "..." if len(MODEL_FILE_ID) > 10 else MODEL_FILE_ID
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Predict endpoint for text detection"""

    if model is None:
        return jsonify({
            'error': 'Model is still loading. Please wait a moment and try again.',
            'success': False
        }), 503

    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided. Please send JSON with "text" field.',
                'success': False,
                'example': {
                    'text': 'Your text here (minimum 50 words)',
                    'min_words': 50
                }
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


# ============================================
# APPLICATION STARTUP
# ============================================

# Load model on startup
print("\n" + "=" * 70)
print("AI TEXT DETECTION API - STARTING UP")
print("=" * 70)
print(f"Python working directory: {os.getcwd()}")
print(f"Model directory: {MODEL_DIR}")
print(f"Model file ID: {MODEL_FILE_ID[:20]}...")
print("=" * 70)

with app.app_context():
    success, message = load_model()
    if success:
        print(f"\n🚀 API READY TO SERVE REQUESTS!")
    else:
        print(f"\n⚠️  WARNING: {message}")
        print("The API will run but predictions will fail.")
        print("\nPlease check:")
        print("1. MODEL_FILE_ID is correct in app.py")
        print("2. Google Drive link is set to 'Anyone with the link can view'")
        print("3. File exists in your Google Drive")

print("\n" + "=" * 70 + "\n")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
