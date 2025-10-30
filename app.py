"""
AI Text Detection - Memory Optimized for Render Free Tier
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import gc

app = Flask(__name__)
CORS(app)

# Delay imports to save memory
model = None
tokenizer = None

MODEL_FILE_ID = "1ukeJocF4VUXf53l1xziC494iZFEK7ZDT"
MODEL_DIR = "roberta-model"
MODEL_FILE_PATH = os.path.join(MODEL_DIR, "model.safetensors")


def download_model_from_drive():
    """Download model file from Google Drive if not present"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if os.path.exists(MODEL_FILE_PATH):
        print(f"✓ Model file exists ({os.path.getsize(MODEL_FILE_PATH) / (1024**2):.2f} MB)")
        return True
    
    print("="*70)
    print("DOWNLOADING MODEL FROM GOOGLE DRIVE")
    print("="*70)
    
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
        gdown.download(url, MODEL_FILE_PATH, quiet=False)
        
        if os.path.exists(MODEL_FILE_PATH):
            print(f"✓ Downloaded ({os.path.getsize(MODEL_FILE_PATH) / (1024**2):.2f} MB)")
            return True
        return False
    except Exception as e:
        print(f"✗ Download error: {e}")
        return False


def load_model_lazy():
    """Lazy load model only when needed"""
    global model, tokenizer
    
    if model is not None:
        return True
    
    print("\n" + "="*70)
    print("LOADING MODEL (LAZY)")
    print("="*70)
    
    # Import only when needed to save memory
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    # Force CPU and optimize memory
    torch.set_num_threads(1)
    device = torch.device('cpu')
    
    try:
        # Download if needed
        if not download_model_from_drive():
            return False
        
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        print("✓ Tokenizer loaded")
        
        # Load model with minimal memory
        print("Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=2,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Load weights
        print("Loading fine-tuned weights...")
        state_dict = torch.load(MODEL_FILE_PATH, map_location=device, weights_only=True)
        
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval()
        
        # Clear cache
        gc.collect()
        
        print("✓ MODEL LOADED")
        print("="*70)
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def predict_text(text, min_words=50):
    """Predict if text is AI-generated"""
    
    # Lazy load model
    if not load_model_lazy():
        return {'error': 'Model loading failed', 'success': False}
    
    word_count = len(text.split())
    if word_count < min_words:
        return {
            'error': f'Minimum {min_words} words required. You have {word_count}.',
            'success': False,
            'word_count': word_count
        }
    
    try:
        import torch
        
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            ai_prob = probs[0, 1].item()
            prediction = 1 if ai_prob >= 0.5 else 0
        
        # Clear cache
        del inputs, outputs, logits, probs
        gc.collect()
        
        return {
            'success': True,
            'prediction': 'AI-GENERATED' if prediction == 1 else 'HUMAN-WRITTEN',
            'prediction_label': prediction,
            'word_count': word_count,
            'text_length': len(text),
            'ai_confidence': f"{ai_prob * 100:.2f}%",
            'human_confidence': f"{(1 - ai_prob) * 100:.2f}%"
        }
        
    except Exception as e:
        return {'error': f'Prediction error: {str(e)}', 'success': False}


@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'AI Text Detection API (Memory Optimized)',
        'status': 'running',
        'model_loaded': model is not None,
        'note': 'Model loads on first prediction request',
        'endpoints': {
            '/': 'GET - API info',
            '/health': 'GET - Health check',
            '/predict': 'POST - Detect AI text',
            '/load': 'GET - Pre-load model'
        }
    })


@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_file_exists': os.path.exists(MODEL_FILE_PATH)
    })


@app.route('/load')
def load():
    """Pre-load model"""
    success = load_model_lazy()
    return jsonify({
        'success': success,
        'model_loaded': model is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Predict endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided',
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


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    
    # DON'T load model on startup - save memory
    print("\n" + "="*70)
    print("AI TEXT DETECTION API - MEMORY OPTIMIZED")
    print("Model will load on first request")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=False)
