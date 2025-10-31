"""
AI Text Detection - Ultra Minimal for Render Free Tier
Uses base RoBERTa (no fine-tuned weights)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = None
tokenizer = None


def load_model():
    """Load base RoBERTa model (no fine-tuned weights)"""
    global model, tokenizer
    
    if model is not None:
        return True
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        print("Loading base RoBERTa model (no fine-tuning)...")
        
        tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilroberta-base",
            num_labels=2
        )
        model.eval()
        
        print("✓ Model loaded (base model only)")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


@app.route('/')
def home():
    return jsonify({
        'message': 'AI Text Detection API',
        'status': 'running',
        'note': 'Using base DistilRoBERTa (smaller, faster, lower accuracy)',
        'model': 'distilroberta-base'
    })


@app.route('/predict', methods=['POST'])
def predict():
    if not load_model():
        return jsonify({'error': 'Model failed to load', 'success': False}), 503
    
    try:
        import torch
        
        data = request.get_json()
        text = data.get('text', '')
        
        if not text or len(text.split()) < 50:
            return jsonify({
                'error': 'Minimum 50 words required',
                'success': False
            }), 400
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            ai_prob = probs[0, 1].item()
            prediction = 1 if ai_prob >= 0.5 else 0
        
        return jsonify({
            'success': True,
            'prediction': 'AI-GENERATED' if prediction == 1 else 'HUMAN-WRITTEN',
            'ai_confidence': f"{ai_prob * 100:.2f}%",
            'word_count': len(text.split())
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


if __name__ == '__main__':
    import os
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 10000)))
