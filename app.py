"""
AI Text Detection - Flask App with Frontend
Serves HTML page + API endpoints
"""

from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

model = None
tokenizer = None


def load_model():
    """Load base RoBERTa model"""
    global model, tokenizer
    
    if model is not None:
        return True
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        print("Loading DistilRoBERTa model...")
        
        tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilroberta-base",
            num_labels=2
        )
        model.eval()
        
        print("✓ Model loaded")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


# HTML Frontend Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Detect-AI | AI Text Detection</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #0f172a;
            line-height: 1.6;
            min-height: 100vh;
            padding: 2rem;
        }
        
        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            border-radius: 24px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
            padding: 3rem;
        }
        
        h1 {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        .subtitle {
            text-align: center;
            color: #64748b;
            font-size: 1.2rem;
            margin-bottom: 3rem;
        }
        
        .input-group {
            margin-bottom: 2rem;
        }
        
        label {
            display: block;
            font-weight: 600;
            margin-bottom: 0.75rem;
            font-size: 1.1rem;
        }
        
        .word-counter {
            float: right;
            color: #64748b;
            font-size: 0.9rem;
        }
        
        .word-count {
            font-weight: 600;
            color: #6366f1;
        }
        
        textarea {
            width: 100%;
            min-height: 250px;
            padding: 1.5rem;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            font-size: 1rem;
            font-family: inherit;
            resize: vertical;
            transition: all 0.3s;
        }
        
        textarea:focus {
            outline: none;
            border-color: #6366f1;
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1);
        }
        
        .btn-group {
            display: flex;
            gap: 1rem;
            margin-top: 1.5rem;
        }
        
        .btn {
            flex: 1;
            padding: 1.2rem 2rem;
            border: none;
            border-radius: 12px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.75rem;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: white;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        }
        
        .btn-primary:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(99, 102, 241, 0.4);
        }
        
        .btn-primary:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .btn-secondary {
            background: white;
            color: #0f172a;
            border: 2px solid #e2e8f0;
        }
        
        .btn-secondary:hover {
            border-color: #6366f1;
            color: #6366f1;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 2rem;
        }
        
        .loading.active {
            display: block;
        }
        
        .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid #e2e8f0;
            border-top-color: #6366f1;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .result-card {
            display: none;
            margin-top: 2rem;
            padding: 2rem;
            border-radius: 16px;
            animation: slideIn 0.5s ease-out;
        }
        
        .result-card.active {
            display: block;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .result-card.human {
            background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
            border: 2px solid #6ee7b7;
            color: #065f46;
        }
        
        .result-card.ai {
            background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
            border: 2px solid #fca5a5;
            color: #991b1b;
        }
        
        .result-card.error {
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            border: 2px solid #fcd34d;
            color: #92400e;
        }
        
        .result-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        
        .result-title {
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 1rem;
        }
        
        .result-details {
            font-size: 1rem;
            opacity: 0.9;
        }
        
        .footer {
            text-align: center;
            margin-top: 3rem;
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Detect-AI</h1>
        <p class="subtitle">AI Text Detection Powered by Machine Learning</p>
        
        <div class="input-group">
            <label>
                Enter text to analyze
                <span class="word-counter">
                    <span class="word-count" id="wordCount">0 words</span>
                </span>
            </label>
            <textarea 
                id="textInput" 
                placeholder="Paste or type at least 50 words here...

For best results, enter at least 150 words. The model will analyze linguistic patterns and writing style to determine whether the text was written by a human or generated by AI."
            ></textarea>
        </div>
        
        <div class="btn-group">
            <button class="btn btn-primary" id="detectBtn" onclick="analyzeText()" disabled>
                <i class="fas fa-magic"></i>
                Detect AI Text
            </button>
            <button class="btn btn-secondary" onclick="clearText()">
                <i class="fas fa-eraser"></i>
                Clear
            </button>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div>Analyzing your text...</div>
        </div>
        
        <div class="result-card" id="resultCard">
            <div class="result-icon" id="resultIcon"></div>
            <div class="result-title" id="resultTitle"></div>
            <div class="result-details" id="resultDetails"></div>
        </div>
    </div>
    
    <div class="footer">
        <p>Built with ❤️ by @jaswanth-140 & @manivardhan623</p>
        <p>Powered by DistilRoBERTa transformer model</p>
    </div>

    <script>
        const textInput = document.getElementById('textInput');
        const wordCountEl = document.getElementById('wordCount');
        const detectBtn = document.getElementById('detectBtn');
        const loading = document.getElementById('loading');
        const resultCard = document.getElementById('resultCard');
        
        textInput.addEventListener('input', function() {
            const text = this.value.trim();
            const words = text.split(/\\s+/).filter(word => word.length > 0);
            const count = words.length;
            
            wordCountEl.textContent = `${count} word${count !== 1 ? 's' : ''}`;
            detectBtn.disabled = count < 50;
        });
        
        async function analyzeText() {
            const text = textInput.value.trim();
            
            if (!text || text.split(/\\s+/).length < 50) {
                showResult('Please enter at least 50 words.', 'error');
                return;
            }
            
            detectBtn.disabled = true;
            loading.classList.add('active');
            resultCard.classList.remove('active');
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    const isAI = data.prediction === 'AI-GENERATED';
                    document.getElementById('resultIcon').textContent = isAI ? '🤖' : '✍️';
                    document.getElementById('resultTitle').textContent = data.prediction;
                    document.getElementById('resultDetails').innerHTML = `
                        <strong>Confidence:</strong> ${data.ai_confidence}<br>
                        <strong>Word Count:</strong> ${data.word_count}
                    `;
                    resultCard.className = `result-card ${isAI ? 'ai' : 'human'} active`;
                } else {
                    showResult(data.error || 'Analysis failed', 'error');
                }
            } catch (error) {
                showResult('Unable to connect to API. Please try again.', 'error');
            } finally {
                loading.classList.remove('active');
                detectBtn.disabled = false;
            }
        }
        
        function showResult(message, type) {
            document.getElementById('resultIcon').textContent = '⚠️';
            document.getElementById('resultTitle').textContent = type === 'error' ? 'Error' : 'Notice';
            document.getElementById('resultDetails').textContent = message;
            resultCard.className = `result-card ${type} active`;
        }
        
        function clearText() {
            textInput.value = '';
            wordCountEl.textContent = '0 words';
            detectBtn.disabled = true;
            resultCard.classList.remove('active');
        }
    </script>
</body>
</html>
"""


@app.route('/')
def home():
    """Serve the HTML frontend"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api')
def api_info():
    """API information endpoint"""
    return jsonify({
        'message': 'AI Text Detection API',
        'status': 'running',
        'model': 'distilroberta-base',
        'endpoints': {
            '/': 'Frontend HTML page',
            '/api': 'API information',
            '/predict': 'POST - Detect AI text'
        }
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Predict endpoint"""
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
    port = int(os.environ.get('PORT', 10000))
    print("\n" + "="*70)
    print("AI TEXT DETECTION - STARTING")
    print("Visit the URL to see the frontend!")
    print("="*70 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False)
