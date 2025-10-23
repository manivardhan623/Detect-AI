import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import classification_report, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from tqdm.auto import tqdm
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Download NLTK resources
nltk.download('punkt', quiet=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Set seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


set_seed(42)

# Create directory for models
os.makedirs("ensemble_models", exist_ok=True)
os.makedirs("results", exist_ok=True)


class TextDataset(Dataset):
    """Dataset for transformer models"""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


def extract_stylometric_features(texts):
    """Extract detailed stylometric features from texts"""
    features = []

    for text in tqdm(texts, desc="Extracting stylometric features"):
        # Tokenize
        sentences = sent_tokenize(text)
        words = word_tokenize(text)

        # Basic counts
        char_count = len(text)
        word_count = len(words)
        sentence_count = len(sentences)

        # Average lengths
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        avg_sentence_length_chars = np.mean([len(s) for s in sentences]) if sentences else 0
        avg_sentence_length_words = np.mean([len(s.split()) for s in sentences]) if sentences else 0

        # Variance in lengths
        word_length_variance = np.var([len(w) for w in words]) if len(words) > 1 else 0
        sentence_length_variance = np.var([len(s.split()) for s in sentences]) if len(sentences) > 1 else 0

        # Character distributions
        uppercase_ratio = sum(1 for c in text if c.isupper()) / (char_count + 1e-10)
        digit_ratio = sum(1 for c in text if c.isdigit()) / (char_count + 1e-10)
        whitespace_ratio = sum(1 for c in text if c.isspace()) / (char_count + 1e-10)
        punctuation_ratio = sum(1 for c in text if c in ".,;:!?\"'()[]{}") / (char_count + 1e-10)

        # Word patterns
        unique_words = len(set([w.lower() for w in words]))
        lexical_diversity = unique_words / (word_count + 1e-10)  # Type-Token Ratio

        # Sentence complexity
        complex_sentence_ratio = sum(1 for s in sentences if len(s.split()) > 20) / (sentence_count + 1e-10)

        # Function word usage
        function_words = ["the", "of", "and", "a", "to", "in", "is", "that", "it", "for"]
        function_word_ratio = sum(1 for w in words if w.lower() in function_words) / (word_count + 1e-10)

        # Readability approximation
        if sentence_count > 0 and word_count > 0:
            avg_words_per_sentence = word_count / sentence_count
            readability_score = 206.835 - 1.015 * avg_words_per_sentence
        else:
            readability_score = 0

        # N-gram repetition
        if len(words) > 1:
            bigrams = list(zip(words, words[1:]))
            unique_bigrams = len(set(bigrams))
            bigram_diversity = unique_bigrams / (len(bigrams) + 1e-10)
        else:
            bigram_diversity = 0

        feature_vector = [
            char_count, word_count, sentence_count,
            avg_word_length, avg_sentence_length_chars, avg_sentence_length_words,
            word_length_variance, sentence_length_variance,
            uppercase_ratio, digit_ratio, whitespace_ratio, punctuation_ratio,
            lexical_diversity, complex_sentence_ratio, function_word_ratio,
            readability_score, bigram_diversity
        ]

        features.append(feature_vector)

    return np.array(features)


def calculate_perplexity_features(texts, model_name="distilgpt2"):
    """Calculate perplexity-based features using a language model"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    features = []
    batch_size = 8  # Adjust based on memory constraints

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Calculating perplexity features"):
            batch_texts = texts[i:i + batch_size]
            batch_features = []

            for text in batch_texts:
                # Tokenize text
                encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                input_ids = encodings.input_ids.to(device)

                try:
                    # Get model outputs
                    outputs = model(input_ids, labels=input_ids)
                    neg_log_likelihood = outputs.loss.item()

                    # Calculate perplexity
                    perplexity = torch.exp(torch.tensor(neg_log_likelihood)).item()

                    # Get logits for token probability statistics
                    logits = outputs.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = input_ids[..., 1:].contiguous()

                    # Get probabilities for the correct tokens
                    probs = torch.softmax(shift_logits, dim=-1)
                    token_probs = probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

                    # Calculate statistics
                    mean_prob = token_probs.mean().item()
                    std_prob = token_probs.std().item() if token_probs.size(0) > 1 else 0
                    min_prob = token_probs.min().item() if token_probs.numel() > 0 else 0

                    # Calculate entropy
                    entropy = -torch.mean(torch.log(token_probs + 1e-10)).item()
                except Exception as e:
                    print(f"Error processing text for perplexity: {e}")
                    perplexity = 1000.0  # Default high perplexity
                    mean_prob = 0.0
                    std_prob = 0.0
                    min_prob = 0.0
                    entropy = 0.0

                # Add features
                batch_features.append([perplexity, mean_prob, std_prob, min_prob, entropy])

            features.extend(batch_features)

    # Clean up
    del model, tokenizer
    torch.cuda.empty_cache()

    return np.array(features)


def get_transformer_embeddings(texts, model_name, pooling="cls"):
    """Extract embeddings from transformer models"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    embeddings = []
    batch_size = 8  # Adjust based on available memory

    for i in tqdm(range(0, len(texts), batch_size), desc=f"Extracting {model_name} embeddings"):
        batch_texts = texts[i:i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract embeddings based on pooling strategy
        if pooling == "cls":
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        elif pooling == "mean":
            # Mean pooling - take attention mask into account for averaging
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()

        embeddings.append(batch_embeddings)

    # Clear memory
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return np.vstack(embeddings)


def train_transformer_model(train_texts, train_labels, val_texts, val_labels, model_name, epochs=3):
    """Train a transformer model for classification"""
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1 * total_steps,
        num_training_steps=total_steps
    )

    # Training loop
    best_val_auc = 0
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(device)

            # Forward pass
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} - Validation"):
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
                labels = batch['label'].to(device)

                # Forward pass
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                val_loss += loss.item()

                # Get predictions
                probs = torch.softmax(logits, dim=1)
                all_preds.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        val_auc = roc_auc_score(all_labels, all_preds)

        print(f"Epoch {epoch + 1}/{epochs}:")
        print(f"  Train Loss: {train_loss / len(train_loader):.4f}")
        print(f"  Val Loss: {val_loss / len(val_loader):.4f}")
        print(f"  Val AUC: {val_auc:.4f}")

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            model_path = f"ensemble_models/{model_name.replace('/', '-')}"
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            print(f"  New best model saved to {model_path}")

    # Load best model
    model = AutoModelForSequenceClassification.from_pretrained(f"ensemble_models/{model_name.replace('/', '-')}")
    model.to(device)

    return model, tokenizer


def get_transformer_predictions(model, tokenizer, texts):
    """Get predictions from transformer model"""
    model.eval()
    predictions = []
    batch_size = 16

    for i in tqdm(range(0, len(texts), batch_size), desc="Getting predictions"):
        batch_texts = texts[i:i + batch_size]

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
            batch_preds = probs[:, 1].cpu().numpy()

        predictions.extend(batch_preds)

    return np.array(predictions)


# Main function - modified to use pre-split datasets
def train_and_evaluate_advanced_ensemble_presplit(train_path, val_path, test_path):
    """Train and evaluate the advanced ensemble model using pre-split datasets"""
    print("Starting advanced ensemble training process...")

    # Load pre-split datasets
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Check if columns need renaming
    if 'generated' in train_df.columns and 'text' in train_df.columns:
        train_df = train_df.rename(columns={'generated': 'label'})
        val_df = val_df.rename(columns={'generated': 'label'})
        test_df = test_df.rename(columns={'generated': 'label'})

    print(f"Train set: {len(train_df)}, Validation set: {len(val_df)}, Test set: {len(test_df)}")
    print(f"Total samples: {len(train_df) + len(val_df) + len(test_df)}")

    # ===== LEVEL 1: Train base transformer models =====
    # Replace problematic DeBERTa model with well-supported models
    transformer_models = [
        "roberta-base",  # Stable, well-supported model
        "distilroberta-base",  # Lighter version of RoBERTa
        "xlnet-base-cased"  # Alternative architecture
    ]

    trained_models = {}
    for model_name in transformer_models:
        print(f"\nTraining {model_name}...")
        model, tokenizer = train_transformer_model(
            train_df['text'].values,
            train_df['label'].values,
            val_df['text'].values,
            val_df['label'].values,
            model_name
        )
        trained_models[model_name] = (model, tokenizer)

    # ===== LEVEL 2: Extract features for meta-model =====
    print("\nExtracting features for meta-model...")

    # Get predictions from transformer models
    train_transformer_preds = {}
    val_transformer_preds = {}
    test_transformer_preds = {}

    for model_name, (model, tokenizer) in trained_models.items():
        print(f"\nGetting predictions from {model_name}...")

        train_preds = get_transformer_predictions(model, tokenizer, train_df['text'].values)
        val_preds = get_transformer_predictions(model, tokenizer, val_df['text'].values)
        test_preds = get_transformer_predictions(model, tokenizer, test_df['text'].values)

        train_transformer_preds[model_name] = train_preds
        val_transformer_preds[model_name] = val_preds
        test_transformer_preds[model_name] = test_preds

    # Extract embeddings - use roberta-base for embeddings
    print("\nExtracting embeddings...")
    train_embeddings = get_transformer_embeddings(
        train_df['text'].values,
        "roberta-base",
        pooling="mean"
    )
    val_embeddings = get_transformer_embeddings(
        val_df['text'].values,
        "roberta-base",
        pooling="mean"
    )
    test_embeddings = get_transformer_embeddings(
        test_df['text'].values,
        "roberta-base",
        pooling="mean"
    )

    # Use PCA to reduce dimensionality of embeddings
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50)
    train_embeddings_reduced = pca.fit_transform(train_embeddings)
    val_embeddings_reduced = pca.transform(val_embeddings)
    test_embeddings_reduced = pca.transform(test_embeddings)

    # Extract stylometric features
    print("\nExtracting stylometric features...")
    train_stylometric = extract_stylometric_features(train_df['text'].values)
    val_stylometric = extract_stylometric_features(val_df['text'].values)
    test_stylometric = extract_stylometric_features(test_df['text'].values)

    # Extract perplexity features using distilgpt2 (smaller and faster than full gpt2)
    print("\nCalculating perplexity features...")
    train_perplexity = calculate_perplexity_features(train_df['text'].values, model_name="distilgpt2")
    val_perplexity = calculate_perplexity_features(val_df['text'].values, model_name="distilgpt2")
    test_perplexity = calculate_perplexity_features(test_df['text'].values, model_name="distilgpt2")

    # Standardize stylometric and perplexity features
    scaler = StandardScaler()
    train_stylometric_scaled = scaler.fit_transform(train_stylometric)
    val_stylometric_scaled = scaler.transform(val_stylometric)
    test_stylometric_scaled = scaler.transform(test_stylometric)

    scaler_perplexity = StandardScaler()
    train_perplexity_scaled = scaler_perplexity.fit_transform(train_perplexity)
    val_perplexity_scaled = scaler_perplexity.transform(val_perplexity)
    test_perplexity_scaled = scaler_perplexity.transform(test_perplexity)

    # Combine all features for meta-model
    train_meta_features = np.hstack([
        np.column_stack(list(train_transformer_preds.values())),
        train_embeddings_reduced,
        train_stylometric_scaled,
        train_perplexity_scaled
    ])

    val_meta_features = np.hstack([
        np.column_stack(list(val_transformer_preds.values())),
        val_embeddings_reduced,
        val_stylometric_scaled,
        val_perplexity_scaled
    ])

    test_meta_features = np.hstack([
        np.column_stack(list(test_transformer_preds.values())),
        test_embeddings_reduced,
        test_stylometric_scaled,
        test_perplexity_scaled
    ])

    print(f"Meta-model feature shape: {train_meta_features.shape}")

    # ===== LEVEL 3: Train meta-models =====
    print("\nTraining meta-models...")

    meta_models = [
        xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            scale_pos_weight=1,
            random_state=42
        ),
        lgb.LGBMClassifier(
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),
        CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            loss_function='Logloss',
            verbose=0,
            random_seed=42
        )
    ]

    meta_preds = []
    for i, model in enumerate(meta_models):
        print(f"\nTraining meta-model {i + 1}...")
        model.fit(train_meta_features, train_df['label'].values)

        # Get predictions on validation set to find best threshold
        val_probs = model.predict_proba(val_meta_features)[:, 1]

        # Find optimal threshold using validation set
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(val_df['label'].values, val_probs)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if len(thresholds) > optimal_idx else 0.5

        print(f"  Optimal threshold: {optimal_threshold:.4f}")

        # Get predictions on test set
        test_probs = model.predict_proba(test_meta_features)[:, 1]
        meta_preds.append(test_probs)

        # Evaluate individual meta-model
        test_preds_binary = (test_probs >= optimal_threshold).astype(int)
        print("\nMeta-model performance:")
        print(classification_report(test_df['label'].values, test_preds_binary))
        print(f"AUC: {roc_auc_score(test_df['label'].values, test_probs):.4f}")

    # ===== LEVEL 4: Ensemble of meta-models =====
    print("\nCreating final ensemble...")

    # Average predictions from all meta-models
    ensemble_probs = np.mean(meta_preds, axis=0)

    # Find optimal threshold for ensemble
    ensemble_val_probs = np.mean([
        model.predict_proba(val_meta_features)[:, 1] for model in meta_models
    ], axis=0)

    precision, recall, thresholds = precision_recall_curve(val_df['label'].values, ensemble_val_probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if len(thresholds) > optimal_idx else 0.5

    # Final ensemble predictions
    ensemble_preds = (ensemble_probs >= optimal_threshold).astype(int)

    # Evaluate final ensemble
    print("\nFinal ensemble performance:")
    print(classification_report(test_df['label'].values, ensemble_preds))
    print(f"AUC: {roc_auc_score(test_df['label'].values, ensemble_probs):.4f}")

    # Plot confusion matrix
    cm = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            cm[i, j] = ((test_df['label'].values == i) & (ensemble_preds == j)).sum()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Final Ensemble')
    plt.savefig('results/confusion_matrix.png')

    # Save feature importance
    for i, model in enumerate(meta_models[:2]):  # XGBoost and LightGBM have feature_importances_
        plt.figure(figsize=(12, 8))
        feature_names = [f"model_{j}" for j in range(len(transformer_models))] + \
                        [f"emb_{j}" for j in range(train_embeddings_reduced.shape[1])] + \
                        [f"style_{j}" for j in range(train_stylometric_scaled.shape[1])] + \
                        [f"perp_{j}" for j in range(train_perplexity_scaled.shape[1])]

        importances = pd.Series(model.feature_importances_, index=feature_names)
        importances = importances.sort_values(ascending=False)
        importances[:30].plot(kind='bar')
        plt.title(f'Feature Importance - Meta-model {i + 1}')
        plt.tight_layout()
        plt.savefig(f'results/feature_importance_{i + 1}.png')

    # Save models for future use
    for i, model in enumerate(meta_models):
        import joblib
        joblib.dump(model, f'ensemble_models/meta_model_{i + 1}.pkl')

    # Save scalers
    import joblib
    joblib.dump(scaler, 'ensemble_models/stylometric_scaler.pkl')
    joblib.dump(scaler_perplexity, 'ensemble_models/perplexity_scaler.pkl')
    joblib.dump(pca, 'ensemble_models/embedding_pca.pkl')

    print("\nAll models saved. Training complete!")

    return {
        "transformer_models": trained_models,
        "meta_models": meta_models,
        "scalers": {
            "stylometric": scaler,
            "perplexity": scaler_perplexity
        },
        "pca": pca,
        "threshold": optimal_threshold,
        "auc": roc_auc_score(test_df['label'].values, ensemble_probs)
    }


# Function for inference with the ensemble
def predict_with_ensemble(texts, ensemble_path="ensemble_models"):
    """Use the trained ensemble to make predictions on new texts"""
    import joblib

    # Load meta-models
    meta_models = []
    for i in range(1, 4):  # We trained 3 meta-models
        model_path = f"{ensemble_path}/meta_model_{i}.pkl"
        if os.path.exists(model_path):
            meta_models.append(joblib.load(model_path))

    # Load transformer models
    transformer_models = {}
    for model_name in ["roberta-base", "distilroberta-base", "xlnet-base-cased"]:
        model_dir = f"{ensemble_path}/{model_name.replace('/', '-')}"
        if os.path.exists(model_dir):
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model.to(device)
            transformer_models[model_name] = (model, tokenizer)

    # Load scalers and PCA
    stylometric_scaler = joblib.load(f"{ensemble_path}/stylometric_scaler.pkl")
    perplexity_scaler = joblib.load(f"{ensemble_path}/perplexity_scaler.pkl")
    pca = joblib.load(f"{ensemble_path}/embedding_pca.pkl")

    # Get transformer predictions
    transformer_preds = {}
    for model_name, (model, tokenizer) in transformer_models.items():
        preds = get_transformer_predictions(model, tokenizer, texts)
        transformer_preds[model_name] = preds

    # Extract embeddings
    embeddings = get_transformer_embeddings(texts, "roberta-base", pooling="mean")
    embeddings_reduced = pca.transform(embeddings)

    # Extract stylometric features
    stylometric = extract_stylometric_features(texts)
    stylometric_scaled = stylometric_scaler.transform(stylometric)

    # Extract perplexity features
    perplexity = calculate_perplexity_features(texts, model_name="distilgpt2")
    perplexity_scaled = perplexity_scaler.transform(perplexity)

    # Combine features
    meta_features = np.hstack([
        np.column_stack(list(transformer_preds.values())),
        embeddings_reduced,
        stylometric_scaled,
        perplexity_scaled
    ])

    # Get predictions from meta-models
    all_preds = []
    for model in meta_models:
        preds = model.predict_proba(meta_features)[:, 1]
        all_preds.append(preds)

    # Ensemble predictions
    ensemble_probs = np.mean(all_preds, axis=0)
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)

    return ensemble_preds, ensemble_probs


# Main execution
if __name__ == "__main__":
    # Replace these with your actual file paths
    train_path = 'C:/Users/jashw/Desktop/BERT/processed_data/train.csv'
    val_path = 'C:/Users/jashw/Desktop/BERT/processed_data/validation.csv'
    test_path = 'C:/Users/jashw/Desktop/BERT/processed_data/test.csv'

    # Train the advanced ensemble
    trained_ensemble = train_and_evaluate_advanced_ensemble_presplit(
        train_path, val_path, test_path
    )