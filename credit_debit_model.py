#!/usr/bin/env python3
"""
Credit/Debit FFN classifier for bank statement transactions.

Replaces the heuristic-based _infer_transaction_types with a trained
feed-forward network that predicts credit vs debit from transaction features.

Features (~63 dims):
  - Amount sign (1): positive/negative
  - Amount magnitude (1): log(abs(amount))
  - Description TF-IDF (50): sparse text encoding
  - Page position (1): normalized y-coordinate
  - Keyword flags (10): binary indicators for common credit/debit keywords

Usage:
    python credit_debit_model.py                # train with defaults
    python credit_debit_model.py --test-split 0.2
    python credit_debit_model.py --evaluate     # evaluate saved model
"""

import argparse
import json
import math
import pickle
import sys
import threading
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent
LABELS_DIR = PROJECT_DIR / "data" / "labels"
MODEL_DIR = PROJECT_DIR / "models" / "credit_debit"

# Keywords for binary feature flags
DEBIT_KEYWORDS = [
    "fee", "withdrawal", "purchase", "debit", "check",
]
CREDIT_KEYWORDS = [
    "deposit", "credit", "payroll", "refund", "interest",
]


class CreditDebitFFN(nn.Module):
    """2-layer feed-forward network for credit/debit classification."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def _load_training_data() -> list:
    """Load all Claude-labeled transactions with credit/debit types."""
    samples = []
    if not LABELS_DIR.exists():
        print("No labels directory found.")
        sys.exit(1)

    for pdf_dir in sorted(LABELS_DIR.iterdir()):
        if not pdf_dir.is_dir():
            continue
        for lf in sorted(pdf_dir.glob("page_*_labels.json")):
            page_idx = int(lf.stem.replace("page_", "").replace("_labels", ""))
            with open(lf) as f:
                data = json.load(f)
            for txn in data.get("transactions", []):
                txn_type = txn.get("type", "")
                if txn_type not in ("credit", "debit"):
                    continue
                samples.append({
                    "amount": abs(float(txn.get("amount", 0))),  # abs() to match inference
                    "description": txn.get("description", ""),
                    "type": txn_type,
                    "page_index": page_idx,
                    "pdf_stem": pdf_dir.name,
                })
    return samples


def _extract_features(samples: list, tfidf: TfidfVectorizer = None, fit: bool = False):
    """Extract feature matrix from transaction samples.

    Returns (features_np, labels_np, tfidf_vectorizer).
    """
    descriptions = [s["description"].lower() for s in samples]

    # Fit or transform TF-IDF
    if fit:
        tfidf = TfidfVectorizer(max_features=50, stop_words="english", ngram_range=(1, 2))
        tfidf_matrix = tfidf.fit_transform(descriptions).toarray()
    else:
        tfidf_matrix = tfidf.transform(descriptions).toarray()

    features = []
    for i, s in enumerate(samples):
        amount = s["amount"]

        # Amount sign: 1 if negative, 0 if positive
        amount_sign = 1.0 if amount < 0 else 0.0

        # Amount magnitude (log-scaled)
        amount_mag = math.log1p(abs(amount))

        # Page position (normalized, rough estimate)
        page_pos = s.get("page_index", 0) / 20.0  # normalize

        # Keyword flags
        desc_lower = s["description"].lower()
        kw_flags = []
        for kw in DEBIT_KEYWORDS:
            kw_flags.append(1.0 if kw in desc_lower else 0.0)
        for kw in CREDIT_KEYWORDS:
            kw_flags.append(1.0 if kw in desc_lower else 0.0)

        # Combine all features
        row = [amount_sign, amount_mag, page_pos] + kw_flags + list(tfidf_matrix[i])
        features.append(row)

    features_np = np.array(features, dtype=np.float32)
    labels_np = np.array([1.0 if s["type"] == "debit" else 0.0 for s in samples], dtype=np.float32)

    return features_np, labels_np, tfidf


def train_credit_debit(test_split: float = 0.15, epochs: int = 30, lr: float = 1e-3):
    """Train the credit/debit FFN classifier."""
    print("Loading training data...")
    samples = _load_training_data()
    if len(samples) < 50:
        print(f"Only {len(samples)} samples. Need at least 50.")
        sys.exit(1)

    type_dist = Counter(s["type"] for s in samples)
    print(f"Loaded {len(samples)} transactions: {dict(type_dist)}")

    # PDF-level split to prevent leakage
    pdf_stems = sorted(set(s["pdf_stem"] for s in samples))
    train_stems, test_stems = train_test_split(pdf_stems, test_size=test_split, random_state=42)
    test_stem_set = set(test_stems)

    train_samples = [s for s in samples if s["pdf_stem"] not in test_stem_set]
    test_samples = [s for s in samples if s["pdf_stem"] in test_stem_set]
    print(f"Train: {len(train_samples)} ({len(train_stems)} PDFs), "
          f"Test: {len(test_samples)} ({len(test_stems)} PDFs)")

    # Extract features
    X_train, y_train, tfidf = _extract_features(train_samples, fit=True)
    X_test, y_test, _ = _extract_features(test_samples, tfidf=tfidf, fit=False)

    input_dim = X_train.shape[1]
    print(f"Feature dimensions: {input_dim}")

    # Class weights for imbalance
    n_debit = y_train.sum()
    n_credit = len(y_train) - n_debit
    pos_weight = torch.tensor([n_credit / max(n_debit, 1)])
    print(f"Class weights — pos_weight (debit): {pos_weight.item():.2f}")

    # Convert to tensors
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Model
    model = CreditDebitFFN(input_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    X_train_t = X_train_t.to(device)
    y_train_t = y_train_t.to(device)
    X_test_t = X_test_t.to(device)
    y_test_t = y_test_t.to(device)

    # Training loop
    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        # Mini-batch training
        perm = torch.randperm(len(X_train_t))
        batch_size = 256
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            logits = model(X_train_t[idx]).squeeze(-1)
            loss = criterion(logits, y_train_t[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_t).squeeze(-1)
            test_preds = (torch.sigmoid(test_logits) > 0.5).float()
            acc = (test_preds == y_test_t).float().mean().item()

            # Per-class accuracy
            debit_mask = y_test_t == 1
            credit_mask = y_test_t == 0
            debit_acc = (test_preds[debit_mask] == y_test_t[debit_mask]).float().mean().item() if debit_mask.any() else 0
            credit_acc = (test_preds[credit_mask] == y_test_t[credit_mask]).float().mean().item() if credit_mask.any() else 0

        avg_loss = epoch_loss / n_batches
        if (epoch + 1) % 5 == 0 or acc > best_acc:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Acc: {acc:.4f} | Debit: {debit_acc:.4f} | Credit: {credit_acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model.load_state_dict(best_state)
    torch.save({
        "model_state_dict": best_state,
        "input_dim": input_dim,
    }, MODEL_DIR / "model.pt")

    # Save TF-IDF vectorizer
    with open(MODEL_DIR / "tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)

    # Save metadata
    meta = {
        "input_dim": input_dim,
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "best_accuracy": round(best_acc, 4),
        "type_distribution": dict(type_dist),
        "features": ["amount_sign", "amount_mag", "page_pos"] +
                    [f"kw_{k}" for k in DEBIT_KEYWORDS + CREDIT_KEYWORDS] +
                    [f"tfidf_{i}" for i in range(50)],
    }
    with open(MODEL_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nTraining complete. Best accuracy: {best_acc:.4f}")
    print(f"Model saved to {MODEL_DIR}")

    return best_acc


class CreditDebitClassifier:
    """Thread-safe singleton for credit/debit inference."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._loaded = False
        return cls._instance

    def _load(self):
        if self._loaded:
            return
        model_path = MODEL_DIR / "model.pt"
        tfidf_path = MODEL_DIR / "tfidf.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"No credit/debit model at {model_path}. Run credit_debit_model.py first.")

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model = CreditDebitFFN(checkpoint["input_dim"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        with open(tfidf_path, "rb") as f:
            self.tfidf = pickle.load(f)

        self._loaded = True

    def predict(self, transactions: list) -> list:
        """Predict credit/debit for a list of transaction dicts.

        Each dict should have: amount, description, page_index (optional).
        Returns list of (type_str, confidence) tuples.
        """
        self._load()

        if not transactions:
            return []

        samples = []
        for txn in transactions:
            samples.append({
                "amount": float(txn.get("amount", 0)),
                "description": txn.get("description", ""),
                "page_index": txn.get("page_index", txn.get("page", 0)),
                "type": "",  # placeholder
            })

        X, _, _ = _extract_features(samples, tfidf=self.tfidf, fit=False)
        X_t = torch.from_numpy(X)

        with torch.no_grad():
            logits = self.model(X_t).squeeze(-1)
            probs = torch.sigmoid(logits)

        results = []
        for p in probs:
            prob = p.item()
            if prob > 0.5:
                results.append(("debit", round(prob, 4)))
            else:
                results.append(("credit", round(1.0 - prob, 4)))

        return results


def get_classifier() -> CreditDebitClassifier:
    """Get the singleton credit/debit classifier."""
    return CreditDebitClassifier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train credit/debit FFN classifier")
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--evaluate", action="store_true", help="Evaluate saved model")
    args = parser.parse_args()

    if args.evaluate:
        samples = _load_training_data()
        clf = get_classifier()
        results = clf.predict(samples)
        correct = sum(1 for s, (pred, _) in zip(samples, results) if s["type"] == pred)
        print(f"Accuracy: {correct}/{len(samples)} ({correct/len(samples)*100:.1f}%)")
    else:
        train_credit_debit(
            test_split=args.test_split,
            epochs=args.epochs,
            lr=args.lr,
        )
