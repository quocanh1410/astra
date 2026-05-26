
# # train.py
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import resample

# Load data
embedder = SentenceTransformer("all-MiniLM-L6-v2")

with open("dataset.json") as f:
    data = json.load(f)

# Encode request thành vector
X = np.array([embedder.encode(item["request"]) for item in data])
y = np.array([item["label"] for item in data])

print(f"Tổng: {len(data)} samples (label0={sum(y==0)}, label1={sum(y==1)})")

# Tách train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Cân bằng train set
X0 = X_train[y_train == 0]
y0 = y_train[y_train == 0]
X1 = X_train[y_train == 1]
y1 = y_train[y_train == 1]

X1_up, y1_up = resample(X1, y1, n_samples=len(X0), random_state=42)

X_train_bal = np.vstack([X0, X1_up])
y_train_bal = np.concatenate([y0, y1_up])

print(f"Balanced train: label0={len(X0)}, label1={len(X1_up)}")

# Train
clf = LogisticRegression(class_weight="balanced", max_iter=1000)
clf.fit(X_train_bal, y_train_bal)

# Evaluate
pred = clf.predict(X_test)
print("\n=== TEST REPORT ===\n")
print(classification_report(y_test, pred,
      target_names=["Cần LLM (0)", "SLM OK (1)"],
      zero_division=0))

# Lưu model
with open("urc_model.pkl", "wb") as f:
    pickle.dump(clf, f)
print("✅ Model lưu vào urc_model.pkl")

# Inference loop
print("\n--- Thử inference ---")
while True:
    req  = input("\nEnter request (Ctrl+C để thoát): ")
    emb  = embedder.encode(req)
    prob = clf.predict_proba([emb])[0][1]
    print(f"  SLM probability: {prob:.3f}")
    if prob > 0.5:
        print("  → Route to SLM (câu đơn giản)")
    else:
        print("  → Route to LLM (câu phức tạp)")




