import json
import re
import pickle
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from model import call_slm, call_llm

# Load URC
embedder = SentenceTransformer("all-MiniLM-L6-v2")
with open("urc_model.pkl", "rb") as f:
    clf = pickle.load(f)

def extract_number(text):
    numbers = re.findall(r'\d+\.?\d*', text)
    return numbers[-1] if numbers else ""

def route_urc(request):
    emb  = embedder.encode(request)
    prob = clf.predict_proba([emb])[0][1]
    return "SLM" if prob > 0.5 else "LLM", prob

# Load 50 câu test từ GSM8K test set
print("Loading GSM8K test set...", flush=True)
gsm8k = load_dataset("gsm8k", "main", split="test")

N_EVAL = 100
results = []

print(f"Evaluating {N_EVAL} questions...\n", flush=True)

for i, ex in enumerate(gsm8k.select(range(N_EVAL))):
    request = ex["question"]
    answer  = ex["answer"].split("####")[-1].strip()

    print(f"[{i+1}/{N_EVAL}] {request[:55]}", flush=True)

    try:
        # 1. Full LLM
        out_llm  = call_llm(request)
        llm_pred = extract_number(out_llm)
        llm_ok   = llm_pred == answer

        # 2. Full SLM
        out_slm  = call_slm(request)
        slm_pred = extract_number(out_slm)
        slm_ok   = slm_pred == answer

        # 3. HERA URC
        route, prob = route_urc(request)
        if route == "SLM":
            hera_out  = out_slm   # tái dụng
            hera_pred = slm_pred
        else:
            hera_out  = out_llm   # tái dụng
            hera_pred = llm_pred
        hera_ok = hera_pred == answer

        results.append({
            "request" : request,
            "answer"  : answer,
            "llm_pred": llm_pred,
            "slm_pred": slm_pred,
            "llm_ok"  : llm_ok,
            "slm_ok"  : slm_ok,
            "hera_ok" : hera_ok,
            "route"   : route,
            "prob"    : round(prob, 3),
        })

        print(f"  answer={answer} | "
              f"LLM={'OK' if llm_ok else 'WRONG'} | "
              f"SLM={'OK' if slm_ok else 'WRONG'} | "
              f"HERA={'OK' if hera_ok else 'WRONG'} (→{route} p={prob:.2f})",
              flush=True)

    except Exception as e:
        print(f"  ERROR: {e}", flush=True)

    # Checkpoint
    if (i + 1) % 10 == 0:
        with open("eval_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Checkpoint saved\n", flush=True)

# Lưu kết quả
with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Tính accuracy
n        = len(results)
llm_acc  = sum(r["llm_ok"]  for r in results) / n * 100
slm_acc  = sum(r["slm_ok"]  for r in results) / n * 100
hera_acc = sum(r["hera_ok"] for r in results) / n * 100
slm_pct  = sum(1 for r in results if r["route"] == "SLM") / n * 100

print(f"\n{'='*100}")
print(f"EVALUATION RESULTS ({n} questions)")
print(f"{'='*100}")
print(f"Full LLM accuracy  : {llm_acc:.1f}%")
print(f"Full SLM accuracy  : {slm_acc:.1f}%")
print(f"HERA accuracy      : {hera_acc:.1f}%")
print(f"HERA SLM usage     : {slm_pct:.1f}% (tiết kiệm LLM)")
print(f"Accuracy drop      : {llm_acc - hera_acc:.1f}%")
print(f"{'='*100}")
