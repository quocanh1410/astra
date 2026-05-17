# # build_real_dataset.py
# import json
# import re
# from datasets import load_dataset
# from model import call_slm, call_llm
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# embedder = SentenceTransformer("all-MiniLM-L6-v2")

# THRESHOLD_HIGH = 0.85
# THRESHOLD_LOW  = 0.65
# N_TARGET       = 100

# def get_similarity(text1: str, text2: str) -> float:
#     emb1 = embedder.encode(text1)
#     emb2 = embedder.encode(text2)
#     return float(cosine_similarity([emb1], [emb2])[0][0])

# # Load GSM8K
# print("Loading GSM8K...")
# gsm8k = load_dataset("gsm8k", "main", split="train")

# dataset = []
# skipped = 0

# print(f"Building dataset ({N_TARGET} samples)...\n")

# for i, ex in enumerate(gsm8k.select(range(N_TARGET))):
#     request = ex["question"]

#     print(f"[{i+1}/{N_TARGET}] {request[:55]}")

#     try:
#         out_slm = call_slm(request)
#         out_llm = call_llm(request)

#         similarity = get_similarity(out_slm, out_llm)

#         if similarity > THRESHOLD_HIGH:
#             label = 1
#         elif similarity < THRESHOLD_LOW:
#             label = 0
#         else:
#             skipped += 1
#             print(f"  ⏭️  Ambiguous (sim={similarity:.3f}) → skip")
#             continue

#         dataset.append({
#             "request"      : request,
#             "weak_output"  : out_slm,
#             "strong_output": out_llm,
#             "similarity"   : round(similarity, 4),
#             "label"        : label,
#         })

#         print(f"  sim={similarity:.3f} → label={label} "
#               f"({'SLM OK' if label==1 else 'cần LLM'})")

#     except Exception as e:
#         skipped += 1
#         print(f"  ⚠️  Error: {e}")

#     # Checkpoint mỗi 10 câu
#     if (i + 1) % 10 == 0:
#         with open("dataset.json", "w") as f:
#             json.dump(dataset, f, indent=2)
#         print(f"  💾 Checkpoint: {len(dataset)} samples saved\n")

# # Lưu final
# with open("dataset.json", "w") as f:
#     json.dump(dataset, f, indent=2)

# # Thống kê
# n1 = sum(d["label"] for d in dataset)
# n0 = len(dataset) - n1

# print(f"\n{'='*50}")
# print(f"✅ DONE")
# print(f"   Saved   : {len(dataset)} samples")
# print(f"   Skipped : {skipped} (ambiguous + error)")
# print(f"   Label=1 : {n1} ({n1/len(dataset)*100:.1f}%) SLM OK")
# print(f"   Label=0 : {n0} ({n0/len(dataset)*100:.1f}%) cần LLM")
# print(f"{'='*50}")



# build_real_dataset.py — đổi hoàn toàn cách tạo label

# build_real_dataset.py
import json
import re
from datasets import load_dataset
from model import call_slm

N_TARGET = 100

def extract_number(text: str) -> str:
    """Lấy số cuối cùng trong output."""
    numbers = re.findall(r'\d+\.?\d*', text)
    return numbers[-1] if numbers else ""

# Load GSM8K
print("Loading GSM8K...", flush=True)
gsm8k = load_dataset("gsm8k", "main", split="train")

dataset = []
n_error = 0

print(f"Building dataset ({N_TARGET} samples)...\n", flush=True)

for i, ex in enumerate(gsm8k.select(range(N_TARGET))):
    request = ex["question"]
    answer  = ex["answer"].split("####")[-1].strip()

    print(f"[{i+1}/{N_TARGET}] {request[:55]}", flush=True)

    try:
        out_slm  = call_slm(request)
        slm_pred = extract_number(out_slm)
        label    = 1 if slm_pred == answer else 0

        correct = "✅" if label == 1 else "❌"
        print(f"  SLM pred={slm_pred} | answer={answer} "
              f"{correct} → label={label}", flush=True)

        dataset.append({
            "request"    : request,
            "weak_output": out_slm,
            "similarity" : 1.0 if label == 1 else 0.0,  # placeholder
            "label"      : label,
            "answer"     : answer,
            "slm_pred"   : slm_pred,
        })

    except Exception as e:
        n_error += 1
        print(f"  ⚠️  Error: {e}", flush=True)

    # Checkpoint mỗi 10 câu
    if (i + 1) % 10 == 0:
        with open("dataset.json", "w") as f:
            json.dump(dataset, f, indent=2)
        n1 = sum(d["label"] for d in dataset)
        n0 = len(dataset) - n1
        print(f"  💾 Checkpoint: {len(dataset)} samples "
              f"(label1={n1}, label0={n0})\n", flush=True)

# Lưu final
with open("dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

# Thống kê
n1 = sum(d["label"] for d in dataset)
n0 = len(dataset) - n1

print(f"\n{'='*50}", flush=True)
print(f"✅ DONE")
print(f"   Saved  : {len(dataset)} samples")
print(f"   Error  : {n_error}")
print(f"   Label 1: {n1} ({n1/len(dataset)*100:.1f}%) SLM đúng")
print(f"   Label 0: {n0} ({n0/len(dataset)*100:.1f}%) SLM sai")
print(f"{'='*50}")