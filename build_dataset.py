
# import json
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from model import call_slm, call_llm
# from dataset import requests_data

# embedder  = SentenceTransformer("all-MiniLM-L6-v2")
# THRESHOLD_HIGH = 0.85
# THRESHOLD_LOW  = 0.65

# dataset  = []
# skipped  = 0

# for i, req in enumerate(requests_data):
#     print(f"\n[{i+1}/{len(requests_data)}] {req[:60]}")

#     # Thêm try/except để không mất data nếu 1 request lỗi
#     try:
#         weak_output   = call_slm(req)
#         strong_output = call_llm(req)
#     except Exception as e:
#         print(f"  ⚠️  Lỗi gọi model: {e} → bỏ qua")
#         continue

#     emb1 = embedder.encode(weak_output)
#     emb2 = embedder.encode(strong_output)
#     similarity = cosine_similarity([emb1], [emb2])[0][0]

#     if similarity > THRESHOLD_HIGH:
#         label = 1
#     elif similarity < THRESHOLD_LOW:
#         label = 0
#     else:
#         skipped += 1
#         print(f"  ⏭️  Ambiguous (sim={similarity:.3f}) → skip")
#         continue

#     dataset.append({
#         "request"      : req,
#         "weak_output"  : weak_output,
#         "strong_output": strong_output,
#         "similarity"   : float(similarity),
#         "label"        : label
#     })
#     print(f"  sim={similarity:.3f} → label={label} "
#           f"({'SLM OK' if label==1 else 'cần LLM'})")

#     # Lưu mỗi 10 request phòng crash
#     if (i + 1) % 10 == 0:
#         with open("dataset.json", "w") as f:
#             json.dump(dataset, f, indent=2)
#         print(f"  💾 Checkpoint: {len(dataset)} samples lưu")

# # Lưu lần cuối
# with open("dataset.json", "w") as f:
#     json.dump(dataset, f, indent=2)

# # Thống kê
# n_pos = sum(d["label"] for d in dataset)
# n_neg = len(dataset) - n_pos
# print(f"\n✅ DONE")
# print(f"   Saved   : {len(dataset)} samples")
# print(f"   Skipped : {skipped} ambiguous")
# print(f"   Label=1 : {n_pos} ({n_pos/len(dataset)*100:.1f}%)")
# print(f"   Label=0 : {n_neg} ({n_neg/len(dataset)*100:.1f}%)")





# profile.py
import json
from dataset import requests_data

dataset = []

for item in requests_data:
    dataset.append({
        "request"      : item["request"],
        "weak_output"  : "manual_label",   # placeholder
        "strong_output": "manual_label",   # placeholder
        "similarity"   : 1.0 if item["label"] == 1 else 0.0,  # placeholder
        "label"        : item["label"],
    })
    print(f"label={item['label']} | {item['request'][:60]}")

with open("dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

n1 = sum(d["label"] for d in dataset)
n0 = len(dataset) - n1
print(f"\n✅ DONE — {len(dataset)} samples (label1={n1}, label0={n0})")