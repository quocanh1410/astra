import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Load model đã train
embedder = SentenceTransformer("all-MiniLM-L6-v2")
with open("urc_model.pkl", "rb") as f:
    clf = pickle.load(f)

app = FastAPI()

class Request(BaseModel):
    text: str

@app.post("/classify")
def classify(req: Request):
    emb  = embedder.encode(req.text)
    prob = clf.predict_proba([emb])[0][1]
    return {
        "request"        : req.text,
        "slm_probability": round(float(prob), 3),
        "route"          : "SLM" if prob > 0.5 else "LLM"
    }