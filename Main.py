from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI(title="K1B-Sle API", version="1.0")

MODEL_NAME = "RikZD/k1b-sle-model"  # GANTI NANTI!

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

LABELS = ["safe", "gambling", "phishing", "malware", "adult"]

class URLRequest(BaseModel):
    url: str
    title: str = ""
    meta: str = ""
    domain_age_days: int = 0
    redirect_count: int = 0

@app.post("/classify")
def classify(req: URLRequest):
    text = f"{req.url} {req.title} {req.meta}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = probs.argmax().item()
        confidence = probs[0][pred].item()
    
    category = LABELS[pred]
    
    if category == "safe":
        action = "allow"
        reasons = ["Situs aman dan terpercaya"]
    else:
        action = "block" if confidence > 0.7 else "flag"
        reasons_map = {
            "gambling": "Kata kunci judi terdeteksi",
            "phishing": "Pola phishing terdeteksi",
            "malware": "Pola malware terdeteksi",
            "adult": "Konten dewasa terdeteksi"
        }
        reasons = [reasons_map.get(category, "Konten mencurigakan")]
        
        if req.domain_age_days < 7:
            reasons.append(f"Domain baru ({req.domain_age_days} hari)")
        if req.redirect_count > 3:
            reasons.append(f"Redirect mencurigakan ({req.redirect_count}x)")
    
    return {
        "url": req.url,
        "category": category,
        "confidence": round(confidence, 3),
        "action": action,
        "reasons": reasons
    }

@app.get("/")
def root():
    return {"status": "K1B-Sle API running", "model": MODEL_NAME}

@app.get("/health")
def health():
    return {"status": "healthy"}
