from fastapi import FastAPI
from pydantic import BaseModel
import os
from openai import OpenAI

# -------------------------
# App-oppsett
# -------------------------
app = FastAPI(
    title="Prosjektbot API",
    description="Backend for FDV- og prosjektassistent",
    version="1.0.0"
)

# -------------------------
# OpenAI-klient
# (API-nøkkel settes i Railway Variables)
# -------------------------
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# -------------------------
# Datamodeller
# -------------------------
class Question(BaseModel):
    question: str
    role: str

# -------------------------
# Root – helsesjekk
# -------------------------
@app.get("/")
def root():
    return {"status": "API running"}

# -------------------------
# Chat-endepunkt
# -------------------------
@app.post("/ask")
def ask_ai(data: Question):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": f"Du er en faglig assistent. Brukerrolle: {data.role}"
            },
            {
                "role": "user",
                "content": data.question
            }
        ]
    )

    return {
        "answer": response.choices[0].message.content
    }

# -------------------------
# FDV-dashboard / score
# -------------------------
@app.get("/fdv-dashboard")
def fdv_dashboard():
    return {
        "total_score": 80,
        "breakdown": {
            "A – Orientering": {"percent": 100},
            "B – Drift": {"percent": 100},
            "C – Tilsyn og vedlikehold": {"percent": 50},
            "D – Dokumentasjon": {"percent": 75},
            "E – Teknisk dokumentasjon": {"percent": 75}
        }
    }
