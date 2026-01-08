# FORCE RAILWAY REDEPLOY

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import openai

# -------------------------------------------------
# OpenAI-oppsett (klassisk SDK)
# -------------------------------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")

# -------------------------------------------------
# App-oppsett
# -------------------------------------------------
app = FastAPI(
    title="Prosjektbot API",
    version="1.0.0"
)

# -------------------------------------------------
# Datamodeller
# -------------------------------------------------
class Question(BaseModel):
    question: str
    role: str

# -------------------------------------------------
# Root
# -------------------------------------------------
@app.get("/")
def root():
    return {"status": "API running"}

# -------------------------------------------------
# Chat-endepunkt (STABIL)
# -------------------------------------------------
@app.post("/ask")
def ask_ai(data: Question):
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"Du er en faglig KI-assistent for prosjektstøtte innen byggautomasjon. Brukerrolle: {data.role}"
                },
                {
                    "role": "user",
                    "content": data.question
                }
            ],
            temperature=0.3
        )

        return {
            "answer": completion.choices[0].message["content"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------
# FDV-dashboard
# -------------------------------------------------
@app.get("/fdv-dashboard")
@app.get("/fdv-dashboard/")
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
