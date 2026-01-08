# FORCE RAILWAY REDEPLOY

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from openai import OpenAI

# -------------------------------------------------
# App-oppsett
# -------------------------------------------------
app = FastAPI(
    title="Prosjektbot API",
    description="Backend for KI-assistent innen byggautomasjon og FDV",
    version="1.0.0"
)

# -------------------------------------------------
# OpenAI-klient
# API-nøkkel settes i Railway → Variables
# Key: OPENAI_API_KEY
# -------------------------------------------------
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# -------------------------------------------------
# Datamodeller
# -------------------------------------------------
class Question(BaseModel):
    question: str
    role: str

# -------------------------------------------------
# Root / helsesjekk
# -------------------------------------------------
@app.get("/")
def root():
    return {"status": "API running"}

# -------------------------------------------------
# Chat-endepunkt (KORREKT for OpenAI Responses API)
# -------------------------------------------------
@app.post("/ask")
def ask_ai(data: Question):
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": f"Du er en faglig KI-assistent for prosjektstøtte innen byggautomasjon. Brukerrolle: {data.role}"
                },
                {
                    "role": "user",
                    "content": data.question
                }
            ]
        )

        # Robust uthenting av tekst fra responsen
        answer = ""

        if response.output:
            for item in response.output:
                if item.get("type") == "message":
                    for content in item.get("content", []):
                        if content.get("type") == "output_text":
                            answer += content.get("text", "")

        if not answer:
            answer = "Ingen tekstlig respons fra modellen."

        return {"answer": answer}

    except Exception as e:
        # Viktig: gir synlig feil i Swagger + Railway logs
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------
# FDV-dashboard (status / score)
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
