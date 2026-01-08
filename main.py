# FORCE RAILWAY REDEPLOY

from fastapi import FastAPI
from pydantic import BaseModel
import os
from openai import OpenAI

app = FastAPI(
    title="Prosjektbot API",
    version="1.0.0"
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Question(BaseModel):
    question: str
    role: str

@app.get("/")
def root():
    return {"status": "API running"}

@app.post("/ask")
def ask_ai(data: Question):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": f"Brukerrolle: {data.role}"},
            {"role": "user", "content": data.question}
        ]
    )
    return {"answer": response.choices[0].message.content}

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
