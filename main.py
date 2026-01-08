from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

app = FastAPI(title="Prosjektbot API")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Question(BaseModel):
    question: str
    role: str

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

@app.get("/")
def root():
    return {"status": "API running"}
