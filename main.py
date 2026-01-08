from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from openai import OpenAI
import shutil

load_dotenv()

app = FastAPI()

# CORS (viktig for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- Models ----------------

class Question(BaseModel):
    question: str
    role: str

# ---------------- Routes ----------------

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

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "filename": file.filename,
        "status": "uploaded"
    }
