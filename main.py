from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import shutil
import threading

from pypdf import PdfReader
from docx import Document
from openai import OpenAI

load_dotenv()

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------------------------------
# Hjelpefunksjoner (isolert)
# -------------------------------------------------

def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text.strip()

def extract_text_from_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs).strip()

# -------------------------------------------------
# Health check
# -------------------------------------------------

@app.get("/")
def root():
    return {"status": "API running"}

# -------------------------------------------------
# Chat – ALLTID uavhengig av filopplasting
# -------------------------------------------------

@app.post("/ask")
def ask_ai(data: dict):
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": f"Brukerrolle: {data.get('role')}"},
                {"role": "user", "content": data.get("question")}
            ],
            timeout=30
        )
        return {"answer": response.choices[0].message.content}

    except Exception as e:
        print("OpenAI-feil:", e)
        raise HTTPException(status_code=500, detail="AI backend error")

# -------------------------------------------------
# Upload – isolert og beskyttet
# -------------------------------------------------

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ext = file.filename.lower().split(".")[-1]

    try:
        if ext == "pdf":
            text = extract_text_from_pdf(file_path)
        elif ext in ["docx", "doc"]:
            text = extract_text_from_docx(file_path)
        else:
            return {
                "filename": file.filename,
                "status": "unsupported_file_type"
            }

        # VIKTIG: aldri returner for mye
        return {
            "filename": file.filename,
            "characters": len(text),
            "preview": text[:800]
        }

    except Exception as e:
        print("Fil-lesefeil:", e)
        raise HTTPException(status_code=500, detail="File processing error")
