from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import shutil

from pypdf import PdfReader
from docx import Document

load_dotenv()

app = FastAPI()

# CORS – nødvendig for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------------------------------
# Hjelpefunksjoner for tekstekstraksjon
# -------------------------------------------------

def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text.strip()

def extract_text_from_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs]).strip()

# -------------------------------------------------
# Routes
# -------------------------------------------------

@app.get("/")
def root():
    return {"status": "API running"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Lagre fil
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Les tekst basert på filtype
    ext = file.filename.lower().split(".")[-1]

    if ext == "pdf":
        text = extract_text_from_pdf(file_path)
    elif ext in ["docx", "doc"]:
        text = extract_text_from_docx(file_path)
    else:
        return {
            "filename": file.filename,
            "status": "unsupported_file_type"
        }

    # Returner kontrollert mengde tekst
    return {
        "filename": file.filename,
        "characters": len(text),
        "preview": text[:1000]  # kun første 1000 tegn
    }
