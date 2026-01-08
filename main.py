from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from docx import Document
import os
import shutil

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------
# Hjelpefunksjoner
# ---------------------------

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

# ---------------------------
# Health
# ---------------------------

@app.get("/")
def root():
    return {"status": "API running"}

# ---------------------------
# Chat med valgfritt vedlegg
# ---------------------------

@app.post("/ask-with-file")
async def ask_with_file(
    question: str = Form(...),
    role: str = Form(...),
    file: UploadFile | None = File(None)
):
    context_text = ""

    # Hvis fil er vedlagt
    if file:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        ext = file.filename.lower().split(".")[-1]

        try:
            if ext == "pdf":
                context_text = extract_text_from_pdf(file_path)
            elif ext in ["docx", "doc"]:
                context_text = extract_text_from_docx(file_path)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")
        except Exception as e:
            raise HTTPException(status_code=500, detail="Failed to read file")

    # Begrens kontekst (viktig!)
    context_text = context_text[:4000]

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    f"Du er en faglig assistent innen byggautomasjon.\n"
                    f"Brukerrolle: {role}\n\n"
                    f"Hvis dokumentkontekst er gitt, bruk den aktivt i svaret."
                )
            }
        ]

        if context_text:
            messages.append({
                "role": "system",
                "content": f"Dokumentkontekst:\n{context_text}"
            })

        messages.append({
            "role": "user",
            "content": question
        })

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            timeout=30
        )

        return {
            "answer": response.choices[0].message.content,
            "used_file": bool(file)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail="AI processing failed")
