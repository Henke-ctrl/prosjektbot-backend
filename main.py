from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from docx import Document
import os
import shutil

# -------------------------------------------------
# Init
# -------------------------------------------------

load_dotenv()

app = FastAPI()

# CORS – tillat frontend
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
# Hjelpefunksjoner – fil-lesing
# -------------------------------------------------

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text.strip()

def read_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs).strip()

# -------------------------------------------------
# Health check
# -------------------------------------------------

@app.get("/")
def root():
    return {"status": "API running"}

# -------------------------------------------------
# Chat med valgfritt vedlegg (HOVEDENDPOINT)
# -------------------------------------------------

@app.post("/ask")
async def ask(
    question: str = Form(None),
    role: str = Form(None),
    file: UploadFile | None = File(None)
):
    # --- Valider input ---
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question is required")

    if not role:
        role = "Ukjent"

    context = ""

    # --- Les vedlegg hvis det finnes ---
    if file is not None and file.filename:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            ext = file.filename.lower().split(".")[-1]

            if ext == "pdf":
                context = read_pdf(file_path)
            elif ext in ["docx", "doc"]:
                context = read_docx(file_path)
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported file type"
                )

        except HTTPException:
            raise
        except Exception as e:
            print("Feil ved fil-lesing:", e)
            raise HTTPException(
                status_code=500,
                detail="Failed to process attachment"
            )

    # --- Begrens kontekst (VIKTIG) ---
    context = context[:3500]

    # --- Bygg meldinger ---
    messages = [
        {
            "role": "system",
            "content": (
                "Du er en faglig KI-assistent for prosjektstøtte "
                "innen byggautomasjon.\n"
                f"Brukerrolle: {role}\n"
                "Svar presist, strukturert og profesjonelt."
            )
        }
    ]

    if context:
        messages.append({
            "role": "system",
            "content": f"Dokumentvedlegg:\n{context}"
        })

    messages.append({
        "role": "user",
        "content": question
    })

    # --- Kall OpenAI ---
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            timeout=30
        )

        return {
            "answer": response.choices[0].message.content,
            "used_attachment": bool(context)
        }

    except Exception as e:
        print("OpenAI-feil:", e)
        raise HTTPException(
            status_code=500,
            detail="AI backend error"
        )
