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

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- Utils ----------------

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

# ---------------- Health ----------------

@app.get("/")
def root():
    return {"status": "API running"}

# ---------------- Attachment Chat ----------------

@app.post("/ask")
async def ask_with_optional_file(
    question: str = Form(...),
    role: str = Form(...),
    file: UploadFile | None = File(None)
):
    if not question.strip():
        raise HTTPException(status_code=400, detail="Empty question")

    context = ""

    # ---- Les fil hvis den finnes ----
    if file and file.filename:
        path = os.path.join(UPLOAD_DIR, file.filename)

        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        ext = file.filename.lower().split(".")[-1]

        try:
            if ext == "pdf":
                context = read_pdf(path)
            elif ext in ["docx", "doc"]:
                context = read_docx(path)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")
        except Exception as e:
            print("Fil-lesefeil:", e)
            raise HTTPException(status_code=500, detail="Failed to read attachment")

    # ---- Begrens kontekst ----
    context = context[:3500]

    messages = [
        {
            "role": "system",
            "content": (
                "Du er en faglig assistent innen byggautomasjon.\n"
                f"Brukerrolle: {role}\n"
                "Svar presist og profesjonelt."
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
        raise HTTPException(status_code=500, detail="AI error")
