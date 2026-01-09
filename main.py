from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from docx import Document
from openpyxl import load_workbook
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

# -------------------------------------------------
# Fil-lesing
# -------------------------------------------------

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(
        page.extract_text() or "" for page in reader.pages
    ).strip()

def read_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs).strip()

def read_excel(path: str) -> str:
    wb = load_workbook(path, data_only=True)
    sheet = wb.active

    rows = []
    for i, row in enumerate(sheet.iter_rows(values_only=True)):
        if i > 50:  # begrens for ytelse
            break
        rows.append(" | ".join(str(cell) if cell is not None else "" for cell in row))

    return "\n".join(rows).strip()

# -------------------------------------------------
# Health
# -------------------------------------------------

@app.get("/")
def root():
    return {"status": "API running"}

# -------------------------------------------------
# Chat med valgfritt vedlegg (PDF / DOCX / XLSX)
# -------------------------------------------------

@app.post("/ask")
async def ask(request: Request):
    content_type = request.headers.get("content-type", "")
    question = None
    role = "Ukjent"
    file = None

    # JSON
    if "application/json" in content_type:
        data = await request.json()
        question = data.get("question")
        role = data.get("role", role)

    # FormData (med fil)
    elif "multipart/form-data" in content_type:
        form = await request.form()
        question = form.get("question")
        role = form.get("role", role)
        file = form.get("file")

    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question is required")

    context = ""

    if file and file.filename:
        path = os.path.join(UPLOAD_DIR, file.filename)
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        ext = file.filename.lower().split(".")[-1]

        if ext == "pdf":
            context = read_pdf(path)
        elif ext in ["docx", "doc"]:
            context = read_docx(path)
        elif ext in ["xlsx", "xls"]:
            context = read_excel(path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

    context = context[:3500]  # viktig grense

    messages = [
        {
            "role": "system",
            "content": (
                "Du er en faglig KI-assistent for prosjektstøtte innen byggautomasjon.\n"
                f"Brukerrolle: {role}\n"
                "Svar presist og faglig korrekt."
            )
        }
    ]

    if context:
        messages.append({
            "role": "system",
            "content": f"Dokument-/Excel-innhold:\n{context}"
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
        "used_attachment": bool(context)
    }

def index_all_datasheets(base_dir: str, vendor: str):
    """
    Leser alle PDF-er i datablad/vendor og lager .json-indekser
    """
    for filename in os.listdir(base_dir):
        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(base_dir, filename)
        print(f"Indekserer {pdf_path}...")
        index_datasheet(pdf_path, vendor)


def search_datasheets(query: str, vendor_dir: str) -> list[str]:
    results = []

    for filename in os.listdir(vendor_dir):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(vendor_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for chunk in data["chunks"]:
            if query.lower() in chunk.lower():
                results.append(f"{data['source']}: {chunk[:200]}...")
                break

    return results
