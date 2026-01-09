from fastapi import FastAPI, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
from docx import Document
from openpyxl import load_workbook
import os, json, re, uuid, shutil
from typing import List

# -------------------------------------------------
# Init
# -------------------------------------------------

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY mangler")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
DATABLAD_DIR = "datablad"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATABLAD_DIR, exist_ok=True)

# -------------------------------------------------
# Session-kontekst (per chat)
# -------------------------------------------------

SESSION_CONTEXT: dict[str, dict] = {}

# -------------------------------------------------
# Konfig
# -------------------------------------------------

CONFIRMATION_PHRASES = {
    "ja", "ja takk", "ok", "okei", "gjerne", "gjerne det", "fortsett"
}

PRODUCT_REGEX = re.compile(r"\b[A-Z]{2,4}\d{3,5}[-\.]?\d*\b")

# -------------------------------------------------
# Fil-lesing
# -------------------------------------------------

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def read_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def read_excel(path: str) -> str:
    wb = load_workbook(path, data_only=True)
    sheet = wb.active
    rows = []
    for i, row in enumerate(sheet.iter_rows(values_only=True)):
        if i > 50:
            break
        rows.append(" | ".join(str(c) if c else "" for c in row))
    return "\n".join(rows)

# -------------------------------------------------
# Indeksering
# -------------------------------------------------

def chunk_text(text: str, size=500, overlap=100):
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return chunks

def index_datasheet(pdf_path: str, vendor: str):
    text = read_pdf(pdf_path)
    data = {
        "vendor": vendor,
        "source": os.path.basename(pdf_path),
        "chunks": chunk_text(text)
    }
    with open(pdf_path.replace(".pdf", ".json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# -------------------------------------------------
# RAG
# -------------------------------------------------

def tokenize(text): return re.findall(r"\w+", text.lower())

def score_chunk(tokens, chunk):
    ct = tokenize(chunk)
    return sum(ct.count(t) for t in tokens)

def search_datasheets(query, vendor):
    vendor_dir = os.path.join(DATABLAD_DIR, vendor)
    if not os.path.exists(vendor_dir):
        return [], []

    tokens = tokenize(query)
    scored = []

    for f in os.listdir(vendor_dir):
        if not f.endswith(".json"):
            continue
        data = json.load(open(os.path.join(vendor_dir, f), encoding="utf-8"))
        for c in data["chunks"]:
            s = score_chunk(tokens, c)
            if s > 0:
                scored.append((s, data["source"], c))

    scored.sort(reverse=True)
    chunks, sources = [], []
    for _, src, txt in scored[:4]:
        chunks.append(f"[{src}]\n{txt}")
        if src not in sources:
            sources.append(src)

    return chunks, sources

def extract_products(text: str) -> List[str]:
    return list(set(PRODUCT_REGEX.findall(text.upper())))

def match_products_by_filename(products, vendor):
    vendor_dir = os.path.join(DATABLAD_DIR, vendor)
    hits = []
    for f in os.listdir(vendor_dir):
        if not f.lower().endswith(".pdf"):
            continue
        fname = f.lower().replace(" ", "")
        for p in products:
            if p.lower().replace(".", "").replace("-", "") in fname:
                hits.append(f)
    return list(set(hits))

def load_chunks_from_sources(vendor, sources):
    vendor_dir = os.path.join(DATABLAD_DIR, vendor)
    chunks = []
    for f in os.listdir(vendor_dir):
        if f.endswith(".json"):
            data = json.load(open(os.path.join(vendor_dir, f), encoding="utf-8"))
            if data["source"] in sources:
                for c in data["chunks"][:4]:
                    chunks.append(f"[{data['source']}]\n{c}")
    return chunks

# -------------------------------------------------
# Health
# -------------------------------------------------

@app.get("/")
def root():
    return {"status": "API running"}

# -------------------------------------------------
# Chat
# -------------------------------------------------

@app.post("/ask")
async def ask(request: Request):
    session_id = request.headers.get("X-Session-ID") or str(uuid.uuid4())
    SESSION_CONTEXT.setdefault(session_id, {"sources": []})

    body = await request.form() if "multipart" in request.headers.get("content-type","") else await request.json()
    question = body.get("question","").strip()
    role = body.get("role","Ukjent")
    file = body.get("file")

    if not question:
        raise HTTPException(400, "Spørsmål mangler")

    # --- Bekreftelse uten innhold ---
    if question.lower() in CONFIRMATION_PHRASES:
        return {
            "answer": (
                "Supert 👍 Hva ønsker du mer informasjon om?\n\n"
                "• Elektriske data\n"
                "• Utgangssignal\n"
                "• Nøyaktighet / toleranser\n"
                "• Monteringskrav\n\n"
                "Still gjerne et konkret oppfølgingsspørsmål."
            ),
            "sources": SESSION_CONTEXT[session_id]["sources"],
            "session_id": session_id,
            "used_rag": bool(SESSION_CONTEXT[session_id]["sources"])
        }

    # --- Produktgjenkjenning ---
    products = extract_products(question)

    rag_chunks, rag_sources = search_datasheets(question, "Siemens")

    # Filnavn fallback
    filename_hits = match_products_by_filename(products, "Siemens")
    for f in filename_hits:
        if f not in rag_sources:
            rag_sources.append(f)

    # Oppfølgingsspørsmål → bruk kontekst
    if not rag_sources and len(question.split()) < 6:
        rag_sources = SESSION_CONTEXT[session_id]["sources"]

    if rag_sources:
        rag_chunks = load_chunks_from_sources("Siemens", rag_sources)
        SESSION_CONTEXT[session_id]["sources"] = rag_sources

    # --- Prompt ---
    messages = [{
        "role": "system",
        "content": (
            "Du er en faglig KI-assistent for byggautomasjon.\n"
            f"Brukerrolle: {role}\n"
            "Svar presist, teknisk korrekt og uten antakelser."
        )
    }]

    if rag_chunks:
        messages.append({
            "role": "system",
            "content": "Relevante utdrag fra datablad:\n" + "\n\n".join(rag_chunks[:3000])
        })

    messages.append({"role": "user", "content": question})

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        timeout=30
    )

    return {
        "answer": resp.choices[0].message.content,
        "sources": rag_sources,
        "session_id": session_id,
        "used_rag": bool(rag_sources)
    }
