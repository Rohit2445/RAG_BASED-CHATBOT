# rag_service.py
import os
import re
import random
import json
import pdfplumber
import boto3
from typing import List, Tuple
from sqlmodel import Session, select, delete
from dotenv import load_dotenv

# local imports (make sure these exist)
from crud import get_all_document_chunks, save_document_chunk
from models import DocumentChunk

load_dotenv()

# ---------------- config ----------------
RAG_MODE = os.getenv("RAG_MODE", "mock").lower()    # "mock" or "bedrock"
PDF_FOLDER = os.getenv("PDFS_FOLDER", "data/pdfs")
EMBED_DIM = int(os.getenv("EMBED_DIM", "128"))
USE_BEDROCK = os.getenv("USE_BEDROCK", "false").lower() in ("1", "true", "yes")
BEDROCK_MODEL_NAME = os.getenv("BEDROCK_MODEL_NAME", "sonet3.5")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# ---------- helper: extract text ----------
def _extract_text_from_pdf(path: str) -> str:
    text_parts: List[str] = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_parts.append(page_text.strip())
    except Exception as e:
        print(f"WARNING: Failed to extract text from {path}: {e}")
        return ""
    return "\n\n".join(text_parts)


# ---------- chunking ----------
def _chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    if not text:
        return []
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if current:
                chunks.append(current.strip())
            if len(para) <= max_chars:
                current = para
            else:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                cur_para = ""
                for s in sentences:
                    if len(cur_para) + len(s) + 1 <= max_chars:
                        cur_para = (cur_para + " " + s).strip() if cur_para else s
                    else:
                        if cur_para:
                            chunks.append(cur_para.strip())
                        cur_para = s
                current = cur_para
    if current:
        chunks.append(current.strip())

    # optional overlap
    if overlap and overlap < max_chars:
        overlapped: List[str] = []
        for i, c in enumerate(chunks):
            if i == 0:
                overlapped.append(c)
            else:
                prev = overlapped[-1]
                tail = prev[-overlap:]
                overlapped.append((tail + "\n\n" + c).strip())
        return overlapped

    return chunks


# ---------- embeddings ----------
def _generate_mock_embedding(text: str, dim: int = EMBED_DIM) -> List[float]:
    seed = abs(hash(text)) % (2**32)
    rng = random.Random(seed)
    return [rng.random() for _ in range(dim)]


def _generate_bedrock_embedding(text: str) -> List[float]:
    """
    Example bedrock invocation. This may need adjustment based on your Bedrock setup.
    We wrap in try/except and return [] (or fallback to mock) on failure.
    """
    try:
        client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
        body = json.dumps({"input": text, "mode": "embedding"})
        resp = client.invoke_model(
            modelId=BEDROCK_MODEL_NAME,
            contentType="application/json",
            accept="application/json",
            body=body
        )
        raw = resp["body"].read().decode()
        parsed = json.loads(raw)
        emb = parsed.get("embedding") or parsed.get("embeddings") or parsed.get("output", {}).get("embedding")
        if isinstance(emb, list):
            return emb
        else:
            print("WARNING: bedrock returned unexpected embedding format; falling back to mock.")
            return _generate_mock_embedding(text)
    except Exception as e:
        print(f"WARNING: Bedrock embedding failed: {e}. Falling back to mock embedding.")
        return _generate_mock_embedding(text)


def _generate_embedding(text: str) -> List[float]:
    if USE_BEDROCK:
        return _generate_bedrock_embedding(text)
    return _generate_mock_embedding(text)


# ---------- PDF loader convenience ----------
def load_all_pdfs(folder_path: str = None) -> List[tuple]:
    """
    Returns list of (filename, text) for all PDFs in folder_path.
    """
    folder = folder_path or PDF_FOLDER
    if not os.path.isdir(folder):
        print(f"WARNING: PDF folder not found: {folder}")
        return []
    pdf_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".pdf")]
    docs: List[tuple] = []
    for p in pdf_files:
        text = _extract_text_from_pdf(p)
        if text:
            docs.append((os.path.basename(p), text))
        else:
            print(f"INFO: no text extracted from {p}, skipping.")
    return docs


# ---------- DB initialization from PDFs ----------
def init_rag_db(session: Session, folder_path: str = None, reinit: bool = False):
    """
    Load PDFs into DocumentChunk table.
    If reinit=True, existing chunks will be deleted first.
    """
    folder_path = folder_path or PDF_FOLDER

    # if reinit, delete all existing chunks
    if reinit:
        try:
            session.exec(delete(DocumentChunk))
            session.commit()
            print("INFO: Cleared existing DocumentChunk rows (reinit=True).")
        except Exception as e:
            print(f"WARNING: failed to clear existing DocumentChunk rows: {e}")

    # quick skip if already initialized and not forcing reinit
    existing = session.exec(select(DocumentChunk)).first()
    if existing and not reinit:
        print("INFO: RAG DB already initialized — skipping (set reinit=True to force reload).")
        return

    docs = load_all_pdfs(folder_path)
    if not docs:
        print(f"INFO: No PDFs found or no text extracted in {folder_path}.")
        return

    total = 0
    for filename, text in docs:
        chunks = _chunk_text(text)
        print(f"INFO: {len(chunks)} chunks from {filename}")
        for c in chunks:
            chunk = DocumentChunk(
                source_document=filename,
                content=c,
                mock_embedding=_generate_embedding(c)
            )
            try:
                save_document_chunk(session, chunk)
                total += 1
            except Exception as e:
                print(f"WARNING: failed to save chunk from {filename}: {e}")
    print(f"INFO: loaded {total} chunks from {len(docs)} PDF(s).")


# ---------- RAG Service ----------
class RagService:
    def __init__(self, session: Session):
        self.session = session
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self):
        return get_all_document_chunks(self.session)

    def _mock_vector_search(self, query: str, top_k: int = 3) -> List[str]:
        """
        Simple keyword scoring retrieval (mock). Returns top_k chunk texts.
        """
        query_keywords = [w.lower() for w in query.split() if len(w) > 3]
        scored: List[tuple] = []
        for chunk in self.knowledge_base:
            score = 0
            for kw in query_keywords:
                if kw in chunk.content.lower():
                    score += 1
            if score > 0:
                scored.append((score, chunk.content))
        # fallback: if nothing matched, return the first top_k chunks (so LLM can still answer)
        if not scored:
            return [c.content for c in (self.knowledge_base[:top_k] if len(self.knowledge_base) >= top_k else self.knowledge_base)]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [content for score, content in scored[:top_k]]

    def _bedrock_generate(self, query: str, context: List[str]) -> str:
        """
        Call Bedrock to generate response using context.
        This is an example wrapper and may need adaptation for your Bedrock / IAM config.
        """
        try:
            client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
            prompt = "Context:\n" + "\n\n".join(context) + "\n\nUser question:\n" + query
            body = {"input": prompt, "max_tokens": 512}
            resp = client.invoke_model(
                modelId=BEDROCK_MODEL_NAME,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body)
            )
            data = json.loads(resp["body"].read().decode())
            return data.get("output") or data.get("generated_text") or str(data)
        except Exception as e:
            print(f"WARNING: Bedrock generation failed: {e}. Falling back to mock generation.")
            return self._mock_llm_generation(query, context)

    def _mock_llm_generation(self, query: str, context: List[str]) -> str:
        context_str = "\n".join(context)
        if not context_str:
            return "I couldn't find any relevant information in the policy to answer that."
        lowerq = query.lower()
        if any(k in lowerq for k in ["deductible", "$1000", "$1,000"]):
            return "The annual deductible for this plan is $1,000."
        if any(k in lowerq for k in ["out-of-pocket maximum", "out of pocket", "max"]):
            return "The annual out-of-pocket maximum is $5,000."
        if any(k in lowerq for k in ["excluded", "not covered", "exclusions", "dental"]):
            return "The policy excludes cosmetic surgery, dental/vision (unless accidental injury), experimental treatments, and war-related injuries."
        return f"Based on the policy context ({context_str[:240]}...), the answer: {query}"

    def generate_response(self, query: str) -> Tuple[str, List[str]]:
        if RAG_MODE == "bedrock" and USE_BEDROCK:
            context = self._mock_vector_search(query)
            answer = self._bedrock_generate(query, context)
            return answer, context
        else:
            context = self._mock_vector_search(query)
            answer = self._mock_llm_generation(query, context)
            return answer, context
