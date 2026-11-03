import re 
import shutil 
from collections import Counter 

from langchain_chroma import Chroma 
import pymupdf
from langchain_core.documents import Document 
from langchain_text_splitters import RecursiveCharacterTextSplitter 


from src.config import (
    CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME, COMPANY_NAMES, PDF_DIR,
)

from src.embedding import get_embeddings

EDGE_LINES = 6
REPEAT_THRESHOLD = 0.5

def _pattern(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())

def _remove_headers_footers(pages: list[Document]) -> None:
    edge_counts = Counter()
    for page in pages:
        lines = [l for l in page.page_content.splitlines() if l.strip()]
        edge_counts.update({_pattern(l) for l in lines[:EDGE_LINES] + lines[-EDGE_LINES:]})

    min_pages = max(2, int(len(pages) * REPEAT_THRESHOLD))
    repeated = {p for p, n in edge_counts.items() if n >= min_pages}

    for page in pages:
        lines = [l for l in page.page_content.splitlines() if l.strip()]
        n = len(lines)
        kept = [
            l for i, l in enumerate(lines)
            if not ((i < EDGE_LINES or i >= n - EDGE_LINES) and _pattern(l) in repeated)
        ]
        page.page_content = re.sub(r"[ \t]+", " ", "\n".join(kept)).strip()


def load_pdfs() -> list[Document]:
    all_pages = []
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        company, quarter = pdf.stem.split("_", 1)  # "tcs_q1_fy27" -> ("tcs", "q1_fy27")
        with pymupdf.open(pdf) as doc:
            pages = [
                Document(page_content=page.get_text(), metadata={"page": i})
                for i, page in enumerate(doc)
            ]
        _remove_headers_footers(pages)

        for page in pages:
            page.metadata = {
                "company": company.lower(),
                "quarter": quarter.upper().replace("_", " "),  # "Q1 FY27"
                "source": pdf.name,
                "page": page.metadata["page"] + 1,  # loader is 0-indexed
            }
        all_pages.extend(p for p in pages if p.page_content)
        print(f"Loaded {pdf.name}: {len(pages)} pages")
    return all_pages

def split_into_chunks(pages: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(pages)
    for chunk in chunks:
        m = chunk.metadata
        name = COMPANY_NAMES.get(m["company"], m["company"])
        chunk.page_content = f"[{name} | {m['quarter']} earnings call]\n{chunk.page_content}"
    return chunks


def build_vectorstore(chunks: list[Document]) -> Chroma:
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)  # full rebuild keeps re-runs idempotent
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )

def main() -> None:
    pages = load_pdfs()
    chunks = split_into_chunks(pages)
    store = build_vectorstore(chunks)
    print(f"\nStored {len(chunks)} chunks in {CHROMA_DIR}")
    print("Per company:", dict(Counter(c.metadata["company"] for c in chunks)))

    # Sanity check: a filtered search must only return the requested company
    print("\nTest search (company=tcs):")
    for doc in store.similarity_search("revenue growth this quarter", k=3, filter={"company": "tcs"}):
        print("  ", doc.metadata)


if __name__ == "__main__":
    main()