from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import CHROMA_DIR, COLLECTION_NAME, TOP_K
from src.embedding import get_embeddings


@lru_cache(maxsize=1)
def _get_vectorstore() -> Chroma:
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Vector store not found at {CHROMA_DIR}. Run `python -m src.ingest` first."
        )
    return Chroma(
        collection_name = COLLECTION_NAME,
        embedding_function = get_embeddings(),
        persist_directory = str(CHROMA_DIR)
    )
def retrieve(query: str, allowed_companies: list[str], k: int = TOP_K) -> list[Document]:
    if not allowed_companies:
        return []
    
    docs = _get_vectorstore().similarity_search(
        query,
        k=k,
        filter = {"company": {"$in":allowed_companies}}
    )

    return [d for d in docs if d.metadata.get("company") in allowed_companies]



