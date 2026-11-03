# Multi-User Document Search & Conversational Q&A

A Streamlit app where users log in with an email ID and ask questions about company earnings-call transcripts. Each user gets answers **only** from the companies they are authorised to see, and follow-up questions keep the context of the conversation.

Built with Python, LangChain, ChromaDB, a local HuggingFace embedding model, and Groq for the LLM.

> 🎥 **Video demo:** _<add link>_

---

## Features

- **Email-based login (simulated)**: users are defined in [`config/users.yaml`](config/users.yaml) with the companies they can access.
- **Access-controlled retrieval**: the vector search is filtered by the user's allowed companies *inside the database query*, so unauthorised text never reaches the LLM.
- **Conversational Q&A**: follow-ups such as *"How does that compare to last year?"* are rewritten into standalone questions before searching.
- **Grounded answers with citations**: the LLM answers only from retrieved excerpts, cites them as `[1]`, `[2]`, and says so when the answer isn't in the user's documents.
- **Source excerpts in the UI**: every answer shows the company, quarter, page and text of the chunks it used.
- **Multi-user safe**: each browser session has its own login and chat history; several users can use the app at the same time without interfering.

---

## Dataset

Publicly available earnings-call transcripts, stored in [`data/pdfs/`](data/pdfs/):

| File | Company | Quarter |
|---|---|---|
| `tcs_q1_fy27.pdf` | TCS | Q1 FY27 |
| `infosys_q1_fy27.pdf` | Infosys | Q1 FY27 |
| `reliance_q1_fy27.pdf` | Reliance Industries | Q1 FY27 |
| `hdfcbank_q2_fy27.pdf` | HDFC Bank | Q2 FY27 |

Filenames follow `<company>_<quarter>.pdf`; the company ID used for access control is taken from the part before the first underscore.

## Demo users

| Email | Access |
|---|---|
| `alice@email.com` | TCS |
| `bob@email.com` | Infosys, HDFC Bank |
| `charlie@email.com` | Reliance Industries |
| `diana@email.com` | TCS, Reliance Industries |

Any other email is rejected.

---

## Setup

**Prerequisites:** Python 3.11+ and a free Groq API key from [console.groq.com](https://console.groq.com/keys).

### 1. Clone and create an environment

```bash
git clone <repo-url>
cd <repo-folder>

# conda
conda create -n docqa python=3.11 -y
conda activate docqa

# or venv
# python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Add your Groq API key

```bash
cp .env.example .env
```

Edit `.env` and set:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Build the vector store

```bash
python -m src.ingest
```

This parses the PDFs, splits them into chunks, embeds them and saves them to `chroma_db/`. The first run downloads the embedding model (~130 MB). Expected output ends with roughly:

```
Stored 314 chunks in .../chroma_db
Per company: {'hdfcbank': 67, 'infosys': 67, 'reliance': 122, 'tcs': 58}
```

Run it as a module (`python -m src.ingest`), not `python src/ingest.py`, so the `src` imports resolve.

### 4. Start the app

```bash
streamlit run app.py
```

Open http://localhost:8501 and log in with one of the demo emails.

---

## Try it

| # | Log in as | Ask | Expected |
|---|---|---|---|
| 1 | alice | *What was the revenue this quarter?* | TCS revenue, TCS sources only |
| 2 | alice | *How does that compare to last year?* | "🔎 Searched for" shows the rewritten question naming TCS; year-on-year comparison |
| 3 | alice | *What did Reliance say about Jio?* | Refuses: not in the documents she can access |
| 4 | diana | *What did Reliance say about Jio?* | Answers from Reliance's transcript |
| 5 | bob | *What was the net interest margin?* | HDFC Bank sources |
| 6 | charlie | *What were TCS's deal wins?* | Refuses: Charlie only has Reliance |

**Simultaneous users:** open a second browser window in incognito mode, log in as a different user, and chat in both. Each window keeps its own user, permissions and history.

---

## How it works

```
                 ┌──────────────── ingestion (one-off: python -m src.ingest) ────────────────┐
 data/pdfs/*.pdf ──► PyMuPDF text ──► strip headers/footers ──► chunk (1000 / 150 overlap)
                                                                  │  metadata: company, quarter, page
                                                                  ▼
                                               bge-small embeddings ──► ChromaDB (chroma_db/)
                 └───────────────────────────────────────────────────────────────────────────┘

 Streamlit (per-session state)
   login ──► users.yaml ──► allowed_companies
   question ──► 1. condense with chat history (Groq) ──► standalone question
            ──► 2. vector search WHERE company IN allowed_companies ──► top 5 chunks
            ──► 3. answer from those chunks only (Groq) ──► answer + [n] citations + sources
```

### Project structure

```
├── app.py                # Streamlit UI: login, chat, sources, logout
├── config/users.yaml     # user → allowed companies
├── data/pdfs/            # earnings-call transcripts
├── src/
│   ├── config.py         # paths, model names, chunking and retrieval settings
│   ├── auth.py           # user validation and permission lookup
│   ├── ingest.py         # PDF → clean text → chunks → embeddings → Chroma
│   ├── embedding.py      # shared embedding model (used by ingest and retrieval)
│   ├── retriever.py      # access-filtered similarity search
│   └── chat.py           # question condensing, prompt, answer generation
├── requirements.txt
└── .env.example
```

### Design decisions

**Access control is enforced at retrieval, not in the prompt.** Every chunk carries a `company` metadata field, and `retrieve()` passes `{"company": {"$in": allowed_companies}}` to Chroma. Chunks from other companies are never returned, so the LLM cannot leak them even under prompt injection. A second check after the query drops anything outside the allowed set, and an empty permission list returns no results (fail closed).

**The LLM must answer only from the retrieved context.** Large models already know public facts about companies like Reliance from their training data. The system prompt forbids prior knowledge and tells the model to say when the answer isn't in the user's documents, so an unauthorised question gets a refusal rather than an answer from memory.

**Follow-ups are rewritten before searching.** The vector store has no memory: *"How does that compare to last year?"* contains no company or metric. A first LLM call rewrites it, using the last few turns, into a standalone question that retrieval can match. The answer step then uses the original question plus history.

**Per-session state for multi-user isolation.** Login, permissions and chat history live in `st.session_state`, which Streamlit scopes to each browser session. Permissions are looked up once at login and never taken from user input. Logging out clears the whole session. Shared, cached objects (embedding model, vector store, LLM client) hold no user data.

**Ingestion choices.**
- *PyMuPDF* for parsing: the transcripts are text-based with almost no tables, so a fast plain-text extractor that keeps page numbers is enough.
- *Header/footer removal* by detecting lines repeated at the top or bottom of most pages (e.g. `Page 9 of 16`, copyright lines), so boilerplate doesn't pollute chunks.
- *1000-character chunks with 150 overlap* keep a question and its answer together without cutting figures across chunks.
- Each chunk is prefixed with `[Company | Quarter earnings call]` so retrieval can match company names and the LLM can attribute excerpts correctly when a user has access to more than one company.

**Models.**
- Embeddings: `BAAI/bge-small-en-v1.5`, run locally (free, no API key, fast on a laptop CPU or Apple GPU).
- LLM: `openai/gpt-oss-120b` on Groq, at temperature 0 for repeatable, factual answers. Change `GROQ_MODEL` in [`src/config.py`](src/config.py) to use another Groq model.

---

## Configuration

Settings in [`src/config.py`](src/config.py):

| Setting | Default | Purpose |
|---|---|---|
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model for ingestion and search |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 1000 / 150 | Chunking |
| `TOP_K` | 5 | Chunks retrieved per question |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | LLM for condensing and answering |
| `MAX_HISTORY_MESSAGES` | 6 | Recent messages sent as conversation context |

**Adding a company:** drop `<company>_<quarter>.pdf` into `data/pdfs/`, optionally add a display name to `COMPANY_NAMES` in `config.py`, grant access in `users.yaml`, and re-run `python -m src.ingest`.

---

## Limitations and next steps

- Authentication is simulated; production would use OAuth/SSO and signed tokens.
- Chat history lives in memory per session and is lost on refresh; it could be persisted per user in Redis or a database.
- For production scale: a hosted vector database (pgvector, Pinecone), a FastAPI backend separate from the UI, and request rate limiting.
- Retrieval quality could improve with hybrid search (BM25 + vectors) for exact figures, re-ranking, and table-aware parsing for financial statements.
