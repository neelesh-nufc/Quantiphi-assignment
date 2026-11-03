from functools import lru_cache 

from langchain_core.documents import Document 
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_groq import ChatGroq

from src.config import COMPANY_NAMES, GROQ_MODEL, MAX_HISTORY_MESSAGES
from src.retriever import retrieve


CONDENSE_SYSTEM = (
    "Given a conversation and a follow-up question, rewrite the follow-up as a "
    "standalone question that can be understood without the conversation. "
    "Include any company names, time periods or metrics it refers to. "
    "Do NOT answer the question. If it is already standalone, return it unchanged. "
    "Return only the rewritten question."
)

ANSWER_SYSTEM = (
    "You are an assistant that answers questions about company earnings call transcripts.\n"
    "Rules:\n"
    "- Answer using ONLY the context below. Never use outside or prior knowledge.\n"
    "- If the context does not contain the answer, say you don't have that information "
    "in the documents you have access to. Do not guess.\n"
    "- Cite the sources you used with their numbers, e.g. [1], [2].\n"
    "- Quote figures exactly as they appear in the context.\n"
    "- Be concise.\n\n"
    "The user has access to documents for: {companies}.\n\n"
    "Context:\n{context}"
)

@lru_cache(maxsize=1)
def _get_llm() -> ChatGroq:
    return ChatGroq(model=GROQ_MODEL, temperature=0)


def _to_messages(history: list[dict]) -> list[BaseMessage]:
    recent = history[-MAX_HISTORY_MESSAGES:]
    return [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in recent
    ]


def condense_question(question: str, history: list[dict]) -> str:
    if not history:
        return question

    prompt = ChatPromptTemplate.from_messages([
        ("system", CONDENSE_SYSTEM),
        MessagesPlaceholder("history"),
        ("human", "Follow-up question: {question}\n\nStandalone question:"),
    ])
    chain = prompt | _get_llm() | StrOutputParser()
    return chain.invoke({"history": _to_messages(history), "question": question}).strip()


def _format_context(docs: list[Document]) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        m = doc.metadata
        name = COMPANY_NAMES.get(m["company"], m["company"])
        blocks.append(f"[{i}] {name} | {m['quarter']} | page {m['page']}\n{doc.page_content}")
    return "\n\n".join(blocks)

def answer_question(question: str, history: list[dict], allowed_companies: list[str]) -> dict:
    """Full conversational RAG turn: condense -> retrieve (access-filtered) -> answer."""
    standalone = condense_question(question, history)
    docs = retrieve(standalone, allowed_companies)

    if not docs:
        return {
            "answer": "I couldn't find anything relevant in the documents you have access to.",
            "sources": [],
            "standalone_question": standalone,
        }

    prompt = ChatPromptTemplate.from_messages([
        ("system", ANSWER_SYSTEM),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])
    chain = prompt | _get_llm() | StrOutputParser()
    answer = chain.invoke({
        "companies": ", ".join(COMPANY_NAMES.get(c, c) for c in allowed_companies),
        "context": _format_context(docs),
        "history": _to_messages(history),
        "question": question,
    })

    return {"answer": answer, "sources": docs, "standalone_question": standalone}