"""
RAG de ejemplo para EcoMarket con LangChain + Chroma.

Uso:
  python rag_ejemplo.py ingest
  python rag_ejemplo.py ask -q "Cual es el estado del pedido ORD-00001?"
  python rag_ejemplo.py repl
"""
# indexa textos de data/ y kb/ en Chroma con embeddings de OpenAI, y responde preguntas con un LLM que solo debe usar el contexto recuperado.
# lee de data/, kb/ y escribe  el indice en chroma_db/
#langChain: documentos (Document), splitter de texto, embedding y chat de OpenAI, plantilla de prompt, salida
#chroma: intenta con langchain_chroma y si falla usa langchain_community.vectorstores.Chroma
#.env: la clave y modelo opcionales se cargan en main() con load _dotenv.
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_chroma import Chroma  # paquete nuevo (si esta instalado)
except Exception:  # pragma: no cover
    from langchain_community.vectorstores import Chroma


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
KB_DIR = ROOT / "kb"
DB_DIR = DATA_DIR / "chroma_db"

# Markdown en kb/ generados desde xlsx/pdf (scripts/build_kb_from_json.py): no reindexar para evitar duplicados.
_SKIP_KB_MD = frozenset(
    {"pedidos.md", "politica_devoluciones.md", "politica_garantia.md"},
)

# Fragmentacion: pedidos (filas Excel) suelen caber en un solo chunk amplio; PDF/markdown largos en trozos mas pequenos.
CHUNK_SIZE_ORDERS = 3200
CHUNK_OVERLAP_ORDERS = 80
CHUNK_SIZE_POLICY = 900
CHUNK_OVERLAP_POLICY = 140

_ORDER_ID_RE = re.compile(r"\bORD-\d+\b", re.IGNORECASE)


def _require_openai_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Falta OPENAI_API_KEY. Copia .env.example a .env y agrega tu clave."
        )


def _iter_markdown_docs() -> list[Document]:
    docs: list[Document] = []
    for path in sorted(KB_DIR.glob("*.md")):
        if path.name in _SKIP_KB_MD:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": str(path.name)}))
    return docs


def _extract_pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n\n".join(parts).strip()


def _pdf_documents() -> list[Document]:
    docs: list[Document] = []
    for path in sorted(DATA_DIR.glob("*.pdf")):
        text = _extract_pdf_text(path)
        if text:
            docs.append(
                Document(page_content=text, metadata={"source": path.name}),
            )
    return docs


def _spreadsheet_orders_to_documents() -> list[Document]:
    import pandas as pd

    path = DATA_DIR / "pedidos_ejemplo.xlsx"
    if not path.is_file():
        return []

    df = pd.read_excel(path, sheet_name=0)
    docs: list[Document] = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        oid = row_dict.get("order_id") or row_dict.get("pedido_id")
        if oid is None or (isinstance(oid, float) and pd.isna(oid)):
            oid = ""
        else:
            oid = str(oid).strip()
        lines: list[str] = []
        if oid:
            lines.append(f"pedido_id: {oid}")
        for k, v in row_dict.items():
            key = str(k)
            if key in ("order_id", "pedido_id"):
                continue
            if pd.isna(v):
                continue
            lines.append(f"{key}: {v}")
        if len(lines) <= 1 and not oid:
            continue
        meta_oid = oid or "desconocido"
        docs.append(
            Document(
                page_content="\n".join(lines),
                metadata={"source": "pedidos_ejemplo.xlsx", "order_id": meta_oid},
            ),
        )
    return docs


def _all_documents() -> list[Document]:
    return _spreadsheet_orders_to_documents() + _pdf_documents() + _iter_markdown_docs()


def _extract_order_id(text: str) -> str | None:
    m = _ORDER_ID_RE.search(text)
    return m.group(0).upper() if m else None


def _embedding_model() -> OpenAIEmbeddings:
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model)


def _chat_model() -> ChatOpenAI:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=0)


def ingest() -> None:
    _require_openai_key()
    order_docs = _spreadsheet_orders_to_documents()
    policy_docs = _pdf_documents() + _iter_markdown_docs()
    if not order_docs and not policy_docs:
        raise RuntimeError(
            "No hay documentos fuente. Agrega contenido a data/ o kb/ y vuelve a intentar."
        )

    splitter_orders = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_ORDERS,
        chunk_overlap=CHUNK_OVERLAP_ORDERS,
    )
    splitter_policy = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_POLICY,
        chunk_overlap=CHUNK_OVERLAP_POLICY,
    )
    chunks: list[Document] = []
    if order_docs:
        chunks.extend(splitter_orders.split_documents(order_docs))
    if policy_docs:
        chunks.extend(splitter_policy.split_documents(policy_docs))
    embeddings = _embedding_model()

    if DB_DIR.exists():
        # Reindexado limpio para evitar duplicados de corridas previas.
        for p in DB_DIR.rglob("*"):
            if p.is_file():
                p.unlink()
        for p in sorted(DB_DIR.rglob("*"), reverse=True):
            if p.is_dir():
                p.rmdir()

    DB_DIR.mkdir(parents=True, exist_ok=True)
    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=str(DB_DIR))
    print(f"Indice creado en {DB_DIR} con {len(chunks)} chunks.")


def _vectorstore() -> Chroma:
    _require_openai_key()
    if not DB_DIR.exists():
        raise RuntimeError("No existe el indice. Ejecuta: python rag_ejemplo.py ingest")
    return Chroma(persist_directory=str(DB_DIR), embedding_function=_embedding_model())


def _prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(
        """Eres el asistente de atencion al cliente de EcoMarket.
Debes responder a cualquier mensaje del cliente (consultas sobre pedidos o politicas, saludos, agradecimientos,
quejas generales, temas ajenos a la tienda o peticiones que requieran sistemas que tu no tienes).

Idioma: espanol claro, breve y respetuoso.

Diagnostico de la base interna (obedece este bloque al decidir como actuar):
{context_status}

Reglas:
1) Si el diagnostico es CONTEXTO_CONFIABLE y el bloque "Contexto interno" no esta vacio, usa exclusivamente
   ese texto para datos sobre pedidos, politicas, FAQs o ajustes documentados. Puedes parafrasear sin inventar
   hechos nuevos.
2) Si el diagnostico es CONTEXTO_NO_CONFIABLE o SIN_CONTEXTO_UTIL, o el contexto interno esta vacio: no inventes
   pedidos, estados de envio, montos, plazos ni politicas. Explica de forma concreta que no cuentas con
   herramientas ni informacion suficiente en esta interfaz para resolver esa solicitud y, si aplica, que
   temas de cuenta, facturacion, incidencias concretas o cambios en sistemas deben canalizarse con un agente
   humano. Para saludos o gratitud sin pregunta operativa, responde brevemente sin prometer acciones que no
   puedas cumplir.
3) No finjas acceso a sistemas externos, enlaces de pago, ejecucion de devoluciones ni modificacion de datos;
   solo orientas con la base cuando el diagnostico lo permite.

Contexto interno (puede estar vacio):
{context}

Mensaje del cliente:
{input}
"""
    )


def ask(question: str, *, threshold: float = 0.2) -> str:
    vs = _vectorstore()
    llm = _chat_model()

    order_id = _extract_order_id(question)
    context_docs: list[Document] = []
    used_order_filter = False

    if order_id:
        try:
            filtered = vs.similarity_search(
                question,
                k=8,
                filter={"order_id": order_id},
            )
        except Exception:
            filtered = []
        if filtered:
            context_docs = filtered
            used_order_filter = True
        else:
            try:
                filtered = vs.similarity_search(
                    f"pedido_id: {order_id}",
                    k=6,
                )
            except Exception:
                filtered = []
            filtered = [d for d in filtered if order_id in (d.page_content or "")]
            if filtered:
                context_docs = filtered
                used_order_filter = True

    scored_docs: list[tuple[Document, float]] = []
    best_score: float | None = None
    low_relevance = False

    if not used_order_filter:
        try:
            scored_docs = vs.similarity_search_with_relevance_scores(question, k=4)
        except Exception:
            scored_docs = []

        if scored_docs:
            best_score = max(score for _, score in scored_docs)

        low_relevance = best_score is not None and best_score < threshold

        retriever = vs.as_retriever(search_kwargs={"k": 4})
        context_docs = retriever.invoke(question)

    if context_docs and (used_order_filter or not low_relevance):
        context_text = "\n\n".join(doc.page_content for doc in context_docs)
    else:
        context_text = ""

    if used_order_filter and context_text:
        context_status = (
            "CONTEXTO_CONFIABLE: se localizo informacion del pedido por identificador (order_id). "
            "Usa solo el bloque de contexto interno para datos operativos del pedido."
        )
    elif not used_order_filter and context_text and not low_relevance:
        context_status = (
            "CONTEXTO_CONFIABLE: hay buena coincidencia con la base interna (o no hay puntuacion de relevancia "
            "disponible y se recuperaron fragmentos). Usa solo el bloque de contexto interno para datos operativos."
        )
    elif low_relevance:
        context_status = (
            "CONTEXTO_NO_CONFIABLE: la similitud con la base interna es baja. No uses fragmentos como hechos "
            "verificados. Explica limitaciones y ofrece escalar a un agente humano si la solicitud lo requiere."
        )
    else:
        context_status = (
            "SIN_CONTEXTO_UTIL: no se recupero texto util desde la base. No inventes datos internos. Atiende con "
            "cortesia y aclara que no dispones de herramientas ni informacion aqui para tramitar esa peticion; "
            "indica cuando convenga contactar a atencion humana."
        )

    chain = _prompt() | llm | StrOutputParser()
    answer = chain.invoke(
        {"context": context_text, "input": question, "context_status": context_status}
    ).strip()
    if not answer:
        answer = (
            "No cuento con herramientas o informacion suficiente en la base interna para "
            "resolver esta solicitud."
        )

    used_kb = bool(context_text)
    if used_kb:
        sources = sorted(
            {doc.metadata.get("source", "desconocido") for doc in context_docs if hasattr(doc, "metadata")}
        )
        return f"{answer}\n\nFuentes: {', '.join(sources)}"
    return answer


def repl() -> None:
    print("EcoMarket RAG REPL. Escribe 'salir' para terminar.")
    while True:
        line = input("Tu: ").strip()
        if not line or line.lower() in {"salir", "exit", "quit"}:
            break
        try:
            out = ask(line)
        except Exception as exc:
            out = f"Error: {exc}"
        print(f"Asistente: {out}\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RAG de ejemplo EcoMarket (LangChain)")
    sub = p.add_subparsers(dest="command", required=True)

    s_ingest = sub.add_parser("ingest", help="Indexa documentos de data/ y kb/")
    s_ingest.set_defaults(func=lambda _args: ingest())

    s_ask = sub.add_parser("ask", help="Realiza una consulta al sistema RAG")
    s_ask.add_argument("-q", "--question", required=True, help="Pregunta del cliente")
    s_ask.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Umbral minimo de relevancia (0 a 1); por debajo no se inyecta contexto al LLM (evita alucinar)",
    )
    s_ask.set_defaults(func=lambda a: print(ask(a.question, threshold=a.threshold)))

    s_repl = sub.add_parser("repl", help="Modo interactivo")
    s_repl.set_defaults(func=lambda _args: repl())

    return p


def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
