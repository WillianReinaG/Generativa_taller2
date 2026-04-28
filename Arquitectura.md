# Arquitectura del proyecto EcoMarket RAG

Este documento resume la estructura y arquitectura específica del proyecto `entregacuatro`.

## 1) Estructura principal del proyecto

```text
entregacuatro/
├─ data/
│  ├─ pedidos_ejemplo.xlsx
│  ├─ politica_devoluciones.pdf
│  ├─ politica_garantia.pdf
│  └─ settings-final.toml
├─ kb/
│  ├─ faq_general.md
│  ├─ pedidos.md
│  ├─ politica_devoluciones.md
│  ├─ politica_garantia.md
│  └─ kb_desde_toml.md
├─ scripts/
│  ├─ build_data_assets.py
│  └─ build_kb_from_json.py
├─ rag_ejemplo.py
├─ README.md
├─ Fase1.md
└─ Fase2.md
```

## 2) Arquitectura funcional (RAG)

```mermaid
flowchart TD
    A[data/pedidos_ejemplo.xlsx] --> ING[ingest en rag_ejemplo.py]
    B[data/politica_devoluciones.pdf] --> ING
    C[data/politica_garantia.pdf] --> ING
    D[kb/*.md complementarios] --> ING

    ING --> SPLIT[RecursiveCharacterTextSplitter\nCHUNK_SIZE=800\nCHUNK_OVERLAP=120]
    SPLIT --> EMB[OpenAIEmbeddings\ntext-embedding-3-small]
    EMB --> VDB[(Chroma DB\n data/chroma_db)]

    Q[Pregunta del usuario] --> RET[Recuperación semántica\nk=4 + threshold]
    VDB --> RET
    RET --> PROMPT[Prompt con reglas de negocio\ncontext_status]
    PROMPT --> LLM[ChatOpenAI\ngpt-4o-mini]
    LLM --> R[Respuesta final + Fuentes]
```

## 3) Flujo operativo recomendado

1. Instalar dependencias: `pip install -r requirements.txt`
2. (Si aplica) migrar datos: `python scripts/build_data_assets.py`
3. Regenerar KB markdown: `python scripts/build_kb_from_json.py`
4. Indexar: `python rag_ejemplo.py ingest`
5. Consultar: `python rag_ejemplo.py ask -q "..."` o `python rag_ejemplo.py repl`

## 4) Puntos clave de diseño

- Se combinan fuentes estructuradas (Excel) y no estructuradas (PDF/Markdown).
- El chunking fijo con solapamiento equilibra precisión y contexto.
- La recuperación usa umbral de relevancia para evitar respuestas con contexto débil.
- Si no hay contexto confiable, el asistente informa límites y sugiere escalar a humano.
