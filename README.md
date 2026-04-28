# EcoMarket RAG - entrega cuatro

Proyecto taller 2 con flujo RAG.

integrantes:
MANUEL GONZALES GONZALEZ
JUAN MANUEL HURTADO
WILLIAN ALBERTO REINA GARCIA

## Objetivo

Permitir que el asistente:

1. Consulte una base interna antes de responder.
2. Responda con evidencia cuando exista contexto.
3. Declare limitacion cuando no tenga herramientas o informacion suficiente.

## Arquitectura

1. **Fuentes en `data/`**: `pedidos_ejemplo.xlsx` (hoja de calculo), `politica_devoluciones.pdf`, `politica_garantia.pdf`, `settings-final.toml`. Markdown opcional en `kb/`.
2. **Migracion inicial (solo si aun tienes JSON legacy)**: `python scripts/build_data_assets.py` convierte `pedidos_ejemplo.json` -> `.xlsx` y `politica_devoluciones.json` -> `.pdf`, genera `politica_garantia.pdf` y elimina esos JSON.
3. **Markdown para el notebook**: `scripts/build_kb_from_json.py` genera `kb/*.md` desde el Excel y los PDF (nombre historico del script).
4. **Embeddings**: OpenAI (`text-embedding-3-small` por defecto).
5. **Fragmentacion**: en `rag_ejemplo.py`, `RecursiveCharacterTextSplitter` con tamanos distintos para **pedidos (Excel)** vs **PDF/Markdown**; en consultas con `ORD-xxxxx` se usa filtro por metadata `order_id` antes de confiar solo en similitud semantica.
6. **Vector DB**: Chroma local en `data/chroma_db/`.
7. **Generacion**: LangChain retrieval chain (`rag_ejemplo.py`).

## Instalacion

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Configura variables en `.env` (puedes copiar desde `.env.example`).

## Uso rapido

0) (Solo una vez, si el repo aun trae JSON en lugar de xlsx/pdf) Migrar datos:

```powershell
python scripts/build_data_assets.py
```

1) Generar documentos KB para LlamaIndex / lectura:

```powershell
python scripts/build_kb_from_json.py
```

2) Indexar:

```powershell
python rag_ejemplo.py ingest
```

3) Preguntar:

```powershell
python rag_ejemplo.py ask -q "Cual es el estado del pedido ORD-00001?"
```

4) Modo interactivo:

```powershell
python rag_ejemplo.py repl
```

## Notebook alterno con LlamaIndex

Se incluye `notebooks/rag_llamaindex.ipynb` para comparar el mismo enfoque RAG usando LlamaIndex sobre la carpeta `kb/`.

## Notas

- Este proyecto prioriza claridad academica de piezas RAG sobre optimizacion avanzada.
- Para mejorar precision en pedidos, `ask()` ya aplica recuperacion por metadata `order_id` cuando detecta patrones `ORD-xxxxx`; aun asi se puede extender a filtros por otras columnas (carrier, categoria, etc.).
