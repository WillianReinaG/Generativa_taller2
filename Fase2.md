# Fase 2 — Datos empresariales, chunking e indexación en EcoMarket RAG



Este documento cumple la rúbrica de la fase 2 para el caso **EcoMarket**: identifica tipos de documentos relevantes, describe una estrategia de segmentación (**chunking**) coherente con el estudio de caso y explicita cómo la **calidad de los documentos** y la forma en que se **dividen** influyen directamente en el rendimiento del sistema RAG.



---



## 1. Tipos de documentos relevantes



En la implementación actual coexisten **varias naturalezas documentales**. Esto es deliberado: el asistente debe resolver consultas operativas (pedidos) y normativas (políticas), más preguntas generales complementarias.



| # | Tipo documental | Ejemplo en el proyecto | Relevancia para EcoMarket |

|---|----------------|------------------------|---------------------------|

| 1 | **Datos tabulares / hoja de cálculo** | `data/pedidos_ejemplo.xlsx` | Estado de pedidos, transportadora, fechas y trazabilidad; preguntas de “¿dónde está mi pedido?” |

| 2 | **Política corporativa en PDF (normativa)** | `data/politica_devoluciones.pdf` | Reglas de devolución, exclusiones y pasos; respuestas alineadas a política oficial |

| 3 | **Política corporativa en PDF (postventa)** | `data/politica_garantia.pdf` | Alcance de garantía, exclusiones y procedimiento; complementa devoluciones sin mezclar conceptos |

| 4 | **Conocimiento complementario en Markdown** | `kb/faq_general.md` (y otros `.md`) | Preguntas frecuentes y tono de marca; reduce huecos cuando la consulta es general |



Así se cubren **tres dimensiones mínimas exigidas por la rúbrica** (operativo, normativo de devoluciones, normativo de garantía) y además una cuarta fuente textual que mejora cobertura en FAQ.



### 1.1 Calidad de los documentos y preparación previa



La calidad **antes** del chunking determina el techo de rendimiento del RAG:



- **Excel**: si las columnas son coherentes y cada fila representa un pedido, el texto derivado es estable y recuperable por identificadores (`order_id` en metadata).

- **PDF**: la extracción de texto (`pypdf`) depende del PDF generado; textos cortos y bien estructurados reducen ruido. Si el texto extraído pierde saltos de línea o títulos, los embeddings pueden mezclar temas distintos en el mismo chunk.

- **Markdown en `kb/`**: suele ser la fuente más limpia para lectura humana y para indexación; conviene mantener redacción clara y evitar duplicar exactamente lo mismo que ya está en `data/` en el mismo índice (en el código se omiten algunos `.md` generados para evitar duplicados con Excel/PDF).



---



## 2. Estrategia de segmentación (chunking): lógica para el caso de estudio



### 2.1 Unidad semántica previa al splitter



Antes de aplicar el splitter fijo, el proyecto ya impone una **lógica de unidades**:



1. **Pedidos (Excel):** cada **fila** se convierte en un `Document` de LangChain con líneas `clave: valor` y metadata (`source`, `order_id`). Una fila suele caber en un solo chunk; si la fila fuera muy larga, el splitter podría partirla, pero el diseño típico EcoMarket mantiene filas compactas.

2. **Políticas (PDF):** el contenido completo del PDF se concatena por páginas en un texto continuo y **ahí sí** el tamaño del documento puede superar ampliamente el tamaño de chunk de políticas; el splitter garantiza fragmentos recuperables.



Esta combinación (“unidad natural” en pedidos + “texto largo partido” en políticas) es una estrategia **coherente con el caso**: las preguntas por pedido buscan una fila concreta; las preguntas por política buscan párrafos dentro de un bloque normativo.



### 2.2 Segmentación por tamaño fijo con solapamiento



En `rag_ejemplo.py` se usan **dos** configuraciones de `RecursiveCharacterTextSplitter`:



- **Pedidos (filas del Excel):** `CHUNK_SIZE_ORDERS = 3200`, `CHUNK_OVERLAP_ORDERS = 80` — prioriza mantener **toda la fila del pedido** en el menor número de chunks posible.

- **PDF y Markdown:** `CHUNK_SIZE_POLICY = 900`, `CHUNK_OVERLAP_POLICY = 140` — fragmentos algo más pequeños para políticas largas y FAQs, con solapamiento para listas y pasos.



**Justificación para EcoMarket:**



| Necesidad del caso | Cómo lo cubre esta estrategia |

|--------------------|-------------------------------|

| Recuperar el fragmento correcto ante preguntas específicas (“¿qué pasa con higiene personal?”) | En políticas, chunks no demasiado grandes → embedding más focalizado |

| Evitar cortar listas o pasos entre dos chunks sin continuidad | Solapamiento en políticas → continuidad entre trozos vecinos |

| Mantener costo y complejidad acotados en proyecto académico | Dos reglas claras (pedidos vs políticas) sin motor de segmentación semántica |

| Pedidos por ID (`ORD-xxxxx`) | Filas amplias en Excel + recuperación por metadata `order_id` en consulta (además del chunking) |

| Homogeneizar pipeline | Misma familia de splitter (`RecursiveCharacterTextSplitter`), parámetros distintos por tipo de fuente |



### 2.3 Comparación frente a alternativas (por qué no se eligieron como base)



| Alternativa | Por qué es menos adecuada como estrategia principal aquí |

|-------------|-----------------------------------------------------------|

| **Documento completo sin partir** | Embeddings demasiado globales; empeora precisión en preguntas acotadas |

| **Chunks muy pequeños (por ejemplo ~200 caracteres)** | Pierden contexto normativo; aumentan falsos positivos en recuperación |

| **Solo chunking semántico por secciones** | Mejor para PDFs largos heterogéneos, pero eleva coste de desarrollo y mantenimiento fuera del alcance del prototipo actual |



La estrategia elegida es por tanto **lógica y defendible**: primero respeta unidades naturales donde existen (filas de pedido), y donde el texto es continuo (PDF), aplica un particionado estable con solapamiento.



---



## 3. Cómo la calidad de los documentos y el chunking afectan directamente el rendimiento del RAG



El rendimiento percibido del asistente depende de una cadena corta:



```

Calidad del texto fuente → calidad del chunk → calidad del embedding → calidad de la recuperación → calidad de la respuesta generada

```



### 3.1 Efectos de la calidad documental



- **Texto ambiguo o incompleto** (p. ej. PDF con extracción pobre): el embedding representa “ruido”; sube la probabilidad de recuperar un chunk equivocado y de que el LLM complete con generalidades o alucine detalles.

- **Datos tabulares sucios** (celdas vacías, IDs duplicados): la pregunta del usuario puede alinearse mal con la fila correcta; afecta recall@k incluso con un buen modelo de embeddings.

- **Redacción clara y estructurada**: mejora la separabilidad semántica entre chunks y reduce mezcla de temas (devoluciones vs garantía).



### 3.2 Efectos del chunking



- **Chunks demasiado grandes** → el vector promedio “diluye” temas distintos; la consulta acota peor y el modelo recibe contexto con ruido (**baja precisión**, más riesgo de contradicciones).

- **Chunks demasiado pequeños** → falta contexto local; la recuperación puede traer un trozo sin la condición necesaria (**respuestas incompletas**).

- **Solapamiento** → amortigua cortes en listas y pasos numerados; mejora robustez cuando la pregunta usa sinónimos o reformulaciones.



### 3.3 Conexión con el comportamiento del sistema



En este proyecto, el rendimiento también se protege en **consulta**: si la similitud con el índice es baja (`threshold`), el sistema evita inyectar contexto poco relevante al LLM, reduciendo invenciones. Cuando la pregunta incluye un identificador **`ORD-xxxxx`**, el sistema puede **recuperar por metadata `order_id`** en Chroma antes de confiar solo en la similitud semántica, lo que corrige consultas donde la formulación no alinea bien con el embedding pero el pedido existe en la base.

Eso muestra comprensión explícita de que ** ni buenos embeddings ni buen chunking garantizan éxito si la recuperación es débil** para una pregunta concreta — en especial con datos tabulares e IDs estables.



---



## 4. Proceso de indexación (de fragmentos a base vectorial)



El flujo en `rag_ejemplo.py` es:



1. **Carga:** Excel (`_spreadsheet_orders_to_documents()`), PDF (`_pdf_documents()`), Markdown complementario (`_iter_markdown_docs()`).

2. **Unificación:** `_all_documents()` concatena los `Document`.

3. **Fragmentación:** dos splitters (`CHUNK_SIZE_ORDERS` / `CHUNK_SIZE_POLICY`) según proceda Excel vs PDF/Markdown.

4. **Vectorización:** `OpenAIEmbeddings` (`text-embedding-3-small` por defecto).

5. **Persistencia:** Chroma en `data/chroma_db/` tras limpiar índices previos para evitar duplicados.



Se conserva metadata (`source`, `order_id`) para trazabilidad y mejoras futuras (p. ej. filtrado por pedido).



---



## 5. Punto adicional: control de calidad del contexto antes de generar



El flujo `ask()` combina dos rutas de control de calidad: (1) relevancia semantica (`similarity_search_with_relevance_scores`) con **umbral** y (2) recuperacion por metadata cuando detecta `ORD-xxxxx` (filtro por `order_id`). Esto cierra el ciclo de calidad: **chunking + embeddings + recuperación + política de uso del contexto**.



---



## Cierre



EcoMarket combina **al menos tres tipos documentales relevantes** (tabular, política PDF de devoluciones, política PDF de garantía), más contenido Markdown de apoyo. La segmentación aplicada es **consistente con el caso**: respeta unidades naturales en pedidos y parte de forma estable los textos normativos largos, con solapamiento para robustez. La **calidad del texto fuente** y la **granularidad del chunk** determinan directamente si la recuperación semántica trae evidencia útil y si el modelo puede responder con precisión sin inventar información.


