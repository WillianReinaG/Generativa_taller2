# Fase 1 — Selección de componentes clave del sistema RAG (EcoMarket)

Este documento identifica los **componentes principales** de la arquitectura RAG del proyecto **EcoMarket** (`entregacuatro`), relaciona las decisiones con lo implementado en el repositorio y fundamenta opciones para un despliegue más cercano a producción.

**Resumen del stack actual (referencia):**

| Componente | Elección en el prototipo |
|------------|--------------------------|
| Embeddings | OpenAI `text-embedding-3-small` (configurable vía `OPENAI_EMBEDDING_MODEL`) |
| Base vectorial | Chroma persistida en `data/chroma_db/` |
| Orquestación | LangChain (`rag_ejemplo.py`) |
| Modelo generativo | OpenAI Chat (`gpt-4o-mini` por defecto, `OPENAI_MODEL`) |
| Fragmentación | `RecursiveCharacterTextSplitter` (pedidos ~3200/80; PDF/Markdown ~900/140) |

---

## 1. Modelo de embeddings

**Pregunta:** ¿Qué modelo utilizarían para convertir los documentos de la empresa en vectores numéricos? Justificar por **precisión**, **costo** y **manejo del español**. ¿Modelo abierto (p. ej. Hugging Face) o propietario?

### Opciones típicas

| Enfoque | Ejemplos | Precisión / idioma | Costo |
|---------|----------|---------------------|-------|
| **Propietario (API)** | OpenAI `text-embedding-3-small` / `large`, Cohere, Google | Muy buen rendimiento multilingüe (incluido español) en benchmarks de recuperación; `large` suele ganar a `small` en tareas difíciles | Coste por millón de tokens; sin GPU propia |
| **Abierto (local / servidor propio)** | `intfloat/multilingual-e5-large`, `sentence-transformers` multilingües | Buen español si el modelo es **multilingual**; hay que validar en **tus** dominios (pedidos, políticas) | Infra (GPU opcional), licencias MIT/Apache; coste operativo propio |

### Elección recomendada para EcoMarket

- **Prototipo y MVP:** mantener **`text-embedding-3-small`** (como en este proyecto): excelente relación **costo / latencia / calidad**, soporte multilingüe sólido y sin mantener clusters de inferencia de embeddings.
- **Escenario compliance / datos sensibles:** valorar embeddings **abiertos** ejecutados **on-premise** (p. ej. familia E5 multilingüe o modelos locales vía Hugging Face + servidor de inferencia) para que los textos no salgan a un tercero; el trade-off es **operación y benchmark** propios.

**Justificación breve:** para atención al cliente en español con políticas y pedidos, la prioridad es **recuperación estable** y coste predecible; los embeddings pequeños de OpenAI suelen ser suficientes. Si el volumen crece o hay requisitos de soberanía de datos, los modelos abiertos multilingües son la alternativa natural.

---

## 2. Base de datos vectorial

**Pregunta:** ¿Dónde almacenar los vectores para una búsqueda por similitud eficiente? Comparar **Pinecone**, **ChromaDB** y **Weaviate** en **escalabilidad**, **costo** y **facilidad de uso** para EcoMarket.

### Comparativa orientada al caso EcoMarket

| Criterio | **ChromaDB** | **Pinecone** | **Weaviate** |
|----------|--------------|--------------|--------------|
| **Modelo de despliegue** | Embebido o servidor propio; en el proyecto: **persistencia local** en disco | SaaS gestionado (índices en la nube del proveedor) | SaaS o **self-hosted** (Docker/K8s) |
| **Escalabilidad** | Adecuada para volúmenes modestos y equipos pequeños; para millones de vectores y QPS altos puede quedarse corta sin diseño cuidadoso | Muy buena para escalar en la nube sin operar shards | Muy buena en cluster; más piezas que operar si es self-hosted |
| **Costo** | Bajo coste directo (principalmente compute propio); ideal para **desarrollo y prototipos** | Coste recurrente por uso/tier; predecible pero suma en producción | Variable: managed cobra; self-hosted implica infra |
| **Facilidad de uso** | **Muy alta** en Python; encaja con LangChain/LlamaIndex como en este repo | Alta; menos ops, pero vendor lock-in y datos en servicio externo | Media-alta; más conceptos (schemas, módulos); muy potente si ya tenéis DevOps |

### Recomendación por fase

- **Fase actual (académica / piloto):** **ChromaDB** — coincide con `data/chroma_db/`, cero cuentas cloud obligatorias, iteración rápida.
- **EcoMarket en crecimiento con varios equipos y SLA:** valorar **Pinecone** o **Weaviate Cloud** si queréis delegar escalado y backups; si la política de datos exige **control total**, **Weaviate self-hosted** o Chroma como servicio interno.

---

## 3. Modelo generativo (LLM) para la fase de generación

**Pregunta:** Qué modelo usa el sistema para **sintetizar** la respuesta final a partir de la pregunta y los fragmentos recuperados, y cómo elegirlo.

### Rol en el pipeline RAG

El LLM no “conoce” la base interna por sí mismo: condiciona la respuesta al **contexto recuperado** (y a las reglas del prompt). Por tanto, interesan: **cumplimiento de instrucciones**, **menor alucinación** cuando el contexto es ambiguo, **latencia** y **costo por consulta**.

### Opciones

| Tipo | Ejemplos | Ventajas para EcoMarket | Desventajas |
|------|----------|-------------------------|-------------|
| **Propietario API** | `gpt-4o-mini`, GPT-4.1, Claude, etc. | Calidad alta en español; poco ops (como `ChatOpenAI` en `rag_ejemplo.py`) | Coste por token; dependencia del proveedor |
| **Abierto self-hosted** | Llama 3.x, Mistral, Qwen | Control de datos y coste marginal por consulta si ya hay GPU | Necesidad de cuantizar, evaluar seguridad del modelo y mantener servicio |

### Elección alineada con el proyecto

- Por defecto se usa **`gpt-4o-mini`** con **temperatura 0** para respuestas más deterministas y alineadas al contexto de atención al cliente.
- Para producción, conviene **evaluar** modelos en escenarios reales (preguntas fuera de política, pedidos inexistentes, límites de herramientas) y medir **tasa de rechazo correcto** frente a invenciones.

---

## 4. Orquestación del pipeline y estrategia de fragmentación (chunking)

**Pregunta:** Cómo se encadenan **ingesta → embeddings → almacén → recuperación → prompt → LLM** y cómo se **parten** los documentos antes de indexar.

### Orquestación (framework)

| Opción | Descripción | EcoMarket |
|--------|-------------|-----------|
| **LangChain** | Abstracciones para documentos, vector stores, prompts y cadenas | **Usada** en `rag_ejemplo.py`; integración directa con OpenAI y Chroma |
| **LlamaIndex** | Especializada en índices y consulta sobre carpetas de documentos | **Notebook** `notebooks/rag_llamaindex.ipynb` para comparar el mismo flujo |
| **Código mínimo sin framework** | Máximo control, más mantenimiento | Posible en producción muy acotada; menos recomendable para equipos que ya usan LangChain |

Para este proyecto, **LangChain** reduce fricción y documentación; **LlamaIndex** es útil si el siguiente paso son índices jerárquicos o agentes sobre la misma KB.

### Fragmentación de textos

En `ingest()` se usan **dos configuraciones** de `RecursiveCharacterTextSplitter`: una para pedidos del Excel (chunks mas amplios, preservando filas completas) y otra para PDF/Markdown (chunks mas cortos con mayor solapamiento). Esto equilibra **granularidad** (acertar el parrafo correcto en politicas) y **contexto** (evitar cortes destructivos), sin perder trazabilidad de pedidos por `order_id`.

| Decisiones de diseño | Impacto |
|----------------------|--------|
| Chunk **amplio para pedidos (Excel)** | Favorece recuperar el registro completo de un order_id con menos fragmentacion |
| Chunk **mas compacto para PDF/Markdown** | Mejora precision semantica en politicas y FAQ, reduciendo ruido |
| **Solapamiento** | Reduce cortes arbitrarios entre chunks consecutivos en textos largos |

**Mejoras futuras:** chunking semantico por secciones (titulos/listas) en PDF y filtros adicionales por metadata (`carrier`, `categoria`, etc.) para consultas operativas mas finas.

---

## 5. Criterios medibles de evaluación (para validar la decisión)

Para demostrar que la selección de embeddings y base vectorial no es solo teórica, se propone medir:

| Métrica | Qué evalúa | Meta inicial sugerida |
|---------|------------|-----------------------|
| **Recall@4** | Si el contexto correcto aparece entre los 4 chunks recuperados | >= 0.85 en preguntas de pedidos/políticas |
| **MRR@4** | Qué tan arriba aparece el chunk correcto | >= 0.70 |
| **Tasa de rechazo correcto** | Casos donde el sistema responde “no tengo herramientas/información” cuando corresponde | >= 0.90 |
| **Latencia p95 (consulta)** | Experiencia de usuario en atención al cliente | <= 3 s en entorno objetivo |
| **Costo por 1,000 consultas** | Sostenibilidad económica | Seguimiento mensual y umbral por presupuesto del curso/negocio |

Estas métricas conectan directamente con la eficacia RAG: mejor recuperación reduce alucinación, y menor latencia/costo mejora viabilidad operativa.

---

## 6. Matriz de decisión ponderada (embeddings + vector DB)

Pesos usados (alineados a la rúbrica y al caso EcoMarket):  
**Español/precisión 30%**, **Costo 25%**, **Escalabilidad 20%**, **Latencia 15%**, **Facilidad operativa 10%**.

### 6.1 Embeddings

Escala de puntaje: 1 (bajo) a 5 (alto).  
Calculo: **Puntaje total = sum(puntaje x peso) / 5**.

| Opción | Español/precisión (30) | Costo (25) | Escalabilidad (20) | Latencia (15) | Operación (10) | **Puntaje total / 100** |
|--------|--------------------------|------------|--------------------|---------------|----------------|-------------------------|
| **OpenAI `text-embedding-3-small`** | 4.5 | 4.0 | 4.0 | 4.0 | 4.5 | **84.5** |
| OpenAI `text-embedding-3-large` | 4.8 | 2.8 | 4.0 | 3.6 | 4.5 | 77.9 |
| Modelo abierto multilingüe (E5/sentence-transformers) | 3.8 | 3.5 | 3.8 | 3.2 | 2.8 | 70.7 |

**Interpretación:** `text-embedding-3-small` ofrece el mejor balance global para esta fase (calidad sólida en español con costo y operación controlados).

### 6.2 Base vectorial

| Opción | Español/precisión (30) | Costo (25) | Escalabilidad (20) | Latencia (15) | Operación (10) | **Puntaje total / 100** |
|--------|--------------------------|------------|--------------------|---------------|----------------|-------------------------|
| **Chroma (local)** | 4.0 | 4.8 | 3.2 | 4.2 | 4.7 | **83.7** |
| Pinecone (SaaS) | 4.2 | 3.2 | 4.8 | 4.4 | 4.2 | 82.0 |
| Weaviate (cloud/self-hosted) | 4.2 | 3.5 | 4.6 | 4.0 | 3.4 | 80.6 |

**Interpretación:** para el tamaño y objetivo académico del proyecto, Chroma local maximiza simplicidad y costo-eficiencia. Pinecone/Weaviate ganan valor en escenarios de gran escala y SLAs exigentes.

---

## 7. Riesgos por componente y mitigaciones

1. **Embeddings API (vendor lock-in y costos variables)**
   - Riesgo: dependencia del proveedor y variación de costos por volumen.
   - Mitigación: parametrizar modelo en `.env`, medir costo por 1,000 consultas y mantener plan alterno con embeddings abiertos.

2. **Vector DB local (límite de escalabilidad)**
   - Riesgo: crecimiento de latencia cuando aumenten documentos, concurrencia y consultas.
   - Mitigación: definir umbrales de migración (número de chunks, p95 de latencia) hacia Pinecone/Weaviate si se supera capacidad local.

3. **Calidad de recuperación en español (drift semántico)**
   - Riesgo: consultas ambiguas o con sinónimos regionales pueden recuperar contexto débil.
   - Mitigación: evaluación periódica con set de preguntas reales, ajuste de chunking y uso de metadata/filtros por `order_id`.

---

## Cierre

La decisión técnica para esta fase queda formalmente definida así:

- **Embedding seleccionado:** `text-embedding-3-small`.
- **Base vectorial seleccionada:** **Chroma local**.

Esta combinación maximiza la eficacia práctica del RAG en EcoMarket porque prioriza:  
1) buen rendimiento en español, 2) costo controlado, 3) implementación simple y estable, y 4) ruta clara de escalamiento futuro con métricas objetivas.
