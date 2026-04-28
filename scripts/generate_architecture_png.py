from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, x, y, w, h, text, fc="#eaf2ff", ec="#3366cc"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=12, linewidth=1.2)
    ax.add_patch(arr)


def main() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.96,
        "EcoMarket RAG - Arquitectura del Proyecto",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )

    box(ax, 0.03, 0.78, 0.2, 0.14, "Fuentes de datos\n- pedidos_ejemplo.xlsx\n- politica_devoluciones.pdf\n- politica_garantia.pdf\n- kb/*.md")
    box(ax, 0.28, 0.78, 0.15, 0.14, "Ingesta\nrag_ejemplo.py")
    box(ax, 0.47, 0.78, 0.18, 0.14, "Chunking\nRecursiveCharacterTextSplitter\nCHUNK_SIZE=800\nCHUNK_OVERLAP=120")
    box(ax, 0.69, 0.78, 0.13, 0.14, "Embeddings\nOpenAI\ntext-embedding-3-small")
    box(ax, 0.84, 0.78, 0.13, 0.14, "Chroma DB\n/data/chroma_db")

    arrow(ax, 0.23, 0.85, 0.28, 0.85)
    arrow(ax, 0.43, 0.85, 0.47, 0.85)
    arrow(ax, 0.65, 0.85, 0.69, 0.85)
    arrow(ax, 0.82, 0.85, 0.84, 0.85)

    box(ax, 0.03, 0.4, 0.15, 0.14, "Pregunta\nusuario")
    box(ax, 0.23, 0.4, 0.23, 0.14, "Recuperacion semantica\nk=4 + threshold")
    box(ax, 0.52, 0.4, 0.2, 0.14, "Prompt\nreglas + context_status")
    box(ax, 0.76, 0.4, 0.16, 0.14, "LLM\nChatOpenAI\ngpt-4o-mini")
    box(ax, 0.84, 0.16, 0.13, 0.14, "Respuesta final\n+ fuentes")

    arrow(ax, 0.18, 0.47, 0.23, 0.47)
    arrow(ax, 0.46, 0.47, 0.52, 0.47)
    arrow(ax, 0.72, 0.47, 0.76, 0.47)
    arrow(ax, 0.84, 0.78, 0.34, 0.54)  # VDB -> Recuperacion
    arrow(ax, 0.84, 0.40, 0.90, 0.30)

    fig.tight_layout()
    fig.savefig("Arquitectura.png", dpi=180)


if __name__ == "__main__":
    main()
