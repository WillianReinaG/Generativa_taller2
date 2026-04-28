"""
Genera fuentes de datos para el RAG:
- Convierte pedidos_ejemplo.json -> pedidos_ejemplo.xlsx (y elimina el JSON si la conversion ok).
- Crea politica_devoluciones.pdf desde politica_devoluciones.json (y elimina el JSON).
- Crea politica_garantia.pdf con texto de garantia EcoMarket.

Ejecutar desde la raiz del proyecto:
  python scripts/build_data_assets.py
"""
from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def _paragraphs_from_policy(policy: dict) -> list[str]:
    blocks: list[str] = []
    blocks.append("Politica de devoluciones EcoMarket")
    blocks.append("")
    if isinstance(policy.get("devolucion_permitida"), list):
        blocks.append("Devolucion permitida")
        for x in policy["devolucion_permitida"]:
            blocks.append(f"- {x}")
        blocks.append("")
    no_dev = policy.get("no_devolucion") or []
    if isinstance(no_dev, list):
        blocks.append("Categorias sin devolucion o con excepciones")
        for item in no_dev:
            if not isinstance(item, dict):
                continue
            cat = item.get("categoria", "")
            motivo = item.get("motivo", "")
            ejemplos = item.get("ejemplos") or []
            ex = ", ".join(str(e) for e in ejemplos) if isinstance(ejemplos, list) else str(ejemplos)
            blocks.append(f"- {cat}: {motivo} (ejemplos: {ex})")
        blocks.append("")
    if isinstance(policy.get("pasos_generales"), list):
        blocks.append("Pasos generales para una devolucion")
        for step in policy["pasos_generales"]:
            blocks.append(f"- {step}")
    return blocks


def _write_pdf_lines(lines: list[str], out_path: Path) -> None:
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        name="Body",
        parent=styles["Normal"],
        fontSize=11,
        leading=14,
        spaceAfter=6,
    )
    story: list = []
    for line in lines:
        if not line.strip():
            story.append(Spacer(1, 0.2 * cm))
            continue
        story.append(Paragraph(escape(line), body))
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    doc.build(story)


def _write_garantia_pdf(out_path: Path) -> None:
    lines = [
        "Politica general de garantia EcoMarket",
        "",
        "Alcance",
        "- Esta politica complementa las condiciones del fabricante y la normativa aplicable.",
        "- La garantia cubre defectos de fabricacion o funcionamiento durante el periodo indicado para cada categoria.",
        "",
        "Periodos orientativos",
        "- Electronicos y electrodomesticos: segun ficha del producto (habitualmente 12 meses salvo indicacion contraria).",
        "- Ropa y calzado: defectos de confeccion o materiales en un plazo razonable desde la entrega (consultar etiqueta).",
        "- Hogar y decoracion: conforme al proveedor; conservar factura y empaque cuando aplique.",
        "",
        "Exclusiones habituales",
        "- Danos por mal uso, caida, liquidos o instalacion incorrecta.",
        "- Desgaste normal por uso.",
        "- Productos personalizados salvo defecto demostrable atribuible al proceso de EcoMarket.",
        "",
        "Como hacer valer la garantia",
        "- Conserva factura o comprobante de compra.",
        "- Contacta a atencion al cliente desde tu cuenta o por los canales oficiales.",
        "- Para electrodomesticos y electronicos puede solicitarse diagnostico o evidencia fotografica.",
        "",
        "Limitacion",
        "- EcoMarket no garantiza disponibilidad de repuestos mas alla del soporte que indique el fabricante.",
    ]
    _write_pdf_lines(lines, out_path)


def main() -> None:
    import pandas as pd

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    pedidos_json = DATA_DIR / "pedidos_ejemplo.json"
    pedidos_xlsx = DATA_DIR / "pedidos_ejemplo.xlsx"
    if pedidos_json.is_file():
        with pedidos_json.open(encoding="utf-8") as f:
            raw = json.load(f)
        rows: list[dict] = []
        for oid, row in raw.items():
            if not isinstance(row, dict):
                continue
            flat = dict(row)
            flat.setdefault("order_id", flat.get("order_id") or oid)
            if isinstance(flat.get("materias"), list):
                flat["materias"] = ", ".join(str(x) for x in flat["materias"])
            rows.append(flat)
        df = pd.DataFrame(rows)
        df.to_excel(pedidos_xlsx, index=False)
        pedidos_json.unlink()
        print(f"Creado {pedidos_xlsx} ({len(df)} filas). Eliminado pedidos_ejemplo.json")
    elif pedidos_xlsx.is_file():
        print(f"Ya existe {pedidos_xlsx}; omito conversion de pedidos.")

    policy_json = DATA_DIR / "politica_devoluciones.json"
    policy_pdf = DATA_DIR / "politica_devoluciones.pdf"
    if policy_json.is_file():
        policy = json.loads(policy_json.read_text(encoding="utf-8"))
        lines = _paragraphs_from_policy(policy if isinstance(policy, dict) else {})
        _write_pdf_lines(lines, policy_pdf)
        policy_json.unlink()
        print(f"Creado {policy_pdf}. Eliminado politica_devoluciones.json")
    elif policy_pdf.is_file():
        print(f"Ya existe {policy_pdf}; omito politica JSON.")

    garantia_pdf = DATA_DIR / "politica_garantia.pdf"
    _write_garantia_pdf(garantia_pdf)
    print(f"Actualizado {garantia_pdf}")


if __name__ == "__main__":
    main()
