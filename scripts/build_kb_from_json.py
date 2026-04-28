from __future__ import annotations

from pathlib import Path

import pandas as pd
from pypdf import PdfReader

try:
    import tomllib  # py311+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
KB_DIR = ROOT / "kb"


def _load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _read_pdf_text(path: Path) -> str:
    if not path.is_file():
        return ""
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n\n".join(parts).strip()


def build_orders_markdown_from_xlsx(path: Path) -> str:
    if not path.is_file():
        return "# Pedidos EcoMarket\n\n_(no se encontro pedidos_ejemplo.xlsx)_\n"
    df = pd.read_excel(path, sheet_name=0)
    lines = ["# Pedidos EcoMarket", ""]
    for _, row in df.iterrows():
        oid = row.get("order_id") or row.get("pedido_id")
        if oid is None or (isinstance(oid, float) and pd.isna(oid)):
            oid = ""
        title = f"## Pedido {oid}" if str(oid).strip() else "## Pedido"
        lines.append(title)
        for k, v in row.items():
            if pd.isna(v):
                continue
            lines.append(f"- {k}: {v}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_policy_markdown_from_pdf(path: Path, title: str) -> str:
    body = _read_pdf_text(path)
    if not body:
        return f"# {title}\n\n_(PDF vacio o no encontrado: {path.name})_\n"
    return f"# {title}\n\n{body}\n"


def build_toml_kb_markdown(cfg: dict) -> str:
    prompts = cfg.get("prompts") or {}
    lines = [
        "# Base adicional desde settings-final.toml",
        "",
        "## Pedidos (documento textual)",
        str(prompts.get("orders_database_document", "")).strip(),
        "",
        "## Politica de devoluciones (documento textual)",
        str(prompts.get("return_policy_document", "")).strip(),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    KB_DIR.mkdir(parents=True, exist_ok=True)

    pedidos_xlsx = DATA_DIR / "pedidos_ejemplo.xlsx"
    (KB_DIR / "pedidos.md").write_text(
        build_orders_markdown_from_xlsx(pedidos_xlsx),
        encoding="utf-8",
    )

    policy_pdf = DATA_DIR / "politica_devoluciones.pdf"
    (KB_DIR / "politica_devoluciones.md").write_text(
        build_policy_markdown_from_pdf(policy_pdf, "Politica de devoluciones EcoMarket"),
        encoding="utf-8",
    )

    garantia_pdf = DATA_DIR / "politica_garantia.pdf"
    (KB_DIR / "politica_garantia.md").write_text(
        build_policy_markdown_from_pdf(garantia_pdf, "Politica de garantia EcoMarket"),
        encoding="utf-8",
    )

    settings_path = DATA_DIR / "settings-final.toml"
    if settings_path.is_file():
        cfg = _load_toml(settings_path)
        (KB_DIR / "kb_desde_toml.md").write_text(
            build_toml_kb_markdown(cfg),
            encoding="utf-8",
        )

    print(f"Documentos KB generados en: {KB_DIR}")


if __name__ == "__main__":
    main()
