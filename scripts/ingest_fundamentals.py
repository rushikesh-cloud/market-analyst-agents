from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.document_ingestion import ingest_pdf_to_pgvector


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest a company PDF (annual report/10-K) into pgvector using Azure Document Intelligence."
    )
    parser.add_argument("--pdf", type=Path, help="Path to the PDF to ingest.")
    parser.add_argument("--pdf-dir", type=Path, help="Directory containing PDFs to ingest.")
    parser.add_argument("--company", required=True, help="Company name for metadata filter.")
    parser.add_argument("--doc-type", default="annual_report", help="Document type metadata.")
    parser.add_argument("--year", default=None, help="Document year metadata (e.g., 2023).")
    parser.add_argument("--collection", default="fundamental_docs", help="pgvector collection name.")
    parser.add_argument("--out-markdown", type=Path, default=None, help="Optional markdown output path.")
    parser.add_argument("--azure-model", default="prebuilt-layout", help="Azure Document Intelligence model ID.")
    parser.add_argument(
        "--embeddings-deployment",
        default=None,
        help="Azure OpenAI embeddings deployment name (defaults to env AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT).",
    )
    parser.add_argument(
        "--connection-string",
        default=None,
        help="pgvector connection string (defaults to env PGVECTOR_CONNECTION_STRING).",
    )
    return parser.parse_args()


def _iter_pdfs(pdf: Optional[Path], pdf_dir: Optional[Path]) -> List[Path]:
    if pdf and pdf_dir:
        raise ValueError("Provide only one of --pdf or --pdf-dir.")
    if pdf:
        return [pdf]
    if pdf_dir:
        return sorted([path for path in pdf_dir.glob("*.pdf") if path.is_file()])
    raise ValueError("You must provide --pdf or --pdf-dir.")


def _resolve_markdown_path(out_markdown: Optional[Path], pdf_path: Path) -> Optional[Path]:
    if not out_markdown:
        return None
    if out_markdown.is_dir() or out_markdown.suffix == "":
        return out_markdown / f"{pdf_path.stem}.md"
    return out_markdown


def main() -> int:
    load_dotenv()
    args = _parse_args()

    try:
        pdfs = _iter_pdfs(args.pdf, args.pdf_dir)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 2

    if not pdfs:
        print("No PDFs found to ingest.")
        return 0

    for pdf_path in pdfs:
        if not pdf_path.exists():
            print(f"Skipping missing file: {pdf_path}")
            continue

        markdown_path = _resolve_markdown_path(args.out_markdown, pdf_path)
        result = ingest_pdf_to_pgvector(
            pdf_path=pdf_path,
            company=args.company,
            doc_type=args.doc_type,
            year=args.year,
            collection_name=args.collection,
            connection_string=args.connection_string,
            markdown_output_path=markdown_path,
            azure_model_id=args.azure_model,
            embeddings_deployment=args.embeddings_deployment,
        )
        print(
            f"Ingested {pdf_path.name} -> {result.chunks_stored} chunks "
            f"into collection '{result.collection_name}'."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
