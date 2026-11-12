"""Batch-ingest a folder of .txt / .md documents through POST /ingest.

Each file becomes one document. Its document_id comes from a line like
"Document ID: POL-101" near the top of the file, otherwise from the filename
("doc1_handbook.txt" -> "doc1_handbook"). README.md is skipped. Re-running is
safe: re-ingesting a document_id replaces its chunks instead of duplicating them.

Usage (with the API running):
  python ingest_docs.py                                  # data/northwind -> localhost
  python ingest_docs.py data/northwind --base-url https://YOUR-SERVICE.onrender.com
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import httpx

THIS_DIR = Path(__file__).resolve().parent
DOC_ID_LINE = re.compile(r"^Document ID:\s*([A-Za-z0-9._-]+)\s*$", re.MULTILINE)


def document_id_for(path: Path, text: str) -> str:
    match = DOC_ID_LINE.search(text)
    return match.group(1) if match else path.stem


def total_vectors(client: httpx.Client, base_url: str) -> int | None:
    response = client.get(f"{base_url}/health/pinecone")
    return response.json().get("total_vector_count")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", nargs="?", default=str(THIS_DIR / "data" / "northwind"))
    parser.add_argument(
        "--base-url",
        default=os.getenv("API_BASE_URL", "http://127.0.0.1:8000"),
        help="API to ingest into (default: $API_BASE_URL or http://127.0.0.1:8000)",
    )
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    files = sorted(
        path
        for path in Path(args.folder).iterdir()
        if path.suffix.lower() in {".txt", ".md"} and path.name.lower() != "readme.md"
    )
    if not files:
        print(f"No .txt or .md files found in {args.folder}")
        return 1

    print(f"Ingesting {len(files)} file(s) into {base_url}\n")
    failures = 0
    total_chunks = 0
    total_cost = 0.0
    # Render's free tier can take ~a minute to wake up, hence the long timeout.
    with httpx.Client(timeout=120) as client:
        for path in files:
            text = path.read_text(encoding="utf-8")
            doc_id = document_id_for(path, text)
            response = client.post(
                f"{base_url}/ingest",
                json={"text": text, "document_id": doc_id, "source": path.name},
            )
            if response.status_code != 200:
                failures += 1
                print(f"  FAILED {path.name} ({doc_id}): {response.status_code} {response.text}")
                continue
            data = response.json()
            total_chunks += data["chunks_indexed"]
            total_cost += data.get("cost_usd", 0.0)
            print(f"  {doc_id:<10} {path.name:<28} {data['chunks_indexed']:>3} chunks")

        print(f"\nIndexed {total_chunks} chunks from {len(files) - failures} file(s), "
              f"embedding cost ${total_cost:.6f}")

        # Pinecone's stats update a few seconds after an upsert, so poll briefly.
        count = None
        for _ in range(10):
            count = total_vectors(client, base_url)
            if count is not None and count >= total_chunks:
                break
            time.sleep(2)
        print(f"Total chunks in the vector store: {count}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
