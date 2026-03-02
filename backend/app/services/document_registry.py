from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

_REGISTRY_PATH = Path("data/ingestion_registry.json")


def _load_registry() -> List[Dict[str, Any]]:
    if not _REGISTRY_PATH.exists():
        return []
    raw = _REGISTRY_PATH.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    data = json.loads(raw)
    if isinstance(data, list):
        return data
    return []


def _save_registry(items: List[Dict[str, Any]]) -> None:
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REGISTRY_PATH.write_text(json.dumps(items, indent=2), encoding="utf-8")


def list_ingested_documents() -> List[Dict[str, Any]]:
    return _load_registry()


def add_ingested_document(entry: Dict[str, Any]) -> Dict[str, Any]:
    items = _load_registry()
    payload = dict(entry)
    payload["id"] = payload.get("id") or str(uuid4())
    items.append(payload)
    _save_registry(items)
    return payload


def remove_ingested_document(doc_id: str) -> bool:
    items = _load_registry()
    updated = [item for item in items if str(item.get("id")) != str(doc_id)]
    if len(updated) == len(items):
        return False
    _save_registry(updated)
    return True

