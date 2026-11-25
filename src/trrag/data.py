"""Doküman derlemi ve soru-cevap setinin yüklenmesi."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# v1 derleminde birebir aynı içeriğe sahip (git blob hash'leri eşit) dosyalar.
# Anahtar: kopya, değer: kanonik doküman.
V1_DUPLICATE_DOCS = {"FM22": "FM04", "FM21": "FM05", "m44": "m20"}

# "Bu çalışmada…" gibi, derlem içinde tek başına anlam taşımayan soruları yakalar.
_CONTEXT_DEPENDENT_RE = re.compile(
    r"(bu (çalışma|makale|araştırma|bölüm)|çalışmada|makalede|araştırmada|yazar|metinde|söz konusu)",
    re.IGNORECASE,
)

QA_DOMAINS = {
    "matematik-fizik_qa_dataset": "matematik-fizik",
    "ml-ai_qa_dataset": "ml-ai",
    "sosyal-bilimler_qa_dataset": "sosyal-bilimler",
    "sosyal_toplum_bilimleri_qa_dataset": "sosyal-toplum",
    "tip_qa_dataset": "tip",
}


@dataclass
class QAItem:
    qid: str
    question: str
    answer: str
    doc_id: str
    section: str
    domain: str
    flags: dict = field(default_factory=dict)


def doc_id_from_name(name: str) -> str:
    return Path(name.strip()).stem


def load_corpus(docs_dir: str | Path) -> dict[str, str]:
    """Klasördeki tüm .md dosyalarını (alt klasörler dahil) ``{doc_id: metin}`` olarak yükler."""
    docs: dict[str, str] = {}
    for path in sorted(Path(docs_dir).rglob("*.md")):
        doc_id = doc_id_from_name(path.name)
        if doc_id in docs:
            raise ValueError(f"Aynı doküman kimliği iki kez bulundu: {doc_id}")
        docs[doc_id] = path.read_text(encoding="utf-8")
    if not docs:
        raise FileNotFoundError(f"{docs_dir} içinde .md dosyası bulunamadı")
    return docs


def corpus_relpaths(docs_dir: str | Path) -> dict[str, str]:
    """``{doc_id: klasöre göre göreli yol}`` (ör. ``tip_cleaned/t01.md``)."""
    root = Path(docs_dir)
    return {doc_id_from_name(p.name): p.relative_to(root).as_posix() for p in sorted(root.rglob("*.md"))}


def is_context_dependent(question: str) -> bool:
    return bool(_CONTEXT_DEPENDENT_RE.search(question))


def load_qa(qa_dir: str | Path) -> list[QAItem]:
    """QA JSON dosyalarını yükler.

    Kimlikler v1 ile aynı şemada üretilir: ``<dosya_adı>_<sıra>``.
    """
    items: list[QAItem] = []
    files = sorted(Path(qa_dir).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"{qa_dir} içinde .json dosyası bulunamadı")
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        domain = QA_DOMAINS.get(path.stem, path.stem)
        for idx, row in enumerate(data):
            items.append(
                QAItem(
                    qid=row.get("id") or f"{path.name}_{idx}",
                    question=row["question"].strip(),
                    answer=row["ground_truth"].strip(),
                    doc_id=doc_id_from_name(row["source_file"]),
                    section=row.get("source_section", ""),
                    domain=domain,
                )
            )
    return items


def canonical_doc(doc_id: str, aliases: dict[str, str] | None) -> str:
    return aliases.get(doc_id, doc_id) if aliases else doc_id
