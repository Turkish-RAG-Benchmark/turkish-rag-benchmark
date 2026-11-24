"""Dokümanların parçalara (chunk) bölünmesi.

v1 ile karşılaştırılabilirlik için llama-index'in bölücüleri kullanılır.
"""

from __future__ import annotations

from dataclasses import dataclass

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


def make_splitter(spec: dict):
    kind = spec.get("type", "sentence")
    if kind == "sentence":
        return SentenceSplitter(chunk_size=int(spec["size"]), chunk_overlap=int(spec["overlap"]))
    if kind == "markdown":
        return MarkdownNodeParser()
    raise ValueError(f"Bilinmeyen chunking türü: {kind}")


def chunk_key(spec: dict) -> str:
    kind = spec.get("type", "sentence")
    if kind == "sentence":
        return f"sentence-{spec['size']}-{spec['overlap']}"
    return kind


def chunk_corpus(docs: dict[str, str], spec: dict) -> list[Chunk]:
    """Derlemi parçalara böler. Boş parçalar atılır.

    Not: v1'de SimpleDirectoryReader dosya yolu/adı gibi meta verileri parça boyutu
    hesabına ve embedding metnine dahil ediyordu. Burada sadece doküman metni kullanılır.
    """
    splitter = make_splitter(spec)
    documents = [Document(text=text, id_=doc_id) for doc_id, text in docs.items()]
    nodes = splitter.get_nodes_from_documents(documents)
    chunks: list[Chunk] = []
    counters: dict[str, int] = {}
    for node in nodes:
        text = node.get_content().strip()
        if not text:
            continue
        doc_id = node.ref_doc_id
        n = counters.get(doc_id, 0)
        counters[doc_id] = n + 1
        chunks.append(Chunk(chunk_id=f"{doc_id}#{n}", doc_id=doc_id, text=text))
    return chunks
