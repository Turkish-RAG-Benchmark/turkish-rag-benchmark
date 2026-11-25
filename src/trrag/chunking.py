"""Dokümanların parçalara (chunk) bölünmesi.

v1 ile karşılaştırılabilirlik için llama-index'in bölücüleri kullanılır.
"""

from __future__ import annotations

from dataclasses import dataclass

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import MetadataMode


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    # Retriever'ların indekslediği metin. Varsayılan olarak ``text`` ile aynıdır;
    # v1 uyumluluk modunda başına "file_path: ..." meta verisi eklenir.
    index_text: str | None = None

    @property
    def retrieval_text(self) -> str:
        return self.index_text if self.index_text is not None else self.text


def _hf_tokenizer(name: str):
    """Chunk boyutunu bir embedding modelinin kendi token'larıyla ölçmek için."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    return lambda text: tok.encode(text, add_special_tokens=False)


def make_splitter(spec: dict):
    kind = spec.get("type", "sentence")
    if kind == "sentence":
        # Varsayılan: llama-index'in tiktoken (cl100k) sayımı, v1 ile aynı.
        # "tokenizer" verilirse boyut o modelin token'larıyla ölçülür; böylece parçalar
        # modelin max_seq_length sınırını aşmaz (kesilme/truncation olmaz).
        tokenizer = _hf_tokenizer(spec["tokenizer"]) if spec.get("tokenizer") else None
        return SentenceSplitter(chunk_size=int(spec["size"]), chunk_overlap=int(spec["overlap"]),
                                tokenizer=tokenizer)
    if kind == "markdown":
        return MarkdownNodeParser()
    raise ValueError(f"Bilinmeyen chunking türü: {kind}")


def chunk_key(spec: dict) -> str:
    kind = spec.get("type", "sentence")
    if kind == "sentence":
        key = f"sentence-{spec['size']}-{spec['overlap']}"
        if spec.get("tokenizer"):
            key += f"-{spec['tokenizer'].replace('/', '_')}"
    else:
        key = kind
    return key + "-v1meta" if spec.get("v1_file_path_prefix") else key


def chunk_corpus(docs: dict[str, str], spec: dict, doc_relpaths: dict[str, str] | None = None) -> list[Chunk]:
    """Derlemi parçalara böler. Boş parçalar atılır.

    Varsayılan olarak sadece doküman metni kullanılır.

    ``v1_file_path_prefix`` verilirse v1 davranışı taklit edilir: v1'de SimpleDirectoryReader
    her dokümana ``file_path`` meta verisi ekliyordu. Bu meta veri parça boyutu hesabından yer
    yiyor, embedding ve BM25 metninin başına "file_path: <yol>" olarak ekleniyordu
    (exact metriği ise sadece parça metnine bakıyordu).
    """
    splitter = make_splitter(spec)
    prefix = spec.get("v1_file_path_prefix")
    documents = []
    for doc_id, text in docs.items():
        if prefix:
            rel = (doc_relpaths or {}).get(doc_id, f"{doc_id}.md")
            doc = Document(text=text, id_=doc_id, metadata={"file_path": f"{prefix.rstrip('/')}/{rel}"})
        else:
            doc = Document(text=text, id_=doc_id)
        documents.append(doc)
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
        index_text = node.get_content(metadata_mode=MetadataMode.EMBED).strip() if prefix else None
        chunks.append(Chunk(chunk_id=f"{doc_id}#{n}", doc_id=doc_id, text=text, index_text=index_text))
    return chunks
