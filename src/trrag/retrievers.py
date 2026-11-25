"""Retriever'lar: yoğun (dense), BM25 ve hibrit füzyon.

Tüm retriever'lar aynı arayüzü uygular::

    retrieve(queries: list[str], k: int) -> list[list[tuple[int, float]]]

Dönen her liste, ilgili sorgu için skora göre azalan sırada ``(chunk_index, skor)`` çiftleridir.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

from .chunking import Chunk
from .text import bm25_tokenizer

log = logging.getLogger(__name__)

Ranking = list[tuple[int, float]]

# Talimat/önek bekleyen modeller. "auto" modunda kullanılır.
# Modelin sentence-transformers yapılandırmasında "query" prompt'u varsa o tercih edilir.
KNOWN_PREFIXES: dict[str, tuple[str, str]] = {
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
}


def resolve_local_model(model: str, model_dirs: list[str] | None) -> Path | None:
    """``model_dirs`` içinde ``<org>__<isim>`` klasörü varsa onu döndürür (Hub'a gitmeden yükleme)."""
    for d in model_dirs or []:
        p = Path(d) / model.replace("/", "__")
        if (p / "config.json").exists():
            return p
    return None


def _top_k(scores: np.ndarray, k: int) -> Ranking:
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, k - 1)[:k]
    # Eşit skorlarda sonucu deterministik yapmak için önce skor, sonra indeks
    idx = idx[np.lexsort((idx, -scores[idx]))]
    return [(int(i), float(scores[i])) for i in idx]


class DenseRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        model: str,
        prefixes: str = "auto",
        query_prefix: str | None = None,
        passage_prefix: str | None = None,
        device: str | None = None,
        batch_size: int = 16,
        cache_dir: str | Path | None = None,
        max_seq_length: int | None = None,
        trust_remote_code: bool = False,
        dtype: str | None = None,
        local_path: str | Path | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        model_kwargs = {"dtype": dtype} if dtype else None
        self.model = SentenceTransformer(str(local_path or model), device=device,
                                         trust_remote_code=trust_remote_code, model_kwargs=model_kwargs)
        if max_seq_length:
            self.model.max_seq_length = int(max_seq_length)
        self.batch_size = batch_size
        self.q_kwargs, self.p_kwargs = self._resolve_prompts(prefixes, query_prefix, passage_prefix)
        self.stats = {"model": model, "max_seq_length": self.model.max_seq_length,
                      "query_prompt": self.q_kwargs, "passage_prompt": self.p_kwargs}
        self.stats["truncated_chunks"] = self._count_truncated(chunks)
        self.matrix = self._embed_corpus(chunks, cache_dir)

    def _resolve_prompts(self, mode, query_prefix, passage_prefix):
        if mode == "none":
            return {}, {}
        if mode == "custom":
            return ({"prompt": query_prefix} if query_prefix else {},
                    {"prompt": passage_prefix} if passage_prefix else {})
        if mode != "auto":
            raise ValueError(f"Bilinmeyen prefixes modu: {mode}")
        prompts = getattr(self.model, "prompts", {}) or {}
        if prompts.get("query"):
            p_name = next((n for n in ("document", "passage") if prompts.get(n)), None)
            return {"prompt_name": "query"}, ({"prompt_name": p_name} if p_name else {})
        if self.model_name in KNOWN_PREFIXES:
            q, p = KNOWN_PREFIXES[self.model_name]
            return {"prompt": q}, {"prompt": p}
        return {}, {}

    def _count_truncated(self, chunks: list[Chunk]) -> int:
        tok = self.model.tokenizer
        limit = self.model.max_seq_length
        n = 0
        for i in range(0, len(chunks), 256):
            lengths = tok([c.retrieval_text for c in chunks[i:i + 256]], add_special_tokens=True,
                          truncation=False)["input_ids"]
            n += sum(len(x) > limit for x in lengths)
        return n

    def _encode(self, texts: list[str], kwargs: dict, show_progress: bool) -> np.ndarray:
        emb = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=show_progress, **kwargs)
        return emb.astype(np.float32)

    def _embed_corpus(self, chunks: list[Chunk], cache_dir) -> np.ndarray:
        cache_file = None
        if cache_dir:
            h = hashlib.sha1()
            h.update(f"{self.model_name}|{self.p_kwargs}|{self.model.max_seq_length}".encode())
            for c in chunks:
                h.update(c.retrieval_text.encode("utf-8"))
                h.update(b"\0")
            cache_file = Path(cache_dir) / "embeddings" / f"{h.hexdigest()}.npy"
            if cache_file.exists():
                log.info("Embedding cache kullanılıyor: %s", cache_file.name)
                return np.load(cache_file).astype(np.float32)
        emb = self._encode([c.retrieval_text for c in chunks], self.p_kwargs, show_progress=True)
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, emb.astype(np.float16))
        return emb

    def retrieve(self, queries: list[str], k: int) -> list[Ranking]:
        q = self._encode(queries, self.q_kwargs, show_progress=False)
        scores = q @ self.matrix.T
        return [_top_k(row, k) for row in scores]

    def close(self):
        del self.model
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


class BM25Retriever:
    """Okapi BM25 (Lucene varyantı; bm25s ile aynı formül, k1=1.5, b=0.75)."""

    def __init__(self, chunks: list[Chunk], tokenizer: str = "tr_snowball", k1: float = 1.5, b: float = 0.75):
        self.tokenize = bm25_tokenizer(tokenizer)
        self.stats = {"tokenizer": tokenizer, "k1": k1, "b": b}
        docs = [Counter(self.tokenize(c.retrieval_text)) for c in chunks]
        vocab: dict[str, int] = {}
        rows, cols, tfs = [], [], []
        for j, counts in enumerate(docs):
            for term, tf in counts.items():
                rows.append(vocab.setdefault(term, len(vocab)))
                cols.append(j)
                tfs.append(tf)
        self.vocab = vocab
        n_docs = len(chunks)
        dl = np.array([sum(c.values()) for c in docs], dtype=np.float64)
        avgdl = dl.mean() if n_docs else 0.0
        rows_a, cols_a, tf_a = np.array(rows), np.array(cols), np.array(tfs, dtype=np.float64)
        df = np.bincount(rows_a, minlength=len(vocab))
        idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        weights = idf[rows_a] * tf_a * (k1 + 1) / (tf_a + k1 * (1 - b + b * dl[cols_a] / avgdl))
        self.matrix = sparse.csr_matrix((weights, (rows_a, cols_a)), shape=(len(vocab), n_docs))
        self.stats["vocab_size"] = len(vocab)

    def retrieve(self, queries: list[str], k: int) -> list[Ranking]:
        out = []
        for q in queries:
            ids = sorted({self.vocab[t] for t in self.tokenize(q) if t in self.vocab})
            if ids:
                scores = np.asarray(self.matrix[ids].sum(axis=0)).ravel()
            else:
                scores = np.zeros(self.matrix.shape[1])
            out.append(_top_k(scores, k))
        return out

    def close(self):
        pass


def fuse(rankings: list[Ranking], mode: str, k: int, rrf_k: int = 60,
         weights: list[float] | None = None) -> Ranking:
    """Birden çok sıralamayı birleştirir.

    - ``simple``: llama-index ``QueryFusionRetriever`` varsayılanı. Her parça için ham
      skorların maksimumu alınır. Skor ölçekleri farklıysa (BM25 ~10-30, cosine 0-1)
      büyük ölçekli retriever sıralamayı tek başına belirler. v1 bu modu kullanmıştı.
    - ``rrf``: Reciprocal Rank Fusion, ``sum 1 / (rrf_k + rank)``.
    - ``relative_score``: her sıralama kendi içinde min-max ölçeklenir, ağırlıklı toplanır.
    """
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = {}
    if mode == "simple":
        for ranking in rankings:
            for idx, score in ranking:
                fused[idx] = max(fused.get(idx, -math.inf), score)
    elif mode == "rrf":
        for w, ranking in zip(weights, rankings):
            for rank, (idx, _) in enumerate(ranking, start=1):
                fused[idx] = fused.get(idx, 0.0) + w / (rrf_k + rank)
    elif mode == "relative_score":
        for w, ranking in zip(weights, rankings):
            if not ranking:
                continue
            scores = [s for _, s in ranking]
            lo, hi = min(scores), max(scores)
            for idx, s in ranking:
                norm = (s - lo) / (hi - lo) if hi > lo else 1.0
                fused[idx] = fused.get(idx, 0.0) + w * norm
    else:
        raise ValueError(f"Bilinmeyen füzyon modu: {mode}")
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:k]


class HybridRetriever:
    def __init__(self, components: list, mode: str = "rrf", candidate_k: int | None = None,
                 rrf_k: int = 60, weights: list[float] | None = None):
        self.components = components
        self.mode = mode
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.weights = weights
        self.stats = {"mode": mode, "rrf_k": rrf_k, "candidate_k": candidate_k, "weights": weights,
                      "components": [getattr(c, "stats", {}) for c in components]}

    def retrieve(self, queries: list[str], k: int) -> list[Ranking]:
        n = self.candidate_k or k
        per_comp = [c.retrieve(queries, n) for c in self.components]
        return [fuse([pc[i] for pc in per_comp], self.mode, k, self.rrf_k, self.weights)
                for i in range(len(queries))]

    def close(self):
        for c in self.components:
            c.close()
