"""Değerlendirme metrikleri.

Her sorunun tek bir ilgili dokümanı ve tek bir referans cevabı olduğundan, bütün metrikler
"ilk doğru sonucun sırası"ndan (rank) türetilebilir:

- HitRate@k = 1[rank <= k]
- MRR@k     = 1/rank   (rank <= k ise, değilse 0)
- nDCG@k    = 1/log2(rank + 1)  (tek ilgili öğe olduğunda IDCG = 1)

v1'deki Recall@k (= HitRate@k) ve MAP@k (= MRR@k) bu kurulumda bilgi taşımadığı için raporlanmaz.
"""

from __future__ import annotations

import math

import numpy as np

from .text import v1_exact_tokens

NOT_FOUND = math.inf


class ExactMatcher:
    """Referans cevabın bir parçada yer alıp almadığını kelime örtüşmesiyle ölçer.

    - ``token``: cevap kelimeleri parçanın kelime kümesinde aranır (düzeltilmiş).
    - ``v1_substring``: v1 davranışı. Kelimeler, parçanın birleştirilmiş metninde alt dize
      olarak aranır ("veri" → "verilerin" içinde eşleşir).
    """

    def __init__(self, mode: str = "token", threshold: float = 0.7):
        if mode not in ("token", "v1_substring"):
            raise ValueError(f"Bilinmeyen exact modu: {mode}")
        self.mode = mode
        self.threshold = threshold
        self._chunk_cache: dict[int, object] = {}

    def answer_tokens(self, answer: str) -> list[str]:
        return v1_exact_tokens(answer)

    def _prepare(self, text: str):
        tokens = v1_exact_tokens(text)
        return set(tokens) if self.mode == "token" else " ".join(tokens)

    def ratio(self, answer_tokens: list[str], text: str, cache_key: int | None = None) -> float:
        if not answer_tokens:
            return 0.0
        if cache_key is not None:
            prepared = self._chunk_cache.get(cache_key)
            if prepared is None:
                prepared = self._chunk_cache[cache_key] = self._prepare(text)
        else:
            prepared = self._prepare(text)
        return sum(1 for w in answer_tokens if w in prepared) / len(answer_tokens)

    def matches(self, answer_tokens: list[str], text: str, cache_key: int | None = None) -> bool:
        return self.ratio(answer_tokens, text, cache_key) >= self.threshold


def first_rank(flags: list[bool]) -> float:
    for i, f in enumerate(flags, start=1):
        if f:
            return float(i)
    return NOT_FOUND


def rank_metrics(ranks: np.ndarray, k: int) -> dict[str, np.ndarray]:
    ranks = np.asarray(ranks, dtype=np.float64)
    hit = ranks <= k
    with np.errstate(divide="ignore"):
        mrr = np.where(hit, 1.0 / ranks, 0.0)
        ndcg = np.where(hit, 1.0 / np.log2(ranks + 1.0), 0.0)
    return {"HitRate": hit.astype(np.float64), "MRR": mrr, "nDCG": ndcg}


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 42) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if n_boot <= 0 or values.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def paired_permutation_test(a: np.ndarray, b: np.ndarray, n_perm: int = 10000, seed: int = 42) -> float:
    """İki sistemin aynı sorular üzerindeki skorları için iki yönlü eşleştirilmiş permütasyon testi."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    observed = abs(diff.mean())
    if diff.size == 0 or np.all(diff == 0):
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, diff.size))
    perm = np.abs((signs * diff).mean(axis=1))
    return float((np.sum(perm >= observed - 1e-12) + 1) / (n_perm + 1))
