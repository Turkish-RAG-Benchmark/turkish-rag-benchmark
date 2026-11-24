import pytest

from trrag.chunking import Chunk
from trrag.retrievers import BM25Retriever, fuse
from trrag.text import bm25_tokenizer, tr_lower


def test_tr_lower():
    assert tr_lower("IŞIK İzmir") == "ışık izmir"


def test_turkish_stemmer_merges_suffixes():
    tok = bm25_tokenizer("tr_snowball")
    assert tok("öğrencilerin")[0] == tok("öğrenci")[0]


def test_bm25_ranks_matching_chunk_first():
    chunks = [Chunk("a#0", "a", "Kalp yetmezliği hastalarında tedavi yöntemleri"),
              Chunk("b#0", "b", "Derin öğrenme modelleri görüntü sınıflandırmada kullanılır"),
              Chunk("c#0", "c", "Osmanlı döneminde eğitim kurumları")]
    r = BM25Retriever(chunks, tokenizer="tr_snowball")
    top = r.retrieve(["derin öğrenme modeli ile sınıflandırma"], k=3)[0]
    assert top[0][0] == 1
    assert len(top) == 3


def test_simple_fusion_is_dominated_by_large_scale():
    dense = [(0, 0.90), (1, 0.85)]          # cosine, 0-1 aralığı
    bm25 = [(2, 14.0), (3, 11.0), (1, 9.0)]  # ham BM25 skorları
    simple = [i for i, _ in fuse([dense, bm25], "simple", k=3)]
    assert simple == [2, 3, 1]  # dense'in en iyi sonucu ilk 3'e bile giremiyor (v1 hatası)
    rrf = [i for i, _ in fuse([dense, bm25], "rrf", k=3)]
    assert rrf[0] == 1  # iki listede de geçen parça en üste çıkar


def test_relative_score_fusion():
    out = fuse([[(0, 1.0), (1, 0.0)], [(1, 10.0), (0, 0.0)]], "relative_score", k=2)
    assert out[0][1] == pytest.approx(1.0) and out[1][1] == pytest.approx(1.0)
    assert [i for i, _ in out] == [0, 1]  # eşitlikte indeks sırası (deterministik)
