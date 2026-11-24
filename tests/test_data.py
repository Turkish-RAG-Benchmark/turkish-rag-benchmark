from pathlib import Path

import pytest

from trrag.data import V1_DUPLICATE_DOCS, is_context_dependent, load_corpus, load_qa

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data/processed/stage2_cleaned"
QA = ROOT / "data/benchmark"

pytestmark = pytest.mark.skipif(not DOCS.exists(), reason="v1 verisi yok")


def test_v1_corpus_and_qa_sizes():
    docs = load_corpus(DOCS)
    qa = load_qa(QA)
    assert len(docs) == 235
    assert len(qa) == 2132
    assert len({q.qid for q in qa}) == len(qa)
    assert {q.doc_id for q in qa} <= set(docs)


def test_v1_duplicates_are_identical():
    docs = load_corpus(DOCS)
    for dup, canon in V1_DUPLICATE_DOCS.items():
        # Temizlenmiş sürümler yalnızca boş satırlarda farklılaşabiliyor
        assert docs[dup].split() == docs[canon].split()


def test_context_dependent_detection():
    assert is_context_dependent("Bu çalışmada hangi yöntem önerilmiştir?")
    assert not is_context_dependent("Psikolojik iyi oluş nedir?")
