import math

import numpy as np
import pytest

from trrag.metrics import NOT_FOUND, ExactMatcher, bootstrap_ci, first_rank, paired_permutation_test, rank_metrics


def test_first_rank():
    assert first_rank([False, True, True]) == 2
    assert first_rank([False, False]) == NOT_FOUND
    assert first_rank([]) == NOT_FOUND


def test_rank_metrics_by_hand():
    ranks = np.array([1, 2, 4, NOT_FOUND])
    m = rank_metrics(ranks, k=3)
    assert m["HitRate"].tolist() == [1, 1, 0, 0]
    assert m["MRR"].tolist() == pytest.approx([1, 0.5, 0, 0])
    assert m["nDCG"].tolist() == pytest.approx([1, 1 / math.log2(3), 0, 0])
    m10 = rank_metrics(ranks, k=10)
    assert m10["MRR"].tolist() == pytest.approx([1, 0.5, 0.25, 0])


def test_exact_token_vs_v1_substring():
    answer = ExactMatcher().answer_tokens("veri madenciliği")
    chunk = "Verilerin madenciliğinde yeni yöntemler"
    # v1: "veri" → "verilerin" içinde alt dize olarak bulunur, "madenciliği" → "madenciliğinde" içinde
    assert ExactMatcher("v1_substring").ratio(answer, chunk) == 1.0
    # token modu: tam kelime eşleşmesi aranır
    assert ExactMatcher("token").ratio(answer, chunk) == 0.0
    assert ExactMatcher("token").matches(answer, "veri madenciliği üzerine")


def test_exact_turkish_casing():
    m = ExactMatcher("token")
    assert m.matches(m.answer_tokens("ISITMA İŞLEMİ"), "ısıtma işlemi uygulandı")


def test_bootstrap_ci_contains_mean():
    vals = np.random.default_rng(0).random(500)
    lo, hi = bootstrap_ci(vals, 500)
    assert lo < vals.mean() < hi


def test_permutation_test():
    a = np.ones(200)
    assert paired_permutation_test(a, a) == 1.0
    assert paired_permutation_test(a, np.zeros(200), n_perm=2000) < 0.01
