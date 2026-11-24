"""Türkçe metin normalizasyonu ve BM25 tokenizer'ları."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Callable

import Stemmer

_WORD_RE = re.compile(r"\w+")
# bm25s'in varsayılan token deseni (v1 BM25 davranışını taklit etmek için)
_BM25S_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")

# v1 değerlendirme kodundaki stopword listesi (exact-match metriği için)
V1_EXACT_STOPWORDS = frozenset(
    {"ve", "bir", "de", "da", "en", "ise", "ki", "ile", "için", "bu", "şu", "o", "ne", "ya", "ama", "fakat"}
)

TURKISH_STOPWORDS = frozenset(
    """
    acaba ama ancak artık aslında az bana bazen bazı bazıları belki ben beni benim beri bile bir birçok biri
    birkaç birkez birşey birşeyi biz bize bizi bizim bu buna bunda bundan bunlar bunları bunların bunu bunun
    burada böyle böylece çok çünkü da daha dahi de defa değil diğer diye dolayı dolayısıyla eğer en fakat gibi
    göre hala halen hangi hatta hem henüz hep hepsi her herhangi herkes hiç hiçbir için ile ilgili ise işte
    itibaren kadar karşın kendi kendine kendisi ki kim kimse mı mi mu mü nasıl ne neden nedenle nerde nerede
    nereye niye niçin o olan olarak oldu olduğu olduğunu olduklarını olmadı olmadığı olmak olması olmayan olmaz
    olsa olsun olup olur olursa oluyor on ona ondan onlar onlardan onları onların onu onun orada öyle oysa
    pek rağmen sadece sanki şey şeyden şeyi şeyler şu şuna şunda şundan şunları şunu tarafından tüm üzere
    ve veya ya yani yapacak yapılan yapılması yapıyor yapmak yaptı yaptığı yaptığını yaptıkları ye yerine yine
    yoksa zaten
    """.split()
)

# bm25s'in "en" stopword listesi (v1 BM25 davranışını taklit etmek için)
ENGLISH_STOPWORDS = frozenset(
    """
    a an and are as at be but by for if in into is it no not of on or such that the their then there these
    they this to was will with
    """.split()
)


def tr_lower(text: str) -> str:
    """Türkçe'ye uygun küçük harf dönüşümü (I→ı, İ→i)."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def v1_exact_tokens(text: str) -> list[str]:
    """v1'deki `tr_normalize` fonksiyonunun birebir karşılığı."""
    words = _WORD_RE.findall(tr_lower(text))
    return [w for w in words if len(w) > 2 and w not in V1_EXACT_STOPWORDS]


@lru_cache(maxsize=None)
def _stemmer(language: str) -> Stemmer.Stemmer:
    return Stemmer.Stemmer(language)


def bm25_tokenizer(name: str) -> Callable[[str], list[str]]:
    """BM25 için tokenizer döndürür.

    - ``v1_en``: v1'in fiilen kullandığı davranış (llama-index BM25Retriever varsayılanı:
      `str.lower`, İngilizce stopword, İngilizce Snowball stemmer).
    - ``tr_plain``: Türkçe küçük harf + Türkçe stopword, kök bulma yok.
    - ``tr_snowball``: ``tr_plain`` + Türkçe Snowball stemmer.
    """
    if name == "v1_en":
        stem = _stemmer("english").stemWords

        def tok(text: str) -> list[str]:
            words = [w for w in _BM25S_TOKEN_RE.findall(text.lower()) if w not in ENGLISH_STOPWORDS]
            return stem(words)

    elif name in ("tr_plain", "tr_snowball"):
        stem = _stemmer("turkish").stemWords if name == "tr_snowball" else None

        def tok(text: str) -> list[str]:
            words = [w for w in _BM25S_TOKEN_RE.findall(tr_lower(text)) if w not in TURKISH_STOPWORDS]
            return stem(words) if stem else words

    else:
        raise ValueError(f"Bilinmeyen BM25 tokenizer: {name}")
    return tok
