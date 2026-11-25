"""Deney çalıştırıcı.

Kullanım::

    python -m trrag.evaluate --config configs/v1_fixed.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .chunking import Chunk, chunk_corpus, chunk_key
from .data import V1_DUPLICATE_DOCS, QAItem, canonical_doc, is_context_dependent, load_corpus, load_qa
from .metrics import NOT_FOUND, ExactMatcher, bootstrap_ci, first_rank, paired_permutation_test, rank_metrics
from .retrievers import BM25Retriever, DenseRetriever, fuse, resolve_local_model

log = logging.getLogger("trrag")

DEFAULT_EVAL = {
    "k_values": [1, 3, 5, 10],
    "exact_mode": "token",
    "exact_threshold": 0.7,
    "merge_duplicate_docs": True,
    "bootstrap": 1000,
    "seed": 42,
}
DEFAULT_RUNTIME = {"device": None, "batch_size": 16, "cache_dir": ".cache", "model_dirs": []}


class Runner:
    def __init__(self, config: dict, config_path: Path, limit: int | None = None):
        self.config = config
        self.root = config_path.parent.parent if config_path.parent.name == "configs" else Path.cwd()
        data_cfg = config["data"]
        self.docs = load_corpus(self._path(data_cfg["docs_dir"]))
        self.qa: list[QAItem] = load_qa(self._path(data_cfg["qa_dir"]))
        if limit:  # hızlı deneme için: her alandan ilk n soru
            n = max(1, limit // len({q.domain for q in self.qa}))
            seen: dict[str, int] = {}
            kept = []
            for q in self.qa:
                seen[q.domain] = seen.get(q.domain, 0) + 1
                if seen[q.domain] <= n:
                    kept.append(q)
            self.qa = kept
        self.eval_defaults = {**DEFAULT_EVAL, **config.get("eval", {})}
        self.runtime = {**DEFAULT_RUNTIME, **config.get("runtime", {})}
        # Makineye özel model klasörleri config'e değil ortam değişkenine yazılır
        env_dirs = os.environ.get("TRRAG_MODEL_DIRS")
        if env_dirs:
            self.runtime["model_dirs"] = list(self.runtime["model_dirs"]) + env_dirs.split(os.pathsep)
        if self.runtime["cache_dir"]:
            self.runtime["cache_dir"] = str(self._path(self.runtime["cache_dir"]))
        self._chunks: dict[str, list[Chunk]] = {}
        self._rankings: dict[str, list] = {}
        self.stats: dict[str, dict] = {}
        missing = sorted({q.doc_id for q in self.qa} - set(self.docs))
        if missing:
            log.warning("QA setinde olup derlemde olmayan dokümanlar: %s", missing)
        for q in self.qa:
            q.flags["context_dependent"] = is_context_dependent(q.question)
        log.info("%d doküman, %d soru yüklendi", len(self.docs), len(self.qa))

    def _path(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else self.root / path

    def chunks(self, spec: dict) -> list[Chunk]:
        key = chunk_key(spec)
        if key not in self._chunks:
            t = time.time()
            self._chunks[key] = chunk_corpus(self.docs, spec)
            words = np.array([len(c.text.split()) for c in self._chunks[key]])
            self.stats[f"chunks:{key}"] = {
                "n_chunks": len(words), "words_mean": round(float(words.mean()), 1),
                "words_median": float(np.median(words)), "words_p95": float(np.percentile(words, 95)),
                "words_max": int(words.max()),
            }
            log.info("Chunking %s: %d parça (%.1fs)", key, len(self._chunks[key]), time.time() - t)
        return self._chunks[key]

    def rankings(self, spec: dict, chunk_spec: dict, k: int) -> list:
        """Bir retriever tanımı için tüm sorguların sıralamalarını döndürür (bellekte önbellekli)."""
        key = json.dumps([spec, chunk_key(chunk_spec), k], sort_keys=True)
        if key in self._rankings:
            return self._rankings[key]
        chunks = self.chunks(chunk_spec)
        queries = [q.question for q in self.qa]
        kind = spec["type"]
        t = time.time()
        if kind == "hybrid":
            cand = int(spec.get("candidate_k", k))
            per_comp = [self.rankings(c, chunk_spec, cand) for c in spec["components"]]
            result = [fuse([pc[i] for pc in per_comp], spec.get("mode", "rrf"), k,
                           int(spec.get("rrf_k", 60)), spec.get("weights"))
                      for i in range(len(queries))]
            stats = {"mode": spec.get("mode", "rrf"), "candidate_k": cand}
        elif kind == "rerank":
            cand = int(spec.get("candidate_k", 50))
            base = self.rankings(spec["base"], chunk_spec, cand)
            result = self._rerank(spec, chunks, queries, base, k)
            stats = {"model": spec["model"], "candidate_k": cand}
        else:
            retriever = self._build(spec, chunks)
            result = retriever.retrieve(queries, k)
            stats = retriever.stats
            retriever.close()
        stats = {**stats, "seconds": round(time.time() - t, 1), "n_chunks": len(chunks)}
        self.stats[key] = stats
        self._rankings[key] = result
        return result

    def _build(self, spec: dict, chunks: list[Chunk]):
        kind = spec["type"]
        if kind == "dense":
            return DenseRetriever(
                chunks, model=spec["model"], prefixes=spec.get("prefixes", "auto"),
                query_prefix=spec.get("query_prefix"), passage_prefix=spec.get("passage_prefix"),
                device=self.runtime["device"], batch_size=int(self.runtime["batch_size"]),
                cache_dir=self.runtime["cache_dir"], max_seq_length=spec.get("max_seq_length"),
                trust_remote_code=bool(spec.get("trust_remote_code", False)),
                dtype=spec.get("dtype", self.runtime.get("dtype")),
                local_path=resolve_local_model(spec["model"], self.runtime["model_dirs"]),
            )
        if kind == "bm25":
            return BM25Retriever(chunks, tokenizer=spec.get("tokenizer", "tr_snowball"),
                                 k1=float(spec.get("k1", 1.5)), b=float(spec.get("b", 0.75)))
        raise ValueError(f"Bilinmeyen retriever türü: {kind}")

    def _rerank(self, spec, chunks, queries, base, k):
        from sentence_transformers import CrossEncoder

        path = resolve_local_model(spec["model"], self.runtime["model_dirs"])
        model = CrossEncoder(str(path or spec["model"]), device=self.runtime["device"],
                             max_length=int(spec.get("max_length", 512)))
        out = []
        for q, ranking in zip(queries, base):
            if not ranking:
                out.append([])
                continue
            scores = model.predict([(q, chunks[i].text) for i, _ in ranking],
                                   batch_size=int(self.runtime["batch_size"]), show_progress_bar=False)
            order = sorted(zip([i for i, _ in ranking], map(float, scores)), key=lambda x: (-x[1], x[0]))
            out.append(order[:k])
        del model
        return out

    def evaluate_system(self, system: dict) -> pd.DataFrame:
        ev = {**self.eval_defaults, **system.get("eval", {})}
        k_values = sorted(int(k) for k in ev["k_values"])
        max_k = k_values[-1]
        chunk_spec = system["chunking"]
        chunks = self.chunks(chunk_spec)
        rankings = self.rankings(system["retriever"], chunk_spec, max_k)
        aliases = V1_DUPLICATE_DOCS if ev["merge_duplicate_docs"] else None
        matcher = ExactMatcher(ev["exact_mode"], float(ev["exact_threshold"]))
        ck = chunk_key(chunk_spec)
        rows = []
        for q, ranking in zip(self.qa, rankings):
            target = canonical_doc(q.doc_id, aliases)
            ans = matcher.answer_tokens(q.answer)
            doc_flags = [canonical_doc(chunks[i].doc_id, aliases) == target for i, _ in ranking]
            exact_flags = [matcher.matches(ans, chunks[i].text, (ck, i)) for i, _ in ranking]
            rows.append({
                "system": system["name"], "qid": q.qid, "domain": q.domain,
                "doc_rank": first_rank(doc_flags), "exact_rank": first_rank(exact_flags),
                "retrieved": len(ranking),
                "context_dependent": q.flags["context_dependent"],
                "answerable": matcher.matches(ans, self.docs.get(q.doc_id, ""), ("doc", q.doc_id)),
            })
        return pd.DataFrame(rows)


def summarize(per_query: pd.DataFrame, k_values: list[int], n_boot: int, seed: int) -> pd.DataFrame:
    subsets = {
        "all": lambda d: d,
        "standalone": lambda d: d[~d.context_dependent],
        "answerable": lambda d: d[d.answerable],
    }
    for dom in sorted(per_query.domain.unique()):
        subsets[f"domain:{dom}"] = lambda d, dom=dom: d[d.domain == dom]
    out = []
    for system, df in per_query.groupby("system", sort=False):
        for subset, fn in subsets.items():
            sub = fn(df)
            for level in ("doc", "exact"):
                for k in k_values:
                    for metric, vals in rank_metrics(sub[f"{level}_rank"].to_numpy(), k).items():
                        lo, hi = bootstrap_ci(vals, n_boot, seed=seed) if subset == "all" else (np.nan, np.nan)
                        out.append({"system": system, "subset": subset, "level": level.capitalize(),
                                    "metric": metric, "k": k, "value": vals.mean() if len(vals) else np.nan,
                                    "ci_low": lo, "ci_high": hi, "n": len(sub)})
    return pd.DataFrame(out)


def wide_table(summary: pd.DataFrame, subset: str = "all") -> pd.DataFrame:
    s = summary[summary.subset == subset].copy()
    s["col"] = s.level + "_" + s.metric + "@" + s.k.astype(str)
    order = list(dict.fromkeys(s.system))
    t = s.pivot(index="system", columns="col", values="value").loc[order]
    cols = sorted(t.columns, key=lambda c: (c.split("_")[0] != "Doc", int(c.split("@")[1]), c))
    return t[cols].round(4)


def compare(per_query: pd.DataFrame, pairs: list, k: int, seed: int) -> pd.DataFrame:
    out = []
    for a, b in pairs:
        da = per_query[per_query.system == a].set_index("qid")
        db = per_query[per_query.system == b].set_index("qid").loc[da.index]
        for level in ("doc", "exact"):
            ma = rank_metrics(da[f"{level}_rank"].to_numpy(), k)["MRR"]
            mb = rank_metrics(db[f"{level}_rank"].to_numpy(), k)["MRR"]
            out.append({"A": a, "B": b, "metric": f"{level.capitalize()}_MRR@{k}",
                        "A_value": ma.mean(), "B_value": mb.mean(), "diff": ma.mean() - mb.mean(),
                        "p_value": paired_permutation_test(ma, mb, seed=seed)})
    return pd.DataFrame(out)


def environment() -> dict:
    env = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ("numpy", "scipy", "llama-index-core", "sentence-transformers", "transformers", "torch", "PyStemmer"):
        try:
            env[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    try:
        import torch

        env["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return env


def run(config_path: str | Path, only: list[str] | None = None, limit: int | None = None) -> Path:
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runner = Runner(config, config_path, limit)
    systems = [s for s in config["systems"] if not only or s["name"] in only]
    out_dir = runner._path(config.get("output_dir", "results/runs")) / (config["name"] + ("-limit" if limit else "")) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, errors = [], {}
    for system in systems:
        log.info("▶ %s", system["name"])
        try:
            frames.append(runner.evaluate_system(system))
        except Exception as e:  # bir sistem hata verirse diğerleri yine çalışsın, ama hata kaydedilsin
            log.exception("%s başarısız oldu", system["name"])
            errors[system["name"]] = f"{type(e).__name__}: {e}"
    if not frames:
        raise RuntimeError(f"Hiçbir sistem çalışmadı: {errors}")

    per_query = pd.concat(frames, ignore_index=True)
    ev = runner.eval_defaults
    k_values = sorted(int(k) for k in ev["k_values"])
    summary = summarize(per_query, k_values, int(ev["bootstrap"]), int(ev["seed"]))

    pq = per_query.copy()
    for col in ("doc_rank", "exact_rank"):
        pq[col] = pq[col].replace(NOT_FOUND, np.nan)
    pq.to_csv(out_dir / "per_query.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    wide = wide_table(summary)
    wide.to_csv(out_dir / "summary_wide.csv")
    if config.get("comparisons"):
        compare(per_query, config["comparisons"], int(config.get("comparison_k", 5)), int(ev["seed"])) \
            .to_csv(out_dir / "comparisons.csv", index=False)

    short_retrievals = per_query[per_query.retrieved < k_values[-1]].groupby("system").size().to_dict()
    (out_dir / "run.json").write_text(json.dumps({
        "config": config, "config_path": str(config_path), "environment": environment(),
        "n_docs": len(runner.docs), "n_questions": len(runner.qa),
        "retriever_stats": runner.stats, "errors": errors,
        "queries_with_fewer_than_max_k_results": short_retrievals,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        cols = [c for c in wide.columns if c.endswith("@5") or c.endswith("@1")]
        print(wide[cols].to_string())
    if errors:
        print("HATALAR:", errors)
    print(f"\nSonuçlar: {out_dir}")
    return out_dir


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Türkçe RAG retrieval benchmark'ını çalıştırır")
    p.add_argument("--config", required=True)
    p.add_argument("--only", nargs="*", help="Sadece bu isimdeki sistemleri çalıştır")
    p.add_argument("--limit", type=int, help="Hızlı deneme için soru sayısını sınırla")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run(args.config, args.only, args.limit)


if __name__ == "__main__":
    main()
