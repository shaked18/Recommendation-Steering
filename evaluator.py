import re
from difflib import get_close_matches
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd


# ============================================================
# INPUT FORMAT
# ============================================================
# You can call:
#   results = evaluate_dataset(records, experiment=1)
#
# or put "experiment" inside each record and call:
#   results = evaluate_dataset(records)
#
# -----------------------------
# Experiment 1: single target
# -----------------------------
# records = [
#     {
#         "experiment": 1,              # optional if passed globally to evaluate_dataset
#         "domain": "smart phones",
#         "target_item": "iPhone 14 Pro",
#         "candidates": [...],
#         "baseline_output": "...",
#         "ablation_output": "...",
#         "actadd_output": "..."
#     }
# ]
#
# Metrics:
#   valid, target_rank, rr, hit@1, hit@5, coverage
#   rank_delta, rr_delta, jaccard@5, pairwise_agreement
#
# --------------------------------
# Experiment 2: cross-domain leakage
# --------------------------------
# records = [
#     {
#         "experiment": 2,
#         "domain": "smart phones",
#         "candidates": [...],
#         "baseline_output": "...",
#         "ablation_output": "...",
#         "actadd_output": "..."
#     }
# ]
#
# Metrics:
#   valid
#   jaccard@5, pairwise_agreement
#
# --------------------------------
# Experiment 3: coefficient evaluation
# --------------------------------
# One record per coefficient.
# Repeat baseline if needed.
#
# records = [
#     {
#         "experiment": 3,
#         "domain": "smart phones",
#         "target_item": "iPhone 14 Pro",
#         "coeff": -0.3,
#         "candidates": [...],
#         "baseline_output": "...",
#         "ablation_output": "...",
#         "actadd_output": "..."
#     },
#     {
#         "experiment": 3,
#         "domain": "smart phones",
#         "target_item": "iPhone 14 Pro",
#         "coeff": -0.5,
#         "candidates": [...],
#         "baseline_output": "...",
#         "ablation_output": "...",
#         "actadd_output": "..."
#     }
# ]
#
# Metrics:
#   same as Exp 1
#   plus grouped mean tables by coefficient
#
# --------------------------------
# Experiment 4: double direction
# --------------------------------
# records = [
#     {
#         "experiment": 4,
#         "domain": "smart phones",
#         "target_item_1": "iPhone 14 Pro",
#         "target_item_2": "Samsung Galaxy S23 Ultra",
#         "candidates": [...],
#         "baseline_output": "...",
#         "ablation_output": "...",
#         "actadd_output": "..."
#     }
# ]
#
# Metrics:
#   aggregated over the 2 targets:
#       target_rank, rr, hit@1, hit@5
#   plus per-target columns:
#       target_rank_1, target_rank_2, rr_1, rr_2, ...
#   pairwise deltas also aggregated + per-target columns


# ============================================================
# PARSER
# ============================================================

def extract_scores(raw_text: str, candidates: List[str]) -> Dict[str, float]:
    scores = {}
    text = raw_text or ""

    for cand in candidates:
        pattern = rf"{re.escape(cand)}.*?\(?\s*score[:\s]*([0-9]+\.?[0-9]*)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            scores[cand] = float(match.group(1))

    return scores


def scores_to_ranking(scores: Dict[str, float]) -> pd.DataFrame:
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    rows = [{"rank": i, "product": product} for i, (product, _) in enumerate(sorted_items, start=1)]
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame({
            "rank": pd.Series(dtype="int64"),
            "product": pd.Series(dtype="string"),
        })
    df["valid"] = len(df) > 0
    return df


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[*_`~]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_ranked_lines(raw_text: str) -> List[Tuple[int, str]]:
    pattern = re.compile(r"^\s*(\d{1,2})\s*[\.\)\:\-]\s*(.+)$", re.MULTILINE)
    matches = pattern.findall(raw_text or "")

    ranked = []
    for rank_str, item in matches:
        item = item.strip()
        item = re.split(r"\s[-–—:]\s", item, maxsplit=1)[0].strip()
        ranked.append((int(rank_str), item))

    return ranked


def _match_candidate(item: str, candidates: List[str]) -> Optional[str]:
    item_norm = _normalize(item)
    candidate_map = {_normalize(c): c for c in candidates}

    if item_norm in candidate_map:
        return candidate_map[item_norm]

    for norm_cand, orig_cand in candidate_map.items():
        if norm_cand in item_norm or item_norm in norm_cand:
            return orig_cand

    close = get_close_matches(item_norm, candidate_map.keys(), n=1, cutoff=0.6)
    if close:
        return candidate_map[close[0]]

    return None


def parse_ranking(raw_text: str, candidates: List[str]) -> pd.DataFrame:
    extracted = _extract_ranked_lines(raw_text)

    seen = set()
    rows = []

    for rank, item in extracted:
        matched = _match_candidate(item, candidates)
        if matched is not None and matched not in seen:
            rows.append({"rank": int(rank), "product": matched})
            seen.add(matched)

    if rows:
        df = pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)
    else:
        df = pd.DataFrame({
            "rank": pd.Series(dtype="int64"),
            "product": pd.Series(dtype="string"),
        })

    df["valid"] = len(df) > 0
    return df


def parse_ranking_or_scores(raw_text: str, candidates: List[str], min_scored_items: int = 3) -> pd.DataFrame:
    scores = extract_scores(raw_text, candidates)
    if len(scores) >= min_scored_items:
        return scores_to_ranking(scores)
    return parse_ranking(raw_text, candidates)


# ============================================================
# METRICS HELPERS
# ============================================================

def _target_rank(df: pd.DataFrame, target_item: str) -> Optional[int]:
    hit = df.loc[df["product"] == target_item, "rank"]
    return int(hit.iloc[0]) if len(hit) > 0 else None


def _reciprocal_rank(rank: Optional[int]) -> float:
    return 1.0 / rank if rank is not None and rank > 0 else 0.0


def _hit_at_k(rank: Optional[int], k: int) -> int:
    return int(rank is not None and rank <= k)


def _coverage(df: pd.DataFrame, candidates: List[str]) -> float:
    return len(df) / len(candidates) if candidates else 0.0


def _top_k(df: pd.DataFrame, k: int) -> set:
    if df.empty or "rank" not in df.columns:
        return set()

    rank_series = pd.to_numeric(df["rank"], errors="coerce")
    tmp = df.copy()
    tmp["rank"] = rank_series
    tmp = tmp.dropna(subset=["rank"])

    if tmp.empty:
        return set()

    return set(tmp.nsmallest(k, "rank")["product"])


def _jaccard(df1: pd.DataFrame, df2: pd.DataFrame, k: int) -> float:
    a, b = _top_k(df1, k), _top_k(df2, k)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def _pairwise_agreement(df1: pd.DataFrame, df2: pd.DataFrame) -> Optional[float]:
    merged = pd.merge(df1[["product", "rank"]], df2[["product", "rank"]], on="product", suffixes=("_a", "_b"))
    if len(merged) < 2:
        return None

    total, agree = 0, 0
    rows = merged.to_dict("records")

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            total += 1
            a = rows[i]["rank_a"] < rows[j]["rank_a"]
            b = rows[i]["rank_b"] < rows[j]["rank_b"]
            if a == b:
                agree += 1

    return agree / total if total else None


def _mean_ignore_none(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _get_experiment(record: Dict[str, Any], experiment: Optional[int]) -> int:
    exp = experiment if experiment is not None else record.get("experiment")
    if exp not in {1, 2, 3, 4}:
        raise ValueError("Experiment must be one of {1, 2, 3, 4}.")
    return int(exp)


def _get_targets(record: Dict[str, Any], exp: int) -> List[str]:
    if exp in {1, 3}:
        if "target_item" not in record:
            raise ValueError(f"Experiment {exp} requires 'target_item'.")
        return [record["target_item"]]

    if exp == 4:
        t1 = record.get("target_item_1")
        t2 = record.get("target_item_2")
        if t1 is None or t2 is None:
            raise ValueError("Experiment 4 requires 'target_item_1' and 'target_item_2'.")
        return [t1, t2]

    return []


def _base_metadata(record: Dict[str, Any], exp: int) -> Dict[str, Any]:
    meta = {
        "experiment": exp,
        "domain": record.get("domain"),
    }
    if "coeff" in record:
        meta["coeff"] = record["coeff"]
    if "target_item" in record:
        meta["target_item"] = record["target_item"]
    if "target_item_1" in record:
        meta["target_item_1"] = record["target_item_1"]
    if "target_item_2" in record:
        meta["target_item_2"] = record["target_item_2"]
    return meta


# ============================================================
# SINGLE EXAMPLE EVALUATION
# ============================================================

def evaluate_one(record: Dict[str, Any], experiment: Optional[int] = None):
    exp = _get_experiment(record, experiment)
    candidates = record["candidates"]

    methods = {
        "baseline": record["baseline_output"],
        "ablation": record.get("ablation_output", ""),
        "actadd": record.get("actadd_output", ""),
    }

    parsed = {m: parse_ranking_or_scores(t, candidates) for m, t in methods.items()}
    meta = _base_metadata(record, exp)

    if exp == 2:
        per_method_rows = []
        for m, df in parsed.items():
            row = {
                **meta,
                "method": m,
                "valid": int(len(df) > 0),
            }
            per_method_rows.append(row)

        per_method = pd.DataFrame(per_method_rows)

        base_df = parsed["baseline"]
        pairwise_rows = []
        for m in ["ablation", "actadd"]:
            df = parsed[m]
            pairwise_rows.append({
                **meta,
                "method": m,
                "jaccard@5": _jaccard(base_df, df, 5),
                "pairwise_agreement": _pairwise_agreement(base_df, df),
            })

        pairwise = pd.DataFrame(pairwise_rows)
        return per_method, pairwise

    target_items = _get_targets(record, exp)

    per_method_rows = []
    for m, df in parsed.items():
        target_ranks = [_target_rank(df, t) for t in target_items]
        target_rrs = [_reciprocal_rank(r) for r in target_ranks]
        target_hits1 = [_hit_at_k(r, 1) for r in target_ranks]
        target_hits5 = [_hit_at_k(r, 5) for r in target_ranks]

        row = {
            **meta,
            "method": m,
            "valid": int(len(df) > 0),
            "coverage": _coverage(df, candidates),
            "num_targets": len(target_items),
            "target_rank": _mean_ignore_none(target_ranks),
            "rr": _mean_ignore_none(target_rrs),
            "hit@1": _mean_ignore_none(target_hits1),
            "hit@5": _mean_ignore_none(target_hits5),
        }

        for idx, t in enumerate(target_items, start=1):
            row[f"target_item_{idx}"] = t
            row[f"target_rank_{idx}"] = target_ranks[idx - 1]
            row[f"rr_{idx}"] = target_rrs[idx - 1]
            row[f"hit@1_{idx}"] = target_hits1[idx - 1]
            row[f"hit@5_{idx}"] = target_hits5[idx - 1]

        per_method_rows.append(row)

    per_method = pd.DataFrame(per_method_rows)

    base_df = parsed["baseline"]
    base_ranks = [_target_rank(base_df, t) for t in target_items]
    base_rrs = [_reciprocal_rank(r) for r in base_ranks]

    pairwise_rows = []
    for m in ["ablation", "actadd"]:
        df = parsed[m]
        ranks = [_target_rank(df, t) for t in target_items]
        rrs = [_reciprocal_rank(r) for r in ranks]

        rank_deltas = [
            (br - r) if (br is not None and r is not None) else None
            for br, r in zip(base_ranks, ranks)
        ]
        rr_deltas = [
            rr - brr for rr, brr in zip(rrs, base_rrs)
        ]

        row = {
            **meta,
            "method": m,
            "num_targets": len(target_items),
            "rank_delta": _mean_ignore_none(rank_deltas),
            "rr_delta": _mean_ignore_none(rr_deltas),
            "jaccard@5": _jaccard(base_df, df, 5),
            "pairwise_agreement": _pairwise_agreement(base_df, df),
        }

        for idx, t in enumerate(target_items, start=1):
            row[f"target_item_{idx}"] = t
            row[f"rank_delta_{idx}"] = rank_deltas[idx - 1]
            row[f"rr_delta_{idx}"] = rr_deltas[idx - 1]

        pairwise_rows.append(row)

    pairwise = pd.DataFrame(pairwise_rows)
    return per_method, pairwise


# ============================================================
# DATASET EVALUATION
# ============================================================

def evaluate_dataset(records: List[Dict[str, Any]], experiment: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    if not records:
        raise ValueError("records must be a non-empty list")

    per_all = []
    pair_all = []

    inferred_exp = _get_experiment(records[0], experiment)
    for rec in records:
        rec_exp = _get_experiment(rec, experiment)
        if rec_exp != inferred_exp:
            raise ValueError("All records in one evaluate_dataset call must belong to the same experiment.")

        per, pair = evaluate_one(rec, experiment=inferred_exp)
        per_all.append(per)
        pair_all.append(pair)

    per_all = pd.concat(per_all, ignore_index=True) if per_all else pd.DataFrame()
    pair_all = pd.concat(pair_all, ignore_index=True) if pair_all else pd.DataFrame()

    group_cols = ["method"]
    if "coeff" in per_all.columns:
        group_cols = ["coeff", "method"]

    mean_per_method = (
        per_all.groupby(group_cols, dropna=False).mean(numeric_only=True).reset_index()
        if not per_all.empty else pd.DataFrame()
    )

    mean_pairwise = (
        pair_all.groupby(group_cols, dropna=False).mean(numeric_only=True).reset_index()
        if not pair_all.empty else pd.DataFrame()
    )

    results = {
        "per_example": per_all,
        "pairwise": pair_all,
        "mean_per_method": mean_per_method,
        "mean_pairwise": mean_pairwise,
    }

    if inferred_exp == 3 and "coeff" in per_all.columns:
        results["mean_per_method_by_coeff"] = (
            per_all.groupby(["coeff", "method"], dropna=False).mean(numeric_only=True).reset_index()
        )
        results["mean_pairwise_by_coeff"] = (
            pair_all.groupby(["coeff", "method"], dropna=False).mean(numeric_only=True).reset_index()
        )

    return results
