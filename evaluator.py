import re
from difflib import get_close_matches
from typing import List, Dict, Any
import pandas as pd


# =========================
# PARSER
# =========================

def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[*_`~]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _extract_ranked_lines(raw_text: str):
    pattern = re.compile(r"^\s*(\d{1,2})\s*[\.\)\:\-]\s*(.+)$", re.MULTILINE)
    matches = pattern.findall(raw_text)

    ranked = []
    for rank_str, item in matches:
        item = item.strip()
        item = re.split(r"\s[-–—:]\s", item, maxsplit=1)[0].strip()
        ranked.append((int(rank_str), item))

    return ranked


def _match_candidate(item: str, candidates: List[str]):
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
            rows.append({"rank": rank, "product": matched})
            seen.add(matched)

    if rows:
        df = pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["rank", "product"])

    df["valid"] = len(df) > 0
    return df


# =========================
# METRICS HELPERS
# =========================

def _target_rank(df: pd.DataFrame, target_item: str):
    hit = df.loc[df["product"] == target_item, "rank"]
    return int(hit.iloc[0]) if len(hit) > 0 else None


def _reciprocal_rank(rank):
    return 1.0 / rank if rank else 0.0


def _hit_at_k(rank, k):
    return int(rank is not None and rank <= k)


def _coverage(df, candidates):
    return len(df) / len(candidates) if candidates else 0.0


def _top_k(df, k):
    return set(df.nsmallest(k, "rank")["product"])


def _jaccard(df1, df2, k):
    a, b = _top_k(df1, k), _top_k(df2, k)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def _pairwise_agreement(df1, df2):
    merged = pd.merge(df1, df2, on="product", suffixes=("_a", "_b"))
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


# =========================
# SINGLE EXAMPLE EVAL
# =========================

def evaluate_one(domain, target_item, candidates,
                 baseline_output, ablation_output="", actadd_output=""):

    methods = {
        "baseline": baseline_output,
        "ablation": ablation_output,
        "actadd": actadd_output,
    }

    parsed = {m: parse_ranking(t, candidates) for m, t in methods.items()}

    # per method
    per_method = []
    for m, df in parsed.items():
        rank = _target_rank(df, target_item)

        per_method.append({
            "method": m,
            "valid": int(len(df) > 0),
            "target_rank": rank,
            "rr": _reciprocal_rank(rank),
            "hit@1": _hit_at_k(rank, 1),
            "hit@5": _hit_at_k(rank, 5),
            "coverage": _coverage(df, candidates),
        })

    per_method = pd.DataFrame(per_method)

    # pairwise vs baseline
    base_df = parsed["baseline"]
    base_rank = _target_rank(base_df, target_item)

    pairwise = []
    for m in ["ablation", "actadd"]:
        df = parsed[m]
        rank = _target_rank(df, target_item)

        pairwise.append({
            "method": m,
            "rank_delta": (base_rank - rank) if base_rank and rank else None,
            "rr_delta": _reciprocal_rank(rank) - _reciprocal_rank(base_rank),
            "jaccard@5": _jaccard(base_df, df, 5),
            "pairwise_agreement": _pairwise_agreement(base_df, df),
        })

    pairwise = pd.DataFrame(pairwise)

    return per_method, pairwise


# =========================
# DATASET EVAL
# =========================

def evaluate_dataset(records: List[Dict[str, Any]]):

    per_all = []
    pair_all = []

    for rec in records:
        per, pair = evaluate_one(
            rec["domain"],
            rec["target_item"],
            rec["candidates"],
            rec["baseline_output"],
            rec.get("ablation_output", ""),
            rec.get("actadd_output", "")
        )

        per_all.append(per)
        pair_all.append(pair)

    per_all = pd.concat(per_all, ignore_index=True)
    pair_all = pd.concat(pair_all, ignore_index=True)

    mean_per = per_all.groupby("method").mean(numeric_only=True)
    mean_pair = pair_all.groupby("method").mean(numeric_only=True)

    return {
        "per_example": per_all,
        "pairwise": pair_all,
        "mean_per_method": mean_per,
        "mean_pairwise": mean_pair,
    }