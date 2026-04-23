#!/usr/bin/env python3
"""
Quality evaluation for the insights shortlist + scoring pipeline.

Purpose: answer "did we lose quality when we added the embedding shortlist
and bumped the batch size?"

For each query you provide, this script runs two pipelines side by side:

  BASELINE: LLM-score every insight in the corpus, no shortlist (the OLD way).
  NEW:      Embedding shortlist (top-N) → LLM-score only the shortlist.

It then reports:
  - Recall@shortlist: what fraction of BASELINE's relevant insights were in
    the NEW pipeline's shortlist? (The key metric — if this is ~1.0, the
    shortlist isn't dropping good insights.)
  - Final overlap: how much do the two relevant sets agree after scoring?
  - Insights NEW missed and BASELINE missed, with titles.
  - LLM call counts for each pipeline.

Usage:
  python eval_shortlist.py "what do users think about email scheduling?"
  python eval_shortlist.py --queries queries.txt         # one query per line
  python eval_shortlist.py --top-n 200 "query..."        # try a smaller shortlist
  python eval_shortlist.py --top-n 200 --top-n 400 --top-n 800 "query..."
                                                         # sweep multiple N values

Cost note: the BASELINE run costs ~ceil(N_corpus/80) LLM calls per query. For
~2800 insights that's ~35 calls. Budget ~40 calls per query evaluated.
"""

import argparse
import logging
import os
import sys
import time

# Make 'skills.*' importable when running from repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oracle import call_llm
from skills import embeddings as emb_helper
from skills.insights import (
    _insight_id,
    _insight_text_for_embedding,
    batch_score_relevance,
    load_cached_insights,
    set_llm,
    EMBEDDINGS_CACHE_FILE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval")

# Suppress noisy HTTP logs from google.genai
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)


def run_baseline(query, insights):
    """Score every insight via LLM (no shortlist). Returns set of insight ids."""
    t0 = time.time()
    relevant = batch_score_relevance(query, insights)
    elapsed = time.time() - t0
    return {r["id"] for r in relevant}, relevant, elapsed


def run_new(query, insights, top_n):
    """Embedding shortlist → LLM score. Returns shortlist ids, relevant ids, elapsed."""
    t0 = time.time()
    shortlist = emb_helper.shortlist_by_similarity(
        query=query,
        items=insights,
        get_id=_insight_id,
        get_text=_insight_text_for_embedding,
        cache_path=EMBEDDINGS_CACHE_FILE,
        top_n=top_n,
    )
    shortlist_ids = {s["id"] for s in shortlist}
    relevant = batch_score_relevance(query, shortlist)
    elapsed = time.time() - t0
    return shortlist_ids, {r["id"] for r in relevant}, relevant, elapsed


def estimate_calls(n, batch_size=80):
    import math
    return math.ceil(n / batch_size)


def compare_one(query, insights, top_ns, id_to_insight):
    print(f"\n{'=' * 80}")
    print(f"QUERY: {query}")
    print(f"{'=' * 80}")
    print(f"Corpus size: {len(insights)}")

    # --- Baseline (no shortlist) ---
    print(f"\n--- BASELINE: full corpus via LLM ({estimate_calls(len(insights))} calls expected) ---")
    baseline_ids, baseline_items, baseline_elapsed = run_baseline(query, insights)
    print(f"  Relevant: {len(baseline_ids)} insights  |  took {baseline_elapsed:.1f}s")

    rows = []
    for top_n in top_ns:
        print(f"\n--- NEW (top-N={top_n}): embedding shortlist → LLM ({estimate_calls(min(top_n, len(insights)))} calls expected) ---")
        shortlist_ids, new_ids, new_items, new_elapsed = run_new(query, insights, top_n)

        recall_at_shortlist = (
            len(baseline_ids & shortlist_ids) / len(baseline_ids) if baseline_ids else 1.0
        )
        overlap = len(baseline_ids & new_ids) / len(baseline_ids) if baseline_ids else 1.0
        missed_by_new = baseline_ids - new_ids
        only_in_new = new_ids - baseline_ids

        print(f"  Shortlist: {len(shortlist_ids)} items  |  Relevant: {len(new_ids)}  |  took {new_elapsed:.1f}s")
        print(f"  Recall@shortlist: {recall_at_shortlist:.1%} "
              f"({len(baseline_ids & shortlist_ids)}/{len(baseline_ids)} baseline-relevant made it past shortlist)")
        print(f"  Final overlap:    {overlap:.1%} "
              f"({len(baseline_ids & new_ids)}/{len(baseline_ids)} baseline-relevant also flagged by NEW)")

        if missed_by_new:
            print(f"\n  Insights BASELINE found but NEW missed ({len(missed_by_new)}):")
            for iid in list(missed_by_new)[:10]:
                title = id_to_insight.get(iid, {}).get("title", "<untitled>")
                in_shortlist = " (filtered at shortlist)" if iid not in shortlist_ids else ""
                print(f"    - {title[:100]}{in_shortlist}")
            if len(missed_by_new) > 10:
                print(f"    ... and {len(missed_by_new) - 10} more")

        if only_in_new:
            print(f"\n  Insights NEW found but BASELINE missed ({len(only_in_new)}):")
            for iid in list(only_in_new)[:5]:
                title = id_to_insight.get(iid, {}).get("title", "<untitled>")
                print(f"    - {title[:100]}")
            if len(only_in_new) > 5:
                print(f"    ... and {len(only_in_new) - 5} more")

        rows.append({
            "query": query,
            "top_n": top_n,
            "corpus": len(insights),
            "baseline_relevant": len(baseline_ids),
            "new_relevant": len(new_ids),
            "recall_at_shortlist": recall_at_shortlist,
            "final_overlap": overlap,
            "baseline_calls": estimate_calls(len(insights)),
            "new_calls": estimate_calls(min(top_n, len(insights))),
        })

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", nargs="?", help="Single query to evaluate")
    parser.add_argument("--queries", help="Path to file with one query per line")
    parser.add_argument(
        "--top-n", type=int, action="append",
        help="Shortlist size to evaluate (repeat flag to sweep multiple). Default: 400",
    )
    args = parser.parse_args()

    if not args.query and not args.queries:
        parser.print_help()
        sys.exit(1)

    queries = []
    if args.query:
        queries.append(args.query)
    if args.queries:
        with open(args.queries, "r") as f:
            queries.extend(line.strip() for line in f if line.strip())

    top_ns = args.top_n if args.top_n else [400]

    # Wire the LLM into the insights module
    set_llm(call_llm)

    log.info("Loading insights...")
    insights = load_cached_insights()
    id_to_insight = {i["id"]: i for i in insights}

    all_rows = []
    for q in queries:
        rows = compare_one(q, insights, top_ns, id_to_insight)
        all_rows.extend(rows)

    # Summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    header = f"{'top_n':>6} {'base_rel':>9} {'new_rel':>8} {'recall@sl':>11} {'overlap':>9} {'calls_saved':>12}"
    print(header)
    print("-" * len(header))
    for r in all_rows:
        saved = r["baseline_calls"] - r["new_calls"]
        print(
            f"{r['top_n']:>6} {r['baseline_relevant']:>9} {r['new_relevant']:>8} "
            f"{r['recall_at_shortlist']:>10.1%} {r['final_overlap']:>8.1%} "
            f"{saved:>5}/{r['baseline_calls']:<6}"
        )

    # Aggregate recall if multiple queries
    if len(queries) > 1:
        print()
        for top_n in top_ns:
            subset = [r for r in all_rows if r["top_n"] == top_n]
            avg_recall = sum(r["recall_at_shortlist"] for r in subset) / len(subset)
            avg_overlap = sum(r["final_overlap"] for r in subset) / len(subset)
            min_recall = min(r["recall_at_shortlist"] for r in subset)
            print(
                f"top_n={top_n}: avg recall@shortlist={avg_recall:.1%}, "
                f"worst-case={min_recall:.1%}, avg final overlap={avg_overlap:.1%} "
                f"(across {len(subset)} queries)"
            )

    print("\nRule of thumb: recall@shortlist >= 95% and final overlap >= 90% means")
    print("the shortlist size is safe. If recall drops below 90%, raise top_n.")


if __name__ == "__main__":
    main()
