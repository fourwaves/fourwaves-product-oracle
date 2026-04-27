"""
Embedding-based shortlist helper — shared between insights and transcripts skills.

Uses Gemini gemini-embedding-001 to narrow a large corpus down to the top-N most
semantically similar items before running expensive LLM relevance scoring.

Design:
  - Embeddings are cached to disk keyed by item id + content hash.
  - Content change → re-embed on next run. New item → embed on demand.
  - Stale entries (item no longer in corpus) are pruned.
  - Cosine similarity is pure-Python (no numpy) — fast enough for ~10k items.

Persistence:
  The cache file is checked into the repo so GH Actions runs don't re-embed
  the whole corpus every 5 minutes. Only new/changed items produce API calls.
"""

import os
import json
import hashlib
import logging
import math
import random
import time

log = logging.getLogger("oracle.embeddings")

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_OUTPUT_DIM = 768  # match retired text-embedding-004 size; keeps cache files small
EMBEDDING_BATCH_SIZE = 100  # max items per embed_content call
EMBED_ROUND_DECIMALS = 4    # float precision on disk (negligible cosine impact)


def _content_hash(text):
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()[:16]


def _cosine(a, b):
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _embed_texts(texts, task_type="RETRIEVAL_DOCUMENT"):
    """Call Gemini embedding API on a list of texts, batched and retried on 429/503."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    all_vecs = []

    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i:i + EMBEDDING_BATCH_SIZE]
        last_exc = None
        for attempt in range(1, 5):
            try:
                resp = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=EMBEDDING_OUTPUT_DIM,
                    ),
                )
                all_vecs.extend([e.values for e in resp.embeddings])
                break
            except Exception as e:
                last_exc = e
                msg = str(e)
                if ("429" in msg or "503" in msg or "UNAVAILABLE" in msg) and attempt < 4:
                    sleep = (2 ** attempt) + random.uniform(-0.5, 0.5)
                    log.warning(f"  Embedding attempt {attempt}/4 failed ({msg[:80]}), retrying in {sleep:.1f}s")
                    time.sleep(sleep)
                    continue
                raise
        else:
            raise last_exc

    return all_vecs


def _load_cache(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            log.warning(f"Corrupt embeddings cache at {path}, starting fresh.")
    return {}


def _save_cache(path, cache):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, path)


def ensure_embeddings(items, get_id, get_text, cache_path):
    """Ensure every item has a fresh embedding cached. Returns {id: vec}.

    Only calls the embedding API for items that are new or whose text changed
    (detected via content hash). Prunes entries for items no longer in corpus.
    """
    cache = _load_cache(cache_path)

    # Figure out what needs (re-)embedding
    to_embed = []  # list of (id, hash, text)
    for item in items:
        iid = get_id(item)
        text = get_text(item)
        h = _content_hash(text)
        entry = cache.get(iid)
        if not entry or entry.get("hash") != h:
            to_embed.append((iid, h, text))

    if to_embed:
        log.info(f"Embedding {len(to_embed)} new/changed item(s) (of {len(items)} total)...")
        vecs = _embed_texts([t[2] for t in to_embed])
        for (iid, h, _), vec in zip(to_embed, vecs):
            cache[iid] = {
                "hash": h,
                "vec": [round(x, EMBED_ROUND_DECIMALS) for x in vec],
            }
    else:
        log.info(f"Embeddings cache hit: all {len(items)} item(s) up-to-date.")

    # Prune stale entries (items removed from corpus)
    current_ids = {get_id(item) for item in items}
    stale = [iid for iid in cache if iid not in current_ids]
    if stale:
        for iid in stale:
            del cache[iid]
        log.info(f"Pruned {len(stale)} stale embedding entr(ies).")

    if to_embed or stale:
        _save_cache(cache_path, cache)

    return {iid: entry["vec"] for iid, entry in cache.items()}


def shortlist_by_similarity(query, items, get_id, get_text, cache_path, top_n):
    """Return the top-N items most semantically similar to the query.

    If top_n >= len(items), skips similarity ranking and returns everything.
    """
    if top_n >= len(items):
        log.info(f"Shortlist: corpus ({len(items)}) <= top_n ({top_n}), skipping.")
        return list(items)

    id_to_vec = ensure_embeddings(items, get_id, get_text, cache_path)

    # Embed the query with task_type=RETRIEVAL_QUERY for asymmetric retrieval
    query_vec = _embed_texts([query], task_type="RETRIEVAL_QUERY")[0]

    scored = []
    for item in items:
        vec = id_to_vec.get(get_id(item))
        if vec is None:
            continue
        sim = _cosine(query_vec, vec)
        scored.append((sim, item))

    scored.sort(reverse=True, key=lambda x: x[0])
    top = [item for _, item in scored[:top_n]]

    if scored:
        cutoff_idx = min(top_n - 1, len(scored) - 1)
        log.info(
            f"Shortlist: kept top {len(top)}/{len(items)} "
            f"(max sim={scored[0][0]:.3f}, cutoff sim={scored[cutoff_idx][0]:.3f}, "
            f"min sim={scored[-1][0]:.3f})"
        )
    return top
