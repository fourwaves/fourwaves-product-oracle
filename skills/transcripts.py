"""
Transcripts Skill — Answer product questions by scanning full call transcripts.

Architecture:
  1. Fetch all transcript metadata from the Notion "Call Transcripts" database
  2. For each transcript, fetch full text from Google Docs (via Drive export API)
  3. Cache fetched transcripts locally (long TTL — transcripts don't change)
  4. Two-pass LLM approach:
     Pass 1: Chunk each transcript and batch-score relevance against the query
     Pass 2: Synthesize response from all relevant chunks

Google Auth: Uses OAuth 2.0 with a saved token.json. Run `python -m skills.transcripts --auth`
to complete the one-time OAuth flow.
"""

import os
import json
import logging
import hashlib
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

log = logging.getLogger("oracle.transcripts")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NOTION_TOKEN = os.environ.get("NOTION_TOKEN", "")
TRANSCRIPTS_DB = "2bc8b055-517b-80d1-ad02-e10d408dcc19"

NOTION_HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28",
}

# Google OAuth
CLIENT_SECRET_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "google_client_secret.json",
)
# Fallback location
CLIENT_SECRET_DOWNLOADS = os.path.expanduser(
    "~/Downloads/client_secret_125911849982-glq9o4r64j74q4t3b8qjtcmn50hbcrvg.apps.googleusercontent.com.json"
)
TOKEN_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "google_token.json")
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Cache
TRANSCRIPT_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "transcript_cache")
METADATA_CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "oracle_transcripts_meta_cache.json")
METADATA_CACHE_TTL_MINUTES = 360  # 6 hours for the Notion metadata
# Individual transcript texts are cached indefinitely (they don't change)

# LLM batching
CHUNK_SIZE_CHARS = 8000  # Split transcripts into chunks of this size
RELEVANCE_BATCH_SIZE = 15  # Number of chunks per LLM scoring batch
MAX_PARALLEL_BATCHES = 5

# Will be set by the router
_call_llm = None


def _llm(system_prompt, user_prompt, model_hint="flash"):
    if _call_llm is None:
        raise RuntimeError("LLM not initialized — call set_llm() first")
    return _call_llm(system_prompt, user_prompt, model_hint)


def set_llm(fn):
    global _call_llm
    _call_llm = fn


# ---------------------------------------------------------------------------
# Google Auth
# ---------------------------------------------------------------------------

def get_google_credentials():
    """Load or refresh Google OAuth credentials.

    Supports two modes:
      1. GOOGLE_TOKEN_JSON env var (for CI / GitHub Actions) — JSON string
      2. Local token.json file (for local development)
    """
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    creds = None
    token_json_env = os.environ.get("GOOGLE_TOKEN_JSON", "")

    if token_json_env:
        # Load from environment variable (CI mode)
        creds = Credentials.from_authorized_user_info(json.loads(token_json_env), SCOPES)
    elif os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        # Persist refreshed token back to file if running locally
        if not token_json_env and os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE, "w") as f:
                f.write(creds.to_json())
        return creds

    raise RuntimeError(
        "Google OAuth token not found or expired. "
        "Run: cd fourwaves-product-oracle && python -m skills.transcripts --auth"
    )


def run_auth_flow():
    """One-time interactive OAuth flow."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    secret_file = CLIENT_SECRET_FILE if os.path.exists(CLIENT_SECRET_FILE) else CLIENT_SECRET_DOWNLOADS
    if not os.path.exists(secret_file):
        raise FileNotFoundError(f"Client secret not found at {secret_file}")

    flow = InstalledAppFlow.from_client_secrets_file(secret_file, SCOPES)
    creds = flow.run_local_server(port=0)
    with open(TOKEN_FILE, "w") as f:
        f.write(creds.to_json())
    log.info(f"Token saved to {TOKEN_FILE}")
    return creds


# ---------------------------------------------------------------------------
# Google Docs: fetch transcript text
# ---------------------------------------------------------------------------

def fetch_doc_text(doc_id, creds):
    """Export a Google Doc as plain text via the Drive API."""
    from googleapiclient.discovery import build

    service = build("drive", "v3", credentials=creds)
    response = service.files().export(fileId=doc_id, mimeType="text/plain").execute()
    if isinstance(response, bytes):
        return response.decode("utf-8")
    return str(response)


def get_cached_transcript(doc_id):
    """Return cached transcript text or None."""
    os.makedirs(TRANSCRIPT_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(TRANSCRIPT_CACHE_DIR, f"{doc_id}.txt")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return f.read()
    return None


def save_cached_transcript(doc_id, text):
    """Cache transcript text to disk."""
    os.makedirs(TRANSCRIPT_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(TRANSCRIPT_CACHE_DIR, f"{doc_id}.txt")
    with open(cache_path, "w") as f:
        f.write(text)


# ---------------------------------------------------------------------------
# Notion: fetch transcript metadata
# ---------------------------------------------------------------------------

def extract_text(prop):
    if not prop:
        return ""
    prop_type = prop.get("type", "")
    if prop_type == "title":
        return " ".join(t.get("plain_text", "") for t in prop.get("title", []))
    elif prop_type == "rich_text":
        return " ".join(t.get("plain_text", "") for t in prop.get("rich_text", []))
    elif prop_type == "select":
        sel = prop.get("select")
        return sel.get("name", "") if sel else ""
    elif prop_type == "multi_select":
        return ", ".join(o.get("name", "") for o in prop.get("multi_select", []))
    elif prop_type == "email":
        return prop.get("email", "") or ""
    elif prop_type == "date":
        d = prop.get("date")
        return d.get("start", "") if d else ""
    return ""


def fetch_transcript_metadata():
    """Fetch all transcript entries from Notion."""
    log.info("Fetching transcript metadata from Notion...")
    url = f"https://api.notion.com/v1/databases/{TRANSCRIPTS_DB}/query"

    all_results = []
    has_more = True
    start_cursor = None
    while has_more:
        body = {"page_size": 100}
        if start_cursor:
            body["start_cursor"] = start_cursor
        resp = requests.post(url, headers=NOTION_HEADERS, json=body)
        resp.raise_for_status()
        data = resp.json()
        all_results.extend(data["results"])
        has_more = data.get("has_more", False)
        start_cursor = data.get("next_cursor")

    log.info(f"Fetched {len(all_results)} transcript entries from Notion.")

    transcripts = []
    for page in all_results:
        props = page.get("properties", {})
        doc_id = extract_text(props.get("Document ID", {})).strip()
        if not doc_id:
            continue

        entry = {
            "page_id": page["id"],
            "name": extract_text(props.get("Name", {})),
            "summary": extract_text(props.get("Summary", {})),
            "short_description": extract_text(props.get("Short description", {})),
            "call_date": extract_text(props.get("Call Date", {})),
            "call_type": extract_text(props.get("Call type", {})),
            "participants": extract_text(props.get("Fourwaves participants", {})),
            "doc_id": doc_id,
            "transcript_link": extract_text(props.get("Transcript Link", {})),
        }
        transcripts.append(entry)

    log.info(f"Found {len(transcripts)} transcripts with Document IDs.")
    return transcripts


def load_cached_metadata():
    """Load transcript metadata from cache if fresh, else fetch from Notion."""
    if os.path.exists(METADATA_CACHE_FILE):
        with open(METADATA_CACHE_FILE, "r") as f:
            cache = json.load(f)
        cached_at = datetime.fromisoformat(cache.get("cached_at", "2000-01-01"))
        age_minutes = (datetime.now() - cached_at).total_seconds() / 60
        if age_minutes < METADATA_CACHE_TTL_MINUTES:
            entries = cache.get("transcripts", [])
            log.info(f"Using cached metadata ({len(entries)} entries, {age_minutes:.0f}min old).")
            return entries
    return refresh_metadata_cache()


def refresh_metadata_cache():
    entries = fetch_transcript_metadata()
    cache = {
        "cached_at": datetime.now().isoformat(),
        "count": len(entries),
        "transcripts": entries,
    }
    with open(METADATA_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)
    log.info(f"Metadata cache refreshed: {len(entries)} entries.")
    return entries


# ---------------------------------------------------------------------------
# Fetch full transcripts (with caching)
# ---------------------------------------------------------------------------

def load_all_transcripts(metadata_entries):
    """Fetch full transcript text for all entries, using disk cache."""
    creds = None  # Lazy-init
    results = []
    fetch_count = 0

    for entry in metadata_entries:
        doc_id = entry["doc_id"]
        cached = get_cached_transcript(doc_id)
        if cached:
            results.append({**entry, "full_text": cached})
            continue

        # Need to fetch from Google
        if creds is None:
            creds = get_google_credentials()

        try:
            text = fetch_doc_text(doc_id, creds)
            save_cached_transcript(doc_id, text)
            results.append({**entry, "full_text": text})
            fetch_count += 1
            log.info(f"  Fetched transcript: {entry['name'][:60]} ({len(text)} chars)")
        except Exception as e:
            log.warning(f"  Failed to fetch {doc_id} ({entry['name'][:40]}): {e}")
            # Still include the entry with summary-only for partial coverage
            results.append({**entry, "full_text": entry.get("summary", "")})

    log.info(f"Loaded {len(results)} transcripts ({fetch_count} freshly fetched, {len(results) - fetch_count} from cache).")
    return results


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_transcript(transcript_entry):
    """Split a transcript into overlapping chunks, each tagged with metadata."""
    text = transcript_entry["full_text"]
    if not text:
        return []

    meta_header = (
        f"Call: {transcript_entry['name']}\n"
        f"Date: {transcript_entry['call_date']}\n"
        f"Type: {transcript_entry['call_type']}\n"
        f"Participants: {transcript_entry['participants']}\n"
    )

    chunks = []
    # Split on paragraphs first for natural boundaries
    paragraphs = text.split("\n\n")
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 > CHUNK_SIZE_CHARS and current_chunk:
            chunks.append({
                "text": meta_header + current_chunk.strip(),
                "call_name": transcript_entry["name"],
                "call_date": transcript_entry["call_date"],
                "call_type": transcript_entry["call_type"],
                "participants": transcript_entry["participants"],
                "doc_id": transcript_entry["doc_id"],
            })
            # Keep last 500 chars as overlap for context continuity
            current_chunk = current_chunk[-500:] + "\n\n" + para
        else:
            current_chunk += "\n\n" + para

    if current_chunk.strip():
        chunks.append({
            "text": meta_header + current_chunk.strip(),
            "call_name": transcript_entry["name"],
            "call_date": transcript_entry["call_date"],
            "call_type": transcript_entry["call_type"],
            "participants": transcript_entry["participants"],
            "doc_id": transcript_entry["doc_id"],
        })

    return chunks


# ---------------------------------------------------------------------------
# Pass 1: Batch relevance scoring on chunks
# ---------------------------------------------------------------------------

def score_chunk_batch(query, batch_chunks, batch_indices):
    """Score a batch of transcript chunks for relevance."""
    system_prompt = """You are a relevance scorer for call transcript chunks. Given a search query and a batch of transcript excerpts, determine which chunks contain information relevant to the query.

A chunk is RELEVANT if:
- It contains user feedback, comments, complaints, or suggestions related to the query topic
- It mentions features, workflows, or pain points connected to the query
- A user or prospect discusses something related to the query
- It contains decisions, discussions, or context about the query topic

Be INCLUSIVE — when in doubt, mark as relevant. It's better to include a borderline chunk than to miss real feedback. The synthesis step will handle prioritization.

Return ONLY a JSON array of the index numbers that are relevant. Example: [0, 3, 7]
If none are relevant, return: []"""

    chunk_texts = []
    for i, chunk in enumerate(batch_chunks):
        # Truncate chunk text for scoring to keep batches manageable
        preview = chunk["text"][:2000]
        chunk_texts.append(f"[{i}] ({chunk['call_name']} — {chunk['call_date']})\n{preview}")

    chunks_block = "\n---\n".join(chunk_texts)
    user_prompt = f"""QUERY: {query}

TRANSCRIPT CHUNKS TO EVALUATE:
{chunks_block}

Return ONLY the JSON array of relevant index numbers (0-based within this batch):"""

    raw = _llm(system_prompt, user_prompt, model_hint="flash")

    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        indices = json.loads(cleaned)
        if not isinstance(indices, list):
            return []
        return [batch_indices[i] for i in indices if 0 <= i < len(batch_chunks)]
    except (json.JSONDecodeError, IndexError, TypeError):
        log.warning(f"Failed to parse relevance response: {raw[:200]}")
        return []


def batch_score_chunks(query, all_chunks):
    """Score all chunks for relevance using parallel batch LLM calls."""
    n = len(all_chunks)
    if n == 0:
        return []

    indices = list(range(n))

    batches = []
    for i in range(0, n, RELEVANCE_BATCH_SIZE):
        batch_indices = indices[i:i + RELEVANCE_BATCH_SIZE]
        batch_chunks = [all_chunks[j] for j in batch_indices]
        batches.append((batch_chunks, batch_indices))

    log.info(f"Scoring {n} chunks in {len(batches)} batches...")

    relevant_indices = set()
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_BATCHES) as executor:
        futures = {}
        for batch_num, (batch, batch_indices) in enumerate(batches):
            future = executor.submit(score_chunk_batch, query, batch, batch_indices)
            futures[future] = batch_num

        for future in as_completed(futures):
            batch_num = futures[future]
            try:
                result = future.result()
                relevant_indices.update(result)
                log.info(f"  Batch {batch_num + 1}/{len(batches)}: {len(result)} relevant")
            except Exception as e:
                log.error(f"  Batch {batch_num + 1} failed: {e}")

    relevant = [all_chunks[i] for i in sorted(relevant_indices)]
    log.info(f"Total relevant chunks: {len(relevant)} out of {n} scanned.")
    return relevant


# ---------------------------------------------------------------------------
# Pass 2: Synthesis
# ---------------------------------------------------------------------------

def synthesize_transcript_response(query, relevant_chunks, total_chunks, total_transcripts):
    """Synthesize a comprehensive response from relevant transcript chunks."""

    # Group chunks by call for context
    calls_seen = {}
    for chunk in relevant_chunks:
        key = chunk["doc_id"]
        if key not in calls_seen:
            calls_seen[key] = {
                "call_name": chunk["call_name"],
                "call_date": chunk["call_date"],
                "call_type": chunk["call_type"],
                "participants": chunk["participants"],
                "chunks": [],
            }
        calls_seen[key]["chunks"].append(chunk["text"])

    # Build knowledge base organized by call
    kb_parts = []
    for doc_id, call_info in calls_seen.items():
        header = (
            f"=== CALL: {call_info['call_name']} ===\n"
            f"Date: {call_info['call_date']} | Type: {call_info['call_type']} | "
            f"Participants: {call_info['participants']}\n"
        )
        # Deduplicate overlapping chunk content
        combined = "\n\n---\n\n".join(call_info["chunks"])
        kb_parts.append(header + combined)

    knowledge_base = ("\n\n" + "=" * 60 + "\n\n").join(kb_parts)

    # If knowledge base is very large, we may need to do a two-stage synthesis
    if len(knowledge_base) > 200000:
        return _large_synthesis(query, calls_seen, total_chunks, total_transcripts)

    system_prompt = """You are the Product Oracle, an expert product analyst for Fourwaves (an event management platform for conferences and academic events).

You have been given relevant excerpts from call transcripts (sales calls, support calls, demos, onboarding, feedback sessions, etc.). Your job is to produce a comprehensive, evidence-packed response to the user's question.

# RULES

1. *EXHAUSTIVE*: Reference ALL relevant information from the transcript excerpts. Do not skip any call that contains relevant content.
2. *EVIDENCE-BASED*: Quote users verbatim when possible. Use blockquotes (>) for every direct quote from a call participant.
3. *TRACEABLE*: Always cite the call name and date when referencing information (e.g., "In the call with [Name] on [Date]...").
4. *STRUCTURED*: Group findings by theme, category, or sub-topic as appropriate. Use clear section headers.
5. *QUANTIFIED*: Count how many calls/users mentioned each theme.
6. *DISTINGUISH SPEAKERS*: Differentiate between what Fourwaves team members said vs. what external users/prospects said.
7. *ACTIONABLE*: End with a "Key Takeaways" section summarizing the main patterns.

# OUTPUT FORMAT (STRICT SLACK MRKDWN)
- Bold: use single * on each side (*bold*). NEVER use ** double asterisks.
- Blockquotes: use > at the start of a line for user quotes.
- Headers: use *BOLD CAPS* with single asterisks.
- Use double line breaks between sections.
- NEVER use # characters or ** double asterisks.
- NEVER use markdown links [text](url).

# RESPONSE STRUCTURE
Start with a one-line summary of what you found (e.g., "Found relevant discussions across 8 calls from 6 users about OpenConf migration.").
Then organize the evidence thematically.
End with *KEY TAKEAWAYS* section."""

    user_prompt = f"""*QUERY*: {query}

*STATS*: {len(relevant_chunks)} relevant excerpts found across {len(calls_seen)} calls (out of {total_transcripts} total calls scanned, {total_chunks} total chunks).

*RELEVANT TRANSCRIPT EXCERPTS*:

{knowledge_base}

Generate a comprehensive response addressing the query using ALL the relevant excerpts above."""

    return _llm(system_prompt, user_prompt, model_hint="pro")


def _large_synthesis(query, calls_seen, total_chunks, total_transcripts):
    """Handle very large result sets by summarizing per-call first, then synthesizing."""
    log.info("Large result set — doing per-call summaries first...")

    # Stage 1: Summarize each call's relevant content
    call_summaries = []

    def summarize_call(doc_id, call_info):
        combined = "\n\n---\n\n".join(call_info["chunks"])
        prompt = f"""Summarize the following transcript excerpts in relation to this query: {query}

Call: {call_info['call_name']}
Date: {call_info['call_date']}
Type: {call_info['call_type']}
Participants: {call_info['participants']}

Transcript excerpts:
{combined[:50000]}

Provide a detailed summary preserving all relevant quotes (with > blockquotes), speaker attributions, and specific feedback. Be thorough — don't skip anything relevant."""

        summary = _llm("You are a transcript analyst. Summarize call content relevant to a query.", prompt, model_hint="flash")
        return {
            "call_name": call_info["call_name"],
            "call_date": call_info["call_date"],
            "call_type": call_info["call_type"],
            "participants": call_info["participants"],
            "summary": summary,
        }

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_BATCHES) as executor:
        futures = {
            executor.submit(summarize_call, doc_id, info): doc_id
            for doc_id, info in calls_seen.items()
        }
        for future in as_completed(futures):
            try:
                call_summaries.append(future.result())
            except Exception as e:
                log.error(f"Per-call summary failed: {e}")

    # Stage 2: Final synthesis from summaries
    summaries_text = ("\n\n" + "=" * 40 + "\n\n").join(
        f"=== {cs['call_name']} ({cs['call_date']}, {cs['call_type']}) ===\n"
        f"Participants: {cs['participants']}\n\n{cs['summary']}"
        for cs in call_summaries
    )

    system_prompt = """You are the Product Oracle, an expert product analyst for Fourwaves.
Synthesize the per-call summaries into a single comprehensive response.

# OUTPUT FORMAT (STRICT SLACK MRKDWN)
- Bold: single * (*bold*). NEVER use ** or #.
- Blockquotes: > for user quotes.
- Group by theme/category. Count mentions per theme.
- End with *KEY TAKEAWAYS*.
- Start with a one-line stat summary."""

    user_prompt = f"""*QUERY*: {query}

*STATS*: Evidence from {len(call_summaries)} calls (out of {total_transcripts} total).

*PER-CALL SUMMARIES*:

{summaries_text}

Synthesize into a comprehensive response."""

    return _llm(system_prompt, user_prompt, model_hint="pro")


# ---------------------------------------------------------------------------
# Skill entry points
# ---------------------------------------------------------------------------

def handle_transcript_query(message_text, call_llm_fn):
    """Handle a question by scanning all call transcripts."""
    set_llm(call_llm_fn)

    # Step 1: Get transcript metadata
    metadata = load_cached_metadata()
    log.info(f"Found {len(metadata)} transcript entries. Loading full texts...")

    # Step 2: Load full transcript texts (cached)
    transcripts = load_all_transcripts(metadata)

    # Step 3: Chunk all transcripts
    all_chunks = []
    for t in transcripts:
        chunks = chunk_transcript(t)
        all_chunks.extend(chunks)

    log.info(f"Total chunks across {len(transcripts)} transcripts: {len(all_chunks)}")

    if not all_chunks:
        return "No call transcripts found in the database, or none could be loaded."

    # Step 4: Score relevance
    relevant = batch_score_chunks(message_text, all_chunks)

    if not relevant:
        return (
            f"I scanned all {len(transcripts)} call transcripts "
            f"({len(all_chunks)} chunks) but couldn't find any content matching your query. "
            f"Try rephrasing or broadening your search?"
        )

    # Step 5: Synthesize response
    response = synthesize_transcript_response(
        message_text, relevant, len(all_chunks), len(transcripts)
    )
    return response


def handle_transcript_followup(thread_context, followup_text, call_llm_fn):
    """Handle a follow-up in a transcript thread."""
    set_llm(call_llm_fn)

    # Classify: answer from context or new scan?
    classification = _llm(
        """You are evaluating a follow-up message in a Slack thread about call transcript analysis.

The thread already contains an analysis of call transcripts. The user is now asking a follow-up.

Classify:
- "context" — Can be answered from the thread (e.g., "tell me more about X", "summarize", "draft an email")
- "new_scan" — Needs a fresh scan (e.g., "now look for feedback about Y", "what about Z?")

Reply ONLY with "context" or "new_scan".""",
        f"Thread:\n{thread_context}\n\nFollow-up:\n{followup_text}",
        model_hint="flash",
    ).strip().lower()

    if classification.startswith("new_scan"):
        return handle_transcript_query(followup_text, call_llm_fn)

    return _llm(
        """You are the Product Oracle for Fourwaves. You are in a Slack thread where you already provided call transcript analysis. Answer the follow-up using information from the thread.

# OUTPUT FORMAT (STRICT SLACK MRKDWN)
- Bold: single * (*bold*). NEVER use ** or #.
- Blockquotes: > for quotes. Be concise.""",
        f"Thread:\n{thread_context}\n\nFollow-up:\n{followup_text}",
        model_hint="pro",
    )


# ---------------------------------------------------------------------------
# CLI: auth setup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--auth", action="store_true", help="Run OAuth flow to generate token.json")
    args = parser.parse_args()

    if args.auth:
        creds = run_auth_flow()
        print(f"Authentication successful! Token saved to {TOKEN_FILE}")
        print("You can now use the transcript skill.")
    else:
        parser.print_help()
