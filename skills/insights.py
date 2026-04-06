"""
Insights Skill — Answer product questions using Notion User Insights.

Two-pass LLM approach:
  Pass 1: Batch relevance scoring — every insight evaluated in small batches
  Pass 2: Synthesis — only relevant insights sent for response generation
"""

import os
import json
import random
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

log = logging.getLogger("oracle.insights")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NOTION_TOKEN = os.environ["NOTION_TOKEN"]
INSIGHTS_DB = "2b98b055-517b-8067-95b2-c15baabe75ca"

NOTION_HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28",
}

VALID_SOURCES = {"Intercom", "Survey", "Offboarding", "Email", "Call Transcript"}

CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "oracle_insights_cache.json")
CACHE_TTL_MINUTES = 1440

RELEVANCE_BATCH_SIZE = 30
MAX_PARALLEL_BATCHES = 5

# Will be set by the router when calling skill functions
_call_llm = None


def _llm(system_prompt, user_prompt, model_hint="flash"):
    if _call_llm is None:
        raise RuntimeError("LLM not initialized — call set_llm() first")
    return _call_llm(system_prompt, user_prompt, model_hint)


def set_llm(fn):
    global _call_llm
    _call_llm = fn


# ---------------------------------------------------------------------------
# Notion: fetch all valid insights
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
    elif prop_type == "email":
        return prop.get("email", "") or ""
    elif prop_type == "date":
        d = prop.get("date")
        return d.get("start", "") if d else ""
    elif prop_type == "number":
        return prop.get("number", "")
    return ""


def fetch_insights_from_notion():
    log.info("Fetching insights from Notion...")
    url = f"https://api.notion.com/v1/databases/{INSIGHTS_DB}/query"

    filter_obj = {
        "and": [
            {"property": "Not an insight", "select": {"does_not_equal": "TRUE"}},
        ]
    }

    all_results = []
    has_more = True
    start_cursor = None
    while has_more:
        body = {"page_size": 100, "filter": filter_obj}
        if start_cursor:
            body["start_cursor"] = start_cursor
        resp = requests.post(url, headers=NOTION_HEADERS, json=body)
        resp.raise_for_status()
        data = resp.json()
        all_results.extend(data["results"])
        has_more = data.get("has_more", False)
        start_cursor = data.get("next_cursor")

    log.info(f"Fetched {len(all_results)} insights from Notion.")

    insights = []
    for page in all_results:
        props = page.get("properties", {})
        source = extract_text(props.get("Source", {}))
        if source not in VALID_SOURCES:
            continue

        insight = {
            "id": page["id"],
            "title": extract_text(props.get("Title", {})),
            "short_description": extract_text(props.get("short_description", {})),
            "long_description": extract_text(props.get("long_description", {})),
            "user_name": extract_text(props.get("User name", {})),
            "user_email": extract_text(props.get("User email", {})),
            "user_role": extract_text(props.get("User role", {})),
            "source": source,
            "date": extract_text(props.get("Date", {})),
            "processed_notes": extract_text(props.get("Processed Notes", {})),
            "follow_up_feedback": extract_text(props.get("Follow-up feedback", {})),
        }
        if not insight["title"] and not insight["short_description"]:
            continue
        insights.append(insight)

    log.info(f"After source filter: {len(insights)} valid insights.")
    return insights


def load_cached_insights():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
        cached_at = datetime.fromisoformat(cache.get("cached_at", "2000-01-01"))
        age_minutes = (datetime.now() - cached_at).total_seconds() / 60
        if age_minutes < CACHE_TTL_MINUTES:
            insights = cache.get("insights", [])
            log.info(f"Using cached insights ({len(insights)}, {age_minutes:.0f}min old).")
            return insights
    return refresh_cache()


def refresh_cache():
    insights = fetch_insights_from_notion()
    cache = {
        "cached_at": datetime.now().isoformat(),
        "count": len(insights),
        "insights": insights,
    }
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)
    log.info(f"Cache refreshed: {len(insights)} insights.")
    return insights


# ---------------------------------------------------------------------------
# Pass 1: Batch relevance scoring
# ---------------------------------------------------------------------------

def format_insight_for_scoring(insight, index):
    parts = [f"[{index}] {insight['title']}"]
    if insight["short_description"]:
        parts.append(f"  Summary: {insight['short_description'][:200]}")
    if insight["long_description"]:
        parts.append(f"  Details: {insight['long_description'][:400]}")
    if insight["user_role"]:
        parts.append(f"  Role: {insight['user_role']}")
    if insight["follow_up_feedback"]:
        parts.append(f"  Follow-up feedback: {insight['follow_up_feedback'][:200]}")
    return "\n".join(parts)


def score_batch(query, batch, batch_indices):
    system_prompt = """You are a relevance scorer. Given a search query and a batch of user insights,
determine which insights are relevant to the query.

An insight is RELEVANT if:
- It directly addresses the topic in the query
- It contains feedback, requests, complaints, or suggestions related to the query topic
- It mentions features, workflows, or pain points connected to the query
- The user's words or context relate to what is being asked about

Be INCLUSIVE — when in doubt, mark as relevant. It's better to include a borderline insight
than to miss one. The synthesis step will handle prioritization.

Return ONLY a JSON array of the index numbers that are relevant. Example: [0, 3, 7, 12]
If none are relevant, return: []"""

    insight_texts = []
    for i, (idx, insight) in enumerate(zip(batch_indices, batch)):
        insight_texts.append(format_insight_for_scoring(insight, i))

    user_prompt = f"""QUERY: {query}

INSIGHTS TO EVALUATE:
{chr(10).join(insight_texts)}

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
        return [batch_indices[i] for i in indices if 0 <= i < len(batch)]
    except (json.JSONDecodeError, IndexError, TypeError):
        log.warning(f"Failed to parse relevance response: {raw[:200]}")
        return []


def batch_score_relevance(query, insights):
    n = len(insights)
    if n == 0:
        return []

    indices = list(range(n))
    random.shuffle(indices)

    batches = []
    for i in range(0, n, RELEVANCE_BATCH_SIZE):
        batch_indices = indices[i:i + RELEVANCE_BATCH_SIZE]
        batch_insights = [insights[j] for j in batch_indices]
        batches.append((batch_insights, batch_indices))

    log.info(f"Scoring {n} insights in {len(batches)} batches...")

    relevant_indices = set()
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_BATCHES) as executor:
        futures = {}
        for batch_num, (batch, batch_indices) in enumerate(batches):
            future = executor.submit(score_batch, query, batch, batch_indices)
            futures[future] = batch_num

        for future in as_completed(futures):
            batch_num = futures[future]
            try:
                result = future.result()
                relevant_indices.update(result)
                log.info(f"  Batch {batch_num + 1}/{len(batches)}: {len(result)} relevant")
            except Exception as e:
                log.error(f"  Batch {batch_num + 1} failed: {e}")

    relevant = [insights[i] for i in sorted(relevant_indices)]
    log.info(f"Total relevant: {len(relevant)} out of {n} scanned.")
    return relevant


# ---------------------------------------------------------------------------
# Pass 2: Synthesis
# ---------------------------------------------------------------------------

def format_insight_for_synthesis(insight):
    parts = [f"*{insight['title']}*"]
    if insight["user_name"]:
        parts.append(f"  User: {insight['user_name']}")
    if insight["user_email"]:
        parts.append(f"  Email: {insight['user_email']}")
    if insight["user_role"]:
        parts.append(f"  Role: {insight['user_role']}")
    if insight["source"]:
        parts.append(f"  Source: {insight['source']}")
    if insight["date"]:
        parts.append(f"  Date: {insight['date']}")
    if insight["short_description"]:
        parts.append(f"  Summary: {insight['short_description']}")
    if insight["long_description"]:
        parts.append(f"  Full feedback: {insight['long_description']}")
    if insight["follow_up_feedback"]:
        parts.append(f"  Follow-up reply: {insight['follow_up_feedback']}")
    return "\n".join(parts)


def synthesize_response(query, relevant_insights, total_scanned):
    system_prompt = """You are the Product Oracle, an expert product analyst for Fourwaves (an event management platform).

You are given a question from the product team and a curated set of user insights that are relevant to the question. Your job is to produce a comprehensive, evidence-packed response.

# RULES

1. *EXHAUSTIVE*: Every single insight provided to you is relevant. You must reference ALL of them. Do not skip any.
2. *EVIDENCE-BASED*: Pack your response with verbatim quotes from users. Use blockquotes (>) for every quote.
3. *TRACEABLE*: Always include the user's name (or email if no name) and date when citing an insight.
4. *STRUCTURED*: Group insights by theme or sub-topic when it makes sense. Use clear section headers.
5. *QUANTIFIED*: State how many users mentioned each theme (e.g., "3 organizers requested this").
6. *ACTIONABLE*: End with a brief "Key Takeaways" section summarizing the main patterns.

# OUTPUT FORMAT (STRICT SLACK MRKDWN)
- Bold: use single * on each side (*bold*). NEVER use ** double asterisks.
- Blockquotes: use > at the start of a line for user quotes.
- Headers: use *BOLD CAPS* with single asterisks.
- Use double line breaks between sections.
- NEVER use # characters or ** double asterisks.
- NEVER use markdown links [text](url).

# RESPONSE STRUCTURE
Start with a one-line summary of what you found (e.g., "Found 12 insights from 10 users about email automation.").
Then organize the evidence thematically.
End with *KEY TAKEAWAYS* section."""

    insight_texts = [format_insight_for_synthesis(ins) for ins in relevant_insights]
    knowledge_base = "\n\n---\n\n".join(insight_texts)

    user_prompt = f"""*QUERY*: {query}

*STATS*: {len(relevant_insights)} relevant insights found out of {total_scanned} total scanned.

*RELEVANT INSIGHTS*:

{knowledge_base}

Generate a comprehensive response addressing the query using ALL the insights above."""

    return _llm(system_prompt, user_prompt, model_hint="pro")


# ---------------------------------------------------------------------------
# Skill entry points
# ---------------------------------------------------------------------------

def handle_insights_query(message_text, call_llm_fn):
    """Handle a product question by scanning all insights."""
    set_llm(call_llm_fn)

    insights = load_cached_insights()
    log.info(f"Scanning {len(insights)} insights for: {message_text[:100]}")

    relevant = batch_score_relevance(message_text, insights)

    if not relevant:
        return f"I scanned all {len(insights)} user insights but couldn't find any that match your query. Try rephrasing?"

    response = synthesize_response(message_text, relevant, len(insights))
    return response


def handle_insights_followup(thread_context, followup_text, call_llm_fn):
    """Handle a follow-up question in an insights thread."""
    set_llm(call_llm_fn)

    # Classify: can we answer from context or need a new scan?
    classification_prompt = """You are evaluating a follow-up message in a Slack thread about user insights.

The thread already contains an analysis of user insights. The user is now asking a follow-up.

Classify the follow-up into one of:
- "context" — Can be answered from the thread (e.g., "tell me more about theme X", "summarize the key points", "draft an email to one of those users")
- "new_scan" — Needs a fresh scan of the insights database (e.g., "now find insights about mobile app bugs", "what about pricing complaints?")

Reply ONLY with "context" or "new_scan"."""

    classification = _llm(
        classification_prompt,
        f"Thread so far:\n{thread_context}\n\nFollow-up message:\n{followup_text}",
        model_hint="flash",
    ).strip().lower()

    if classification.startswith("new_scan"):
        # Run a full new scan
        return handle_insights_query(followup_text, call_llm_fn)

    # Answer from thread context
    response = _llm(
        """You are the Product Oracle, an expert product analyst for Fourwaves (an event management platform).

You are in an ongoing Slack thread where you already provided an analysis of user insights. The user is asking a follow-up question. Answer it using the information already in the thread conversation.

# RULES
1. Answer directly based on what was already discussed in the thread.
2. If the user asks you to draft an email, write it in the appropriate language (French if the user's insight quotes are in French, English otherwise).
3. Keep verbatim quotes when referencing insights.
4. Be concise — the user already has the full analysis, they just need a specific follow-up answered.

# OUTPUT FORMAT (STRICT SLACK MRKDWN)
- Bold: use single * on each side (*bold*). NEVER use ** double asterisks.
- Blockquotes: use > at the start of a line for quotes.
- NEVER use # characters or ** double asterisks.""",
        f"Thread conversation:\n{thread_context}\n\nFollow-up question:\n{followup_text}",
        model_hint="pro",
    )
    return response
