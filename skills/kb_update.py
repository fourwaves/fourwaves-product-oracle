"""
KB Update Skill — Update Intercom help center articles from Notion product release pages.

Flow:
  1. Extract Notion page URLs from the Slack message
  2. Fetch each Notion page's content via the Notion API
  3. Understand the scope of what was released
  4. Fetch ALL Intercom help center articles
  5. LLM identifies which articles need updates + what changes to make
  6. Format a proposal for the user in Slack mrkdwn (with article links)
  7. On approval, execute the changes via the Intercom API
"""

import os
import re
import json
import logging
from html.parser import HTMLParser

import requests

log = logging.getLogger("oracle.kb_update")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NOTION_TOKEN = os.environ["NOTION_TOKEN"]
NOTION_HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28",
}

INTERCOM_TOKEN = os.environ["INTERCOM_TOKEN"]
INTERCOM_HEADERS = {
    "Authorization": f"Bearer {INTERCOM_TOKEN}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

INTERCOM_AUTHOR_ID = 7827618  # Fourwaves help center author

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class HTMLTextExtractor(HTMLParser):
    """Strip HTML tags and return plain text."""
    def __init__(self):
        super().__init__()
        self._text = []

    def handle_data(self, data):
        self._text.append(data)

    def get_text(self):
        return " ".join(self._text)


def html_to_text(html):
    extractor = HTMLTextExtractor()
    extractor.feed(html or "")
    return extractor.get_text().strip()


def strip_code_fences(text):
    """Remove markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return text


def _sanitize_block_content(text):
    """Strip any literal triple-backticks from text destined for a code block.

    KB article prose never legitimately contains ```, so any occurrence is either
    LLM noise or accidental and would break Slack's fence parsing. Replace with
    three single quotes so the content stays readable.
    """
    if not text:
        return text
    return text.replace("```", "'''")


def render_changes_as_mrkdwn(changes_list, fr_changes_list=None):
    """Render a list of change dicts as Slack mrkdwn, deterministically.

    Each change dict has: type, section, why, before, after, screenshot_description.
    Only 'type' is required; other fields are used when present.

    If fr_changes_list is provided (matching length and order), each Before/After
    block is shown in both English and French so the user can validate both
    versions before approving.
    """
    if not changes_list:
        return ""

    has_fr = bool(fr_changes_list) and len(fr_changes_list) == len(changes_list)
    before_label = "*Before (EN):*" if has_fr else "*Before:*"
    after_label = "*After (EN):*" if has_fr else "*After:*"

    parts = []
    for i, change in enumerate(changes_list, 1):
        type_ = str(change.get("type", "UPDATE")).upper()
        section = change.get("section", "")
        why = change.get("why", "")
        change_fr = fr_changes_list[i - 1] if has_fr else None

        header = f"{i}. *[{type_}]*"
        if section:
            header += f' — Section: "{section}"'
        parts.append(header)
        if why:
            parts.append(f"Why: {why}")

        if type_ == "SCREENSHOT":
            desc = change.get("screenshot_description") or change.get("description", "")
            if desc:
                parts.append("")
                parts.append(desc)
        else:
            before = _sanitize_block_content((change.get("before") or "").strip())
            after = _sanitize_block_content((change.get("after") or "").strip())
            before_fr = _sanitize_block_content(((change_fr or {}).get("before") or "").strip())
            after_fr = _sanitize_block_content(((change_fr or {}).get("after") or "").strip())

            if before:
                parts.append("")
                parts.append(before_label)
                parts.append("```")
                parts.append(before)
                parts.append("```")
                if before_fr:
                    parts.append("")
                    parts.append("*Before (FR):*")
                    parts.append("```")
                    parts.append(before_fr)
                    parts.append("```")
            if after:
                parts.append("")
                parts.append(after_label)
                parts.append("```")
                parts.append(after)
                parts.append("```")
                if after_fr:
                    parts.append("")
                    parts.append("*After (FR):*")
                    parts.append("```")
                    parts.append(after_fr)
                    parts.append("```")

        parts.append("")  # blank line between changes

    return "\n".join(parts).rstrip() + "\n"


def extract_notion_urls(text):
    """Extract Notion page URLs from a Slack message."""
    # Slack wraps URLs in <url> or <url|label>
    urls = re.findall(r'<(https?://(?:www\.)?notion\.[a-z]+/[^|>]+)', text)
    if not urls:
        # Fallback: plain URLs
        urls = re.findall(r'https?://(?:www\.)?notion\.[a-z]+/\S+', text)
    return urls


def notion_url_to_page_id(url):
    """Extract Notion page ID from a URL. Handles various formats."""
    # Remove query params
    url = url.split("?")[0]
    # The page ID is typically the last 32 hex chars (with or without dashes)
    match = re.search(r'([a-f0-9]{32})$', url.replace("-", ""))
    if match:
        raw = match.group(1)
        return f"{raw[:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:]}"
    # Try the last path segment with dashes
    parts = url.rstrip("/").split("/")
    last = parts[-1].split("-")[-1] if parts else ""
    if len(last) == 32:
        return f"{last[:8]}-{last[8:12]}-{last[12:16]}-{last[16:20]}-{last[20:]}"
    return None


# ---------------------------------------------------------------------------
# Notion: fetch page content
# ---------------------------------------------------------------------------

def fetch_notion_page(page_id):
    """Fetch a Notion page's properties and only the QA Notes section content."""
    # Get page properties
    resp = requests.get(
        f"https://api.notion.com/v1/pages/{page_id}",
        headers=NOTION_HEADERS,
    )
    resp.raise_for_status()
    page = resp.json()

    # Get page title
    title = ""
    props = page.get("properties", {})
    for prop in props.values():
        if prop.get("type") == "title":
            title = " ".join(t.get("plain_text", "") for t in prop.get("title", []))
            break

    # Get all top-level blocks, then extract only content after "QA Notes" heading
    blocks_text = fetch_notion_blocks_qa_only(page_id)

    return {"id": page_id, "title": title, "content": blocks_text}


def fetch_notion_blocks_qa_only(page_id):
    """Fetch all top-level blocks and return only content under the QA Notes section."""
    url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    all_blocks = []
    has_more = True
    start_cursor = None

    while has_more:
        params = {"page_size": 100}
        if start_cursor:
            params["start_cursor"] = start_cursor
        resp = requests.get(url, headers=NOTION_HEADERS, params=params)
        resp.raise_for_status()
        data = resp.json()
        all_blocks.extend(data.get("results", []))
        has_more = data.get("has_more", False)
        start_cursor = data.get("next_cursor")

    # Find the QA Notes heading and collect everything after it
    qa_section_started = False
    qa_blocks = []

    for block in all_blocks:
        block_type = block.get("type", "")
        block_data = block.get(block_type, {})
        rich_text = block_data.get("rich_text", [])
        text = " ".join(t.get("plain_text", "") for t in rich_text)

        if block_type.startswith("heading") and "qa notes" in text.lower().replace("🧪", "").strip().lower():
            qa_section_started = True
            continue  # Skip the heading itself

        if qa_section_started:
            qa_blocks.append(block)

    if not qa_blocks:
        log.warning(f"No 'QA Notes' section found in page {page_id}. Returning empty content.")
        return ""

    # Render the QA blocks to text
    return render_blocks(qa_blocks)


def fetch_child_blocks(block_id):
    """Fetch all child blocks of a given block."""
    url = f"https://api.notion.com/v1/blocks/{block_id}/children"
    all_blocks = []
    has_more = True
    start_cursor = None

    while has_more:
        params = {"page_size": 100}
        if start_cursor:
            params["start_cursor"] = start_cursor
        resp = requests.get(url, headers=NOTION_HEADERS, params=params)
        resp.raise_for_status()
        data = resp.json()
        all_blocks.extend(data.get("results", []))
        has_more = data.get("has_more", False)
        start_cursor = data.get("next_cursor")

    return all_blocks


def render_blocks(blocks, depth=0):
    """Render a list of Notion blocks to plain text, recursing into children."""
    if depth > 5:
        return ""

    parts = []
    for block in blocks:
        block_type = block.get("type", "")
        block_data = block.get(block_type, {})

        rich_text = block_data.get("rich_text", [])
        text = " ".join(t.get("plain_text", "") for t in rich_text)

        if block_type.startswith("heading"):
            level = block_type[-1]
            parts.append(f"\n{'#' * int(level)} {text}\n")
        elif block_type == "paragraph":
            parts.append(text)
        elif block_type in ("bulleted_list_item", "numbered_list_item"):
            parts.append(f"- {text}")
        elif block_type == "to_do":
            checked = "x" if block_data.get("checked") else " "
            parts.append(f"[{checked}] {text}")
        elif block_type == "code":
            lang = block_data.get("language", "")
            parts.append(f"```{lang}\n{text}\n```")
        elif block_type == "toggle":
            parts.append(f"> {text}")
        elif block_type == "callout":
            icon = block_data.get("icon", {}).get("emoji", "")
            parts.append(f"{icon} {text}")
        elif block_type == "image":
            caption = " ".join(t.get("plain_text", "") for t in block_data.get("caption", []))
            parts.append(f"[Image: {caption}]" if caption else "[Image]")
        elif block_type == "divider":
            parts.append("---")
        elif text:
            parts.append(text)

        # Recurse into children
        if block.get("has_children"):
            child_blocks = fetch_child_blocks(block["id"])
            child_text = render_blocks(child_blocks, depth + 1)
            if child_text:
                parts.append(child_text)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Intercom: fetch all articles
# ---------------------------------------------------------------------------

def fetch_all_intercom_articles():
    """Fetch all published Intercom help center articles."""
    articles = []
    page = 1
    while True:
        resp = requests.get(
            "https://api.intercom.io/articles",
            headers=INTERCOM_HEADERS,
            params={"page": page, "per_page": 150},
        )
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("data", data.get("articles", []))
        articles.extend(batch)
        pages = data.get("pages", {})
        if page >= pages.get("total_pages", 1):
            break
        page += 1

    log.info(f"Fetched {len(articles)} Intercom articles.")
    return articles


def format_article_for_scoring(article):
    """Format an article compactly for LLM relevance scoring."""
    body_text = html_to_text(article.get("body", ""))[:600]
    return (
        f"[ID:{article['id']}] {article.get('title', '?')}\n"
        f"  Description: {article.get('description', '')[:200]}\n"
        f"  Content: {body_text}\n"
        f"  URL: {article.get('url', '')}"
    )


def _raise_for_intercom(resp, op_label):
    """Like resp.raise_for_status(), but include Intercom's response body in the message.

    Intercom returns useful error details in the JSON body (e.g.,
    {"errors": [{"code": "parameter_invalid", "message": "..."}]}). The default
    raise_for_status throws those away.
    """
    if resp.ok:
        return
    body_text = ""
    try:
        data = resp.json()
        errors = data.get("errors") if isinstance(data, dict) else None
        if isinstance(errors, list) and errors:
            body_text = "; ".join(
                f"{e.get('code', '?')}: {e.get('message', '')}".strip(": ")
                for e in errors if isinstance(e, dict)
            )
        if not body_text:
            body_text = json.dumps(data)[:500]
    except Exception:
        body_text = (resp.text or "")[:500]
    raise requests.HTTPError(
        f"Intercom {op_label} {resp.status_code}: {body_text}",
        response=resp,
    )


def update_intercom_article(article_id, title=None, body=None, description=None, translated_content=None):
    """Update an Intercom article. Use translated_content to update specific locales."""
    payload = {}
    if title:
        payload["title"] = title
    if body:
        payload["body"] = body
    if description:
        payload["description"] = description
    if translated_content:
        payload["translated_content"] = translated_content

    resp = requests.put(
        f"https://api.intercom.io/articles/{article_id}",
        headers=INTERCOM_HEADERS,
        json=payload,
    )
    _raise_for_intercom(resp, f"update {article_id}")
    return resp.json()


def create_intercom_article(title, body, description="", parent_id=None, parent_type=None, state="draft", translated_content=None):
    """Create a new Intercom article (as draft by default)."""
    payload = {
        "title": title,
        "body": body,
        "author_id": INTERCOM_AUTHOR_ID,
        "state": state,
    }
    if description:
        payload["description"] = description
    if parent_id:
        payload["parent_id"] = parent_id
        payload["parent_type"] = parent_type or "collection"
    if translated_content:
        payload["translated_content"] = translated_content

    resp = requests.post(
        "https://api.intercom.io/articles",
        headers=INTERCOM_HEADERS,
        json=payload,
    )
    _raise_for_intercom(resp, "create article")
    return resp.json()


def fetch_intercom_article_full(article_id):
    """Fetch a single article (includes translated_content)."""
    resp = requests.get(
        f"https://api.intercom.io/articles/{article_id}",
        headers=INTERCOM_HEADERS,
    )
    resp.raise_for_status()
    return resp.json()


def get_article_fr_body(article):
    """Return the French body of an Intercom article, or empty string."""
    return ((article.get("translated_content") or {}).get("fr") or {}).get("body", "")


def translate_changes_to_fr(article, changes_list_en, call_llm_fn):
    """Translate a list of English KB change proposals into matching French ones.

    Looks at the article's French body to find verbatim 'before' text and
    translates 'after' text naturally. Returns a list with the same length and
    order as changes_list_en, or None if the article has no French version.
    """
    fr_body = get_article_fr_body(article)
    if not fr_body or not changes_list_en:
        return None

    fr_text = html_to_text(fr_body)[:4000]

    system_prompt = """You are translating help center change proposals from English to French.

You are given:
1. A list of English change proposals for an article (each has type, section, why, before, after).
2. The current French version of the article body.

For each English change, produce the equivalent French change as a JSON object.

RULES:
- Output a JSON array with the SAME length and SAME order as the input.
- For UPDATE/REMOVE: "before" must be the EXACT existing French text from the article that will be replaced — find and copy it verbatim from the French body. Do not paraphrase.
- For UPDATE/ADD: "after" is the natural French translation of the new English text (not word-for-word).
- Translate "section" to its French equivalent if a corresponding section exists in the French article; otherwise keep the English section name.
- Translate "why" to French.
- Translate UI labels using their French equivalents in the Fourwaves app (e.g., "Subject" → "Sujet", "Save and continue" → "Sauvegarder et continuer").
- Use "vous" form (formal).
- Présent, état actuel uniquement. N'utilisez PAS « désormais », « maintenant », « a été mis à jour », « récemment », « dorénavant », ni aucune formulation qui évoque un changement par rapport à un état antérieur.
- Each change object has the same fields as input: {type, section, why, before, after, screenshot_description}.
- For ADD: "before" can be empty string. For REMOVE: "after" can be empty string. For SCREENSHOT: include "screenshot_description" only.
- Return ONLY the JSON array, no prose, no code fences."""

    user_prompt = (
        f"ENGLISH CHANGES (JSON):\n{json.dumps(changes_list_en, ensure_ascii=False)}\n\n"
        f"CURRENT FRENCH ARTICLE BODY (plain text, for finding verbatim 'before' text):\n{fr_text}"
    )

    try:
        raw = call_llm_fn(system_prompt, user_prompt, model_hint="pro")
        cleaned = strip_code_fences(raw)
        fr_list = json.loads(cleaned)
        if isinstance(fr_list, list) and len(fr_list) == len(changes_list_en):
            return fr_list
        log.warning(
            f"FR translation length mismatch for article {article.get('id')}: "
            f"got {len(fr_list) if isinstance(fr_list, list) else 'non-list'}, "
            f"expected {len(changes_list_en)}."
        )
    except json.JSONDecodeError as e:
        log.warning(f"FR translation JSON parse failed for article {article.get('id')}: {e}")
    except Exception as e:
        log.warning(f"FR translation failed for article {article.get('id')}: {e}")
    return None


def translate_new_article_proposal_to_fr(en_plan, call_llm_fn):
    """Translate a new-article proposal (title/description/outline) to French.

    Returns a dict with 'title', 'description', 'outline' in French, or None.
    """
    if not en_plan:
        return None

    system_prompt = """Translate this help center new-article proposal from English to French.

Return a JSON object: {"title": "...", "description": "...", "outline": "..."}

RULES:
- Naturally translate, not word-for-word.
- Use "vous" form (formal).
- Translate UI labels using their French equivalents in the Fourwaves app.
- TITLE: imperative verb form in French, sentence case, 3-8 words.
- DESCRIPTION: start with "Cet article explique comment..." — one sentence, period at end.
- OUTLINE: keep the same structure (section headers + bullets) as the English outline, translated naturally.
- Présent, état actuel uniquement. N'utilisez PAS « désormais », « maintenant », « a été ajouté », « récemment », « nouvelle fonctionnalité », « dorénavant ».
- Return ONLY the JSON object, no prose, no code fences."""

    user_prompt = (
        f"ENGLISH TITLE: {en_plan.get('title', '')}\n"
        f"ENGLISH DESCRIPTION: {en_plan.get('description', '')}\n"
        f"ENGLISH OUTLINE:\n{en_plan.get('outline', '')}"
    )

    try:
        raw = call_llm_fn(system_prompt, user_prompt, model_hint="pro")
        cleaned = strip_code_fences(raw)
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return {
                "title": data.get("title", ""),
                "description": data.get("description", ""),
                "outline": data.get("outline", ""),
            }
    except json.JSONDecodeError as e:
        log.warning(f"FR new-article JSON parse failed: {e}")
    except Exception as e:
        log.warning(f"FR new-article translation failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Skill handler: propose KB changes
# ---------------------------------------------------------------------------

def handle_kb_update(message_text, call_llm_fn):
    """
    Phase 1: Analyze Notion pages + Intercom articles, propose changes.
    Returns a Slack mrkdwn message with the proposed plan.
    """
    # 1. Extract Notion URLs
    urls = extract_notion_urls(message_text)
    if not urls:
        return "I couldn't find any Notion page URLs in your message. Please share the Notion links for the features you released."

    log.info(f"Found {len(urls)} Notion URL(s): {urls}")

    # 2. Fetch Notion pages
    notion_pages = []
    for url in urls:
        page_id = notion_url_to_page_id(url)
        if not page_id:
            log.warning(f"Could not extract page ID from: {url}")
            continue
        try:
            page = fetch_notion_page(page_id)
            notion_pages.append(page)
            log.info(f"  Fetched Notion page: {page['title']}")
        except Exception as e:
            log.error(f"  Failed to fetch Notion page {page_id}: {e}")

    if not notion_pages:
        return "I couldn't fetch any of the Notion pages. Please check the URLs and make sure the Notion integration has access to those pages."

    # Check for empty QA Notes sections
    empty_pages = [p for p in notion_pages if not p["content"].strip()]
    if empty_pages:
        empty_names = ", ".join(f"*{p['title']}*" for p in empty_pages)
        if len(empty_pages) == len(notion_pages):
            return f"I couldn't find any content under the *QA Notes* section for: {empty_names}. I only read the QA Notes section of Notion feature pages. Please make sure the section exists and has content."

    # 3. Understand the release scope (from QA Notes only)
    release_summary_prompt = """You are reading QA Notes from product feature pages. These notes describe what was built and how the feature works.

Create a concise bullet-point summary of the key features and changes released. Each bullet should be one clear sentence.

OUTPUT FORMAT (strict Slack mrkdwn):
- Use single * for bold (*bold*). NEVER use ** or ## or ### or any markdown headers.
- Use plain - for bullet points.
- Keep it to one section: a flat bullet list of key changes. No sub-sections, no numbered lists, no headers.
- Be specific but concise. Each bullet = one feature or change."""

    pages_content = "\n\n===\n\n".join(
        f"PAGE: {p['title']}\n\nQA NOTES:\n{p['content']}" for p in notion_pages
    )

    release_summary = call_llm_fn(release_summary_prompt, pages_content, model_hint="pro")
    log.info(f"Release summary generated ({len(release_summary)} chars).")

    # 4. Fetch all Intercom articles
    articles = fetch_all_intercom_articles()
    published = [a for a in articles if a.get("state") == "published"]
    log.info(f"Analyzing against {len(published)} published articles.")

    # 5. Score relevance in batches
    BATCH_SIZE = 15
    relevant_articles = []

    for i in range(0, len(published), BATCH_SIZE):
        batch = published[i:i + BATCH_SIZE]
        batch_text = "\n\n".join(format_article_for_scoring(a) for a in batch)

        scoring_prompt = """You are identifying which help center articles need to be updated based on a product release.

Given the release summary and a batch of help center articles, return a JSON array of article IDs that are relevant and likely need updates.

An article is RELEVANT if:
- It directly covers the SAME feature or workflow that was changed in this release
- It describes specific functionality that was modified, added, or removed by this release
- The article's existing content would be INCORRECT or INCOMPLETE without an update

An article is NOT relevant if:
- It merely mentions a related concept but covers a different feature
- It's in the same general area (e.g., both involve forms) but the release doesn't change what the article describes
- The connection is only tangential or thematic

Be PRECISE — only include articles whose content is directly affected by the release. Do not include articles just because they share a keyword or general topic. We want quality over quantity.

If the user's request includes explicit exclusions or scoping constraints (e.g., "ignore X", "skip Y", "except for Z"), honor them: exclude any article whose only relevance is the excluded topic.

Return ONLY a JSON array of ID numbers. Example: [123, 456]
If none are relevant, return: []"""

        user_prompt = (
            f"RELEASE SUMMARY:\n{release_summary}\n\n"
            f"ARTICLES TO EVALUATE:\n{batch_text}\n\n"
            f"USER REQUEST (verbatim — honor any explicit exclusions or scoping constraints stated here):\n{message_text}"
        )

        try:
            raw = call_llm_fn(scoring_prompt, user_prompt, model_hint="flash")
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
            ids = json.loads(cleaned)
            if isinstance(ids, list):
                id_set = {str(x) for x in ids}
                for a in batch:
                    if str(a["id"]) in id_set:
                        relevant_articles.append(a)
        except Exception as e:
            log.warning(f"Batch scoring failed: {e}")

    log.info(f"Found {len(relevant_articles)} potentially relevant articles.")

    if not relevant_articles:
        # Still suggest creating a new article
        new_article_proposal = call_llm_fn(
            """Based on this product release, propose a new help center article.

TITLE FORMAT RULES:
- Use imperative verb form (e.g., "Generate badges", "Manage tracks")
- Do NOT use gerund form or "How to..." prefix
- Use sentence case (only capitalize the first letter of the first word, e.g., "Generate badges"), 3-8 words

DESCRIPTION FORMAT RULES:
- Start with "This article explains how to..." or "This article explains how you can..."
- Exactly ONE sentence, ending with a period, 60-120 characters

If the user's request includes explicit exclusions (e.g., "ignore X"), honor them: do not include the excluded topic in the article.

Return a JSON object with: {"title": "...", "description": "...", "outline": "..."}
The outline should be a bullet-pointed structure of what the article should cover.""",
            f"Release:\n{release_summary}\n\nUSER REQUEST (verbatim — honor any explicit exclusions or scoping constraints stated here):\n{message_text}",
            model_hint="pro",
        )
        return (
            f"I scanned all {len(published)} published help center articles and none seem directly affected by this release.\n\n"
            f"However, I recommend *creating a new article*:\n\n{new_article_proposal}\n\n"
            f"Reply with *yes, proceed* to create and publish the article, or tell me what to adjust."
        )

    # 6. Detailed analysis: for each relevant article, determine exact changes
    detailed_proposals = []

    for article in relevant_articles:
        body_text = html_to_text(article.get("body", ""))
        body_html = article.get("body", "")[:4000]
        article_detail = (
            f"ARTICLE: {article['title']}\n"
            f"ID: {article['id']}\n"
            f"URL: {article.get('url', 'N/A')}\n"
            f"Description: {article.get('description', '')}\n"
            f"Current content (plain text):\n{body_text[:3000]}\n\n"
            f"Current HTML (for header analysis):\n{body_html}"
        )

        proposal_prompt = """You are a technical writer updating help center documentation after a product release.

Given the release details and an existing help center article, determine what specific changes should be made.

CRITICAL: Only propose changes that are DIRECTLY related to this article's topic. If the release affects a different feature than what this article covers, return an empty list []. Do NOT propose adding content about tangentially related features.

SCOPE — what NOT to propose:
- Do NOT propose changes whose only purpose is to fix existing inconsistencies in the article (header levels, formatting, styling, typos, tone, restructuring, etc.). The user is reviewing changes for THIS release only — drive-by cleanup of pre-existing content makes the approval process painful.
- Only propose a change if the release content itself requires it. Existing sections that are not affected by the release must be left alone, even if their headers or style are inconsistent.
- If the user's request includes explicit exclusions or constraints (e.g., "ignore X", "skip Y", "except for Z"), honor them: do not propose changes about the excluded topics, even if they would otherwise be relevant.

HEADER HIERARCHY RULES (apply ONLY to NEW sections being added by this release):
- When ADDING a new section because of the release, look at the existing HTML headers around where it will be inserted and pick the matching level: a new top-level section uses the same level as other top-level sections, a new sub-section uses the same level as sibling sub-sections.
- Do NOT propose normalizing or "fixing" header levels of existing sections, even if you notice they are inconsistent. That is out of scope for this skill.

WRITING STYLE for any new/edited text (must match existing articles):
- Short, straight-to-the-point sentences. No filler or marketing language.
- Use bullet lists or numbered lists whenever possible instead of paragraphs.
- Step-by-step instructions for how-to content (1. Go to... 2. Click... 3. Select...).
- Speak directly to the user ("You can...", "Click...", "Go to...").
- Keep the tone helpful and professional, not casual or overly friendly.
- Present tense, current-state only. The knowledge base describes how the product works today — NOT that something changed. Do NOT use "now", "will now", "has been updated", "recently", "from now on", or any wording that signals a change from a previous state. Example: write "Variables display their short label." NOT "Variables will now display their short label.".

OUTPUT FORMAT: Return ONLY a JSON array of change objects. No prose, no markdown, no code fences around the JSON. If no changes are needed, return: []

Each change object has these fields:
{
  "type": "UPDATE" | "ADD" | "REMOVE" | "SCREENSHOT",
  "section": "name of the section in the article",
  "why": "one sentence explaining why this change is needed",
  "before": "exact current text that will be changed (REQUIRED for UPDATE and REMOVE; omit or set to empty for ADD and SCREENSHOT)",
  "after": "exact new text that will replace it (REQUIRED for UPDATE and ADD; omit or set to empty for REMOVE and SCREENSHOT)",
  "screenshot_description": "what screenshot to add or update (only for SCREENSHOT type)"
}

RULES:
- "before" and "after" must contain the actual prose text, NOT a description of it.
- Plain text only inside "before" and "after" — no markdown, no code fences, no HTML tags.
- Keep each change narrowly scoped (one paragraph or one bullet point per change).

Example output:
[
  {"type": "UPDATE", "section": "Content and Formatting", "why": "Clarify which variable label value is shown on certificates.", "before": "You can insert form variables that will be automatically replaced by the corresponding form values for each registration.", "after": "You can insert form variables that will be automatically replaced by the corresponding form values for each registration. The system uses the short label value for variables."},
  {"type": "ADD", "section": "Preview", "why": "Document the new preview step.", "after": "Click Preview to see what the certificate will look like before distributing it."}
]"""

        user_prompt = (
            f"RELEASE SUMMARY:\n{release_summary}\n\n"
            f"{article_detail}\n\n"
            f"USER REQUEST (verbatim — honor any explicit exclusions or scoping constraints stated here):\n{message_text}"
        )

        try:
            raw = call_llm_fn(proposal_prompt, user_prompt, model_hint="pro")
            cleaned = strip_code_fences(raw)
            # Tolerate legacy NO_CHANGES outputs in case the LLM reverts to the old format
            if cleaned.strip().upper().startswith("NO_CHANGES"):
                continue
            changes_list = json.loads(cleaned)
            if not isinstance(changes_list, list):
                log.warning(f"Article {article['id']}: proposal was not a JSON list, skipping.")
                continue
            if changes_list:
                # Generate French versions of the changes for bilingual review.
                # Re-fetch the article if translated_content isn't already populated.
                article_for_fr = article
                if not get_article_fr_body(article):
                    try:
                        article_for_fr = fetch_intercom_article_full(article["id"])
                    except Exception as e:
                        log.warning(f"Couldn't fetch full article {article['id']} for FR: {e}")
                fr_changes_list = translate_changes_to_fr(article_for_fr, changes_list, call_llm_fn)

                detailed_proposals.append({
                    "article_id": str(article["id"]),
                    "article_title": article["title"],
                    "article_url": article.get("url", ""),
                    "changes_list": changes_list,
                    "fr_changes_list": fr_changes_list,
                    "changes": render_changes_as_mrkdwn(changes_list, fr_changes_list),
                })
        except json.JSONDecodeError as e:
            log.warning(f"Failed to parse proposal JSON for article {article['id']}: {e}. Raw: {raw[:300]!r}")
        except Exception as e:
            log.warning(f"Failed to analyze article {article['id']}: {e}")

    # 7. Check if a new article should also be created
    new_article_prompt = """Based on this product release and the list of existing articles being updated, determine if a NEW help center article should be created.

A new article is needed if:
- The release introduces a completely new feature not covered by any existing article
- The scope of changes is large enough to warrant a standalone article
- Users would benefit from a dedicated guide

TITLE FORMAT RULES (must follow exactly):
- Use imperative verb form (e.g., "Generate badges", "Create rooms and assign them to sessions", "Manage tracks")
- Do NOT use gerund form ("Generating...", "Creating...")
- Do NOT use "How to..." prefix
- Use sentence case (only capitalize the first letter of the first word, e.g., "Generate badges")
- Keep it concise: 3-8 words
- Be specific about the object being acted on

DESCRIPTION FORMAT RULES (must follow exactly):
- Start with "This article explains how to..." or "This article explains how you can..."
- Write exactly ONE sentence
- End with a period
- Target 60-120 characters (roughly 10-20 words)
- Mention the key action AND its context/purpose
- Do NOT start with "In this article..." or "Learn how to..."

If the user's request includes explicit exclusions (e.g., "ignore X"), honor them: do not propose a new article whose primary topic is the excluded one, and do not include the excluded topic in the outline.

If a new article is needed, return a JSON object:
{"needed": true, "title": "...", "description": "...", "outline": "full draft of the article content in plain text with section headers and bullet points — this will be shown in a Slack code block for review"}

If NOT needed, return: {"needed": false}"""

    existing_titles = "\n".join(f"- {p['article_title']}" for p in detailed_proposals)
    new_article_raw = call_llm_fn(
        new_article_prompt,
        (
            f"Release:\n{release_summary}\n\n"
            f"Articles being updated:\n{existing_titles}\n\n"
            f"USER REQUEST (verbatim — honor any explicit exclusions or scoping constraints stated here):\n{message_text}"
        ),
        model_hint="pro",
    )

    new_article_plan = None
    try:
        cleaned = new_article_raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        parsed = json.loads(cleaned)
        if parsed.get("needed"):
            new_article_plan = parsed
    except (json.JSONDecodeError, AttributeError):
        pass

    # Translate the new-article proposal to French for bilingual review.
    if new_article_plan:
        fr_plan = translate_new_article_proposal_to_fr(new_article_plan, call_llm_fn)
        if fr_plan:
            new_article_plan["fr_title"] = fr_plan.get("title", "")
            new_article_plan["fr_description"] = fr_plan.get("description", "")
            new_article_plan["fr_outline"] = fr_plan.get("outline", "")

    # 8. Format the Slack response
    return format_proposal_message(detailed_proposals, new_article_plan, len(published), notion_pages, release_summary)


def render_proposals_section(proposals, new_article_plan):
    """Render the ARTICLES TO UPDATE + NEW ARTICLE sections as Slack mrkdwn."""
    parts = []
    if proposals:
        parts.append(f"*ARTICLES TO UPDATE ({len(proposals)}):*\n")
        for i, p in enumerate(proposals, 1):
            parts.append(f"*{i}. {p['article_title']}*")
            if p.get("article_url"):
                parts.append(f"    {p['article_url']}")
            parts.append(p["changes"])
            parts.append("")

    if new_article_plan:
        has_fr = bool(new_article_plan.get("fr_title") or new_article_plan.get("fr_outline"))
        title_label = "*Title (EN):*" if has_fr else "*Title:*"
        desc_label = "*Description (EN):*" if has_fr else "*Description:*"
        outline_label = "*Outline (EN):*" if has_fr else None

        parts.append("*NEW ARTICLE RECOMMENDED:*\n")
        parts.append(f"{title_label} {new_article_plan.get('title', 'TBD')}")
        if has_fr:
            parts.append(f"*Title (FR):* {new_article_plan.get('fr_title', '')}")
        parts.append(f"{desc_label} {new_article_plan.get('description', '')}")
        if has_fr:
            parts.append(f"*Description (FR):* {new_article_plan.get('fr_description', '')}")
        parts.append("")

        outline = _sanitize_block_content((new_article_plan.get("outline") or "").strip())
        if outline:
            if outline_label:
                parts.append(outline_label)
            parts.append("```")
            parts.append(outline)
            parts.append("```")
        outline_fr = _sanitize_block_content((new_article_plan.get("fr_outline") or "").strip())
        if outline_fr:
            parts.append("")
            parts.append("*Outline (FR):*")
            parts.append("```")
            parts.append(outline_fr)
            parts.append("```")
        parts.append("")
    return "\n".join(parts)


def _render_article_message(index, proposal):
    """Build a single Slack message for one article's proposed changes."""
    lines = [f"*{index}. {proposal['article_title']}*"]
    if proposal.get("article_url"):
        lines.append(f"    {proposal['article_url']}")
    lines.append(proposal["changes"])
    return "\n".join(lines).rstrip()


def _render_new_article_message(new_article_plan):
    """Build a single Slack message for a recommended new article."""
    has_fr = bool(new_article_plan.get("fr_title") or new_article_plan.get("fr_outline"))
    title_label = "*Title (EN):*" if has_fr else "*Title:*"
    desc_label = "*Description (EN):*" if has_fr else "*Description:*"
    outline_label = "*Outline (EN):*" if has_fr else None

    lines = ["*NEW ARTICLE RECOMMENDED:*\n"]
    lines.append(f"{title_label} {new_article_plan.get('title', 'TBD')}")
    if has_fr:
        lines.append(f"*Title (FR):* {new_article_plan.get('fr_title', '')}")
    lines.append(f"{desc_label} {new_article_plan.get('description', '')}")
    if has_fr:
        lines.append(f"*Description (FR):* {new_article_plan.get('fr_description', '')}")
    lines.append("")

    outline = _sanitize_block_content((new_article_plan.get("outline") or "").strip())
    if outline:
        if outline_label:
            lines.append(outline_label)
        lines.append("```")
        lines.append(outline)
        lines.append("```")
    outline_fr = _sanitize_block_content((new_article_plan.get("fr_outline") or "").strip())
    if outline_fr:
        lines.append("")
        lines.append("*Outline (FR):*")
        lines.append("```")
        lines.append(outline_fr)
        lines.append("```")
    return "\n".join(lines).rstrip()


def format_proposal_message(proposals, new_article_plan, total_articles, notion_pages, release_summary):
    """Build the proposal as a list of Slack messages (one per article + header + trailer).

    Posting one Slack message per article guarantees that code fences inside an
    article never accidentally span a Slack message boundary, which is the only
    fully reliable way to avoid mrkdwn rendering issues.
    """
    messages = []

    # 1. Header — feature summary + scan count
    feature_names = ", ".join(p["title"] for p in notion_pages)
    header = (
        "*FEATURE SUMMARY (from QA Notes):*\n\n"
        f"{release_summary}\n\n"
        f"_Scanned {total_articles} published help center articles for: {feature_names}_"
    )
    messages.append(header)

    # 2. Empty-state short-circuit
    if not proposals and not new_article_plan:
        messages.append(
            "After detailed analysis, no changes are needed for any existing articles "
            "and no new article is recommended."
        )
        return messages

    # 3. One message per article
    if proposals:
        messages.append(f"*ARTICLES TO UPDATE ({len(proposals)}):*")
        for i, p in enumerate(proposals, 1):
            messages.append(_render_article_message(i, p))

    # 4. New article (own message)
    if new_article_plan:
        messages.append(_render_new_article_message(new_article_plan))

    # 5. Trailer — note + approval prompt
    trailer = (
        "_Note: When approved, updates will be applied to both English and French "
        "versions, and any new article will be created and published._\n\n"
        "---\n"
        "You can ask me to revise the proposal, or reply *yes, proceed* to apply the changes."
    )
    messages.append(trailer)

    return messages


# ---------------------------------------------------------------------------
# Skill handler: revise KB proposal based on user feedback
# ---------------------------------------------------------------------------

def handle_kb_revision(thread_context, revision_request, call_llm_fn):
    """
    Revise a KB update proposal based on user feedback.
    Uses the full thread context to understand the original proposal and corrections.
    Returns an updated Slack mrkdwn proposal.
    """
    revision_prompt = """You are revising a knowledge base update proposal based on user feedback.

You have the full Slack thread with the original proposal and the user's corrections. Generate a REVISED proposal that incorporates ALL of the user's feedback.

IMPORTANT:
- Read the full thread carefully to understand what was originally proposed.
- Apply ALL corrections the user asked for. Do not ignore any feedback.
- If the user asked to remove a change, remove it entirely from the output.
- If the user corrected factual details, apply those corrections to the before/after text.
- Show the FULL revised proposal (all articles and all changes that still apply), not just what changed.

SCOPE — what NOT to propose:
- Do NOT propose changes whose only purpose is to fix existing inconsistencies in the article (header levels, formatting, styling, typos, tone, restructuring, etc.). Only propose a change if the release content itself requires it. Existing sections that are not affected by the release must be left alone, even if their headers or style are inconsistent.

HEADER HIERARCHY RULES (apply ONLY to NEW sections being added by this release):
- When ADDING a new section, match the header level used by existing same-level sections in the article (top-level new section = same level as other top-level sections; sub-section = same level as sibling sub-sections).
- Do NOT propose normalizing or "fixing" header levels of existing sections.

WRITING STYLE for any new/edited text (must match existing Fourwaves help center articles):
- Short, straight-to-the-point sentences. No filler or marketing language.
- Use bullet lists or numbered lists whenever possible instead of paragraphs.
- Step-by-step instructions for how-to content (1. Go to... 2. Click... 3. Select...).
- Speak directly to the user ("You can...", "Click...", "Go to...").
- Keep the tone helpful and professional, not casual or overly friendly.
- Present tense, current-state only. The knowledge base describes how the product works today — NOT that something changed. Do NOT use "now", "will now", "has been updated", "recently", "from now on", or any wording that signals a change from a previous state. Example: write "Variables display their short label." NOT "Variables will now display their short label.".

OUTPUT FORMAT: Return ONLY a JSON object with this shape. No prose, no markdown, no code fences around the JSON:

{
  "articles": [
    {
      "article_title": "exact title of the article",
      "article_url": "url of the article (copy from thread if available, else empty string)",
      "changes_en": [
        {
          "type": "UPDATE" | "ADD" | "REMOVE" | "SCREENSHOT",
          "section": "name of the section in the article",
          "why": "one sentence explaining why this change is needed",
          "before": "exact current English text (REQUIRED for UPDATE and REMOVE; omit or empty for ADD and SCREENSHOT)",
          "after": "exact new English text (REQUIRED for UPDATE and ADD; omit or empty for REMOVE and SCREENSHOT)",
          "screenshot_description": "what screenshot to add or update (only for SCREENSHOT type)"
        }
      ],
      "changes_fr": [
        { "same shape as changes_en, but in French. Same length and order. Use null instead of an array if the original thread did not include French versions for this article." }
      ]
    }
  ],
  "new_article": null | {
    "title": "English title",
    "description": "English description",
    "outline": "English outline",
    "fr_title": "French title (or empty string if the thread had no French version)",
    "fr_description": "French description (or empty string)",
    "fr_outline": "French outline (or empty string)"
  }
}

RULES:
- "before" and "after" must contain actual prose text, NOT a description of it.
- Plain text only inside "before" and "after" — no markdown, no code fences, no HTML tags.
- Keep each change narrowly scoped (one paragraph or one bullet point per change).
- If an article has no remaining changes after the revision, omit it from "articles".
- If no new article is needed, set "new_article" to null.
- For "changes_fr": use natural French (not word-for-word), use "vous" form, translate UI labels using their French equivalents in the Fourwaves app, and apply the same present-tense / current-state rule (no « désormais », « maintenant », « a été mis à jour », etc.). Match length and order with "changes_en". If the original thread did not include French for an article, set "changes_fr" to null."""

    raw = call_llm_fn(
        revision_prompt,
        f"Full thread conversation:\n{thread_context}\n\nUser's latest revision request:\n{revision_request}",
        model_hint="pro",
    )

    try:
        cleaned = strip_code_fences(raw)
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        log.warning(f"Revision JSON parse failed: {e}. Raw: {raw[:300]!r}")
        return (
            "I couldn't parse my own revised proposal. Please ask me to revise again, "
            "or describe the corrections in a different way."
        )

    revised_proposals = []
    for a in data.get("articles", []) or []:
        # Accept both new ("changes_en") and old ("changes") field names for backward compat
        changes_list = a.get("changes_en") or a.get("changes") or []
        if not changes_list:
            continue
        fr_changes_list = a.get("changes_fr") or None
        if not isinstance(fr_changes_list, list):
            fr_changes_list = None
        if fr_changes_list and len(fr_changes_list) != len(changes_list):
            fr_changes_list = None
        revised_proposals.append({
            "article_title": a.get("article_title", ""),
            "article_url": a.get("article_url", ""),
            "changes_list": changes_list,
            "fr_changes_list": fr_changes_list,
            "changes": render_changes_as_mrkdwn(changes_list, fr_changes_list),
        })

    new_article_plan = data.get("new_article") or None

    if not revised_proposals and not new_article_plan:
        return [
            "*REVISED PROPOSAL:*\n\n"
            "No changes remain after your revisions. Let me know if you'd like to start over."
        ]

    messages = ["*REVISED PROPOSAL:*"]

    if revised_proposals:
        messages.append(f"*ARTICLES TO UPDATE ({len(revised_proposals)}):*")
        for i, p in enumerate(revised_proposals, 1):
            messages.append(_render_article_message(i, p))

    if new_article_plan:
        messages.append(_render_new_article_message(new_article_plan))

    messages.append(
        "_Note: When approved, updates will be applied to both English and French "
        "versions, and any new article will be created and published._\n\n"
        "---\n"
        "You can ask me to revise again, or reply *yes, proceed* to apply the changes."
    )
    return messages


# ---------------------------------------------------------------------------
# Skill handler: execute approved changes
# ---------------------------------------------------------------------------

def execute_approved_changes(original_query, approval_text, thread_context, call_llm_fn):
    """
    Phase 2: Apply the approved changes to Intercom.
    Uses thread context to understand what to apply (no dependency on pending_changes.json).
    Returns a Slack mrkdwn summary of what was done.
    """
    # Use LLM to extract the final approved changes from the thread
    extract_prompt = """You are reading a Slack thread where a KB update was proposed, possibly revised, and then approved.

Extract the FINAL version of the changes to apply. Look at the MOST RECENT proposal in the thread (the user may have asked for revisions — use the last revised version, not the original).

Return a JSON object:
{
  "article_updates": [
    {
      "article_title": "...",
      "article_url": "...",
      "changes_description": "full plain-text description of all changes to make to this article — include the English Before/After text for each change so a downstream LLM can apply them",
      "change_summary": [
        {"type": "UPDATE" | "ADD" | "REMOVE" | "SCREENSHOT", "section": "exact section name"}
      ]
    }
  ],
  "new_article": null or {
    "title": "...",
    "description": "...",
    "outline": "..."
  }
}

RULES:
- "change_summary" must contain ONE entry per change shown in the most recent proposal for that article, in the same order. Use the type (UPDATE / ADD / REMOVE / SCREENSHOT) and the section name shown in the proposal.
- "changes_description" is the human-readable description used by the next step to actually rewrite the HTML. Keep it complete and unambiguous.
- If the user's approval message specifies only certain changes to apply (e.g., "only article 1", "only changes 1 and 3"), include only those — and only those entries in change_summary.
- Return ONLY the JSON object, nothing else."""

    raw = call_llm_fn(
        extract_prompt,
        f"Thread conversation:\n{thread_context}\n\nApproval message:\n{approval_text}",
        model_hint="pro",
    )

    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
        pending = json.loads(cleaned)
    except (json.JSONDecodeError, AttributeError):
        return "I couldn't parse the changes from our conversation. Could you clarify which changes to apply?"

    article_updates = pending.get("article_updates", [])
    new_article_plan = pending.get("new_article", None)

    # results: list of dicts with keys kind, title, url, lang_note, items, error
    results = []

    # Fetch all articles to resolve titles to IDs
    all_articles = fetch_all_intercom_articles()
    articles_by_title = {}
    for a in all_articles:
        articles_by_title[a.get("title", "").lower()] = a

    # Apply article updates
    for update in article_updates:
        article_title = update.get("article_title", "")
        article_url = update.get("article_url", "")
        changes_description = update.get("changes_description", "")
        change_summary = update.get("change_summary", []) or []

        # Find article by title (case-insensitive)
        article = articles_by_title.get(article_title.lower())
        if not article:
            # Try partial match
            for title, a in articles_by_title.items():
                if article_title.lower() in title or title in article_title.lower():
                    article = a
                    break

        if not article:
            results.append({
                "kind": "skipped",
                "title": article_title,
                "url": article_url,
                "items": change_summary,
                "error": "could not find article in Intercom",
            })
            continue

        article_id = article["id"]
        log.info(f"Updating article {article_id}: {article_title}")

        try:
            # Fetch current article content (includes translated_content)
            resp = requests.get(
                f"https://api.intercom.io/articles/{article_id}",
                headers=INTERCOM_HEADERS,
            )
            resp.raise_for_status()
            current = resp.json()

            # --- English update ---
            current_body = current.get("body", "")
            current_title = current.get("title", "")
            current_description = current.get("description", "")

            update_prompt = """You are updating a help center article's HTML content based on approved changes.

Given the current HTML body and the specific changes to make, produce the updated HTML body.

RULES:
- Preserve the existing HTML structure and formatting exactly
- Only make the specified changes — do not rewrite unrelated sections
- Keep the same HTML tags and CSS classes
- HEADER HIERARCHY: When ADDING a new section, use the same header level as existing same-level sections in the article (e.g., if top-level sections use <h2>, the new top-level section must use <h2>). Do NOT normalize or change the header level of existing sections, even if you notice inconsistencies — only the new section being added is in scope.
- If adding new content, match the style of the existing content exactly:
  * Short, straight-to-the-point sentences. No filler or marketing language.
  * Use <ul>/<ol> lists whenever possible instead of long paragraphs.
  * Step-by-step numbered instructions for how-to content.
  * Speak directly to the user ("You can...", "Click...", "Go to...").
  * Present tense, current-state only. The KB describes how the product works today — NOT that something changed. Do NOT use "now", "will now", "has been updated", "recently", "from now on", or any wording that signals a change from a previous state. Example: write "Variables display their short label." NOT "Variables will now display their short label.". If the approved changes describe a behavior change, rewrite it as the new current behavior — do not narrate the change itself.
- Return ONLY the updated HTML body, nothing else"""

            user_prompt = (
                f"CHANGES TO APPLY:\n{changes_description}\n\n"
                f"CURRENT TITLE: {current_title}\n"
                f"CURRENT DESCRIPTION: {current_description}\n"
                f"CURRENT HTML BODY:\n{current_body}"
            )

            new_en_body = strip_code_fences(call_llm_fn(update_prompt, user_prompt, model_hint="pro"))

            # --- French update (if French version exists) ---
            translated = current.get("translated_content", {})
            fr_content = translated.get("fr", {})
            fr_body = fr_content.get("body", "")
            fr_title = fr_content.get("title", "")
            fr_description = fr_content.get("description", "")

            new_fr_body = None
            if fr_body:
                fr_update_prompt = """You are updating the FRENCH version of a help center article's HTML content.

You are given:
1. The changes that were approved (described in English)
2. The updated ENGLISH HTML body (already approved)
3. The current FRENCH HTML body

Apply the equivalent changes to the French body. The French version should convey the same information as the English version, but naturally translated — not a word-for-word translation.

RULES:
- Preserve the existing French HTML structure and formatting exactly
- Only change sections that correspond to the English changes
- Keep the same HTML tags and CSS classes
- Match the tone and style of the existing French content
- Use "vous" form (formal), consistent with the existing French articles
- Translate UI labels to their French equivalents (e.g., "Event Data" → "Données", "Registrations" → "Inscriptions", "Actions" → "Actions", "Save and continue" → "Sauvegarder et continuer")
- Présent, état actuel uniquement. La base de connaissances décrit le fonctionnement actuel du produit — pas un changement. N'utilisez PAS « désormais », « maintenant », « a été mis à jour », « récemment », « dorénavant », ni aucune formulation qui évoque un changement par rapport à un état antérieur. Exemple : écrire « Les variables affichent leur libellé court. » et NON « Les variables affichent désormais leur libellé court. ».
- Return ONLY the updated French HTML body, nothing else"""

                fr_user_prompt = (
                    f"CHANGES (English):\n{changes_description}\n\n"
                    f"UPDATED ENGLISH HTML:\n{new_en_body[:3000]}\n\n"
                    f"CURRENT FRENCH TITLE: {fr_title}\n"
                    f"CURRENT FRENCH DESCRIPTION: {fr_description}\n"
                    f"CURRENT FRENCH HTML BODY:\n{fr_body}"
                )

                new_fr_body = strip_code_fences(call_llm_fn(fr_update_prompt, fr_user_prompt, model_hint="pro"))

            # Apply the updates (both languages in one API call).
            # Intercom treats translated_content[<locale>] as a full replacement
            # for that locale, so we always send the complete FR object (title,
            # description, body, author_id) — sending only `body` causes a 400.
            if new_fr_body:
                fr_payload = {
                    "type": fr_content.get("type", "article_translated_content"),
                    "title": fr_title,
                    "description": fr_description,
                    "body": new_fr_body,
                    "author_id": fr_content.get("author_id") or INTERCOM_AUTHOR_ID,
                    "state": fr_content.get("state", "published"),
                }
                update_intercom_article(
                    article_id,
                    body=new_en_body,
                    translated_content={"fr": fr_payload},
                )
                lang_note = "EN + FR"
            else:
                update_intercom_article(article_id, body=new_en_body)
                lang_note = "EN only (no French version found)"

            url = article.get("url") or article_url
            results.append({
                "kind": "updated",
                "title": article_title,
                "url": url,
                "lang_note": lang_note,
                "items": change_summary,
            })

        except Exception as e:
            log.error(f"Failed to update article {article_id}: {e}")
            results.append({
                "kind": "failed",
                "title": article_title,
                "url": article.get("url") or article_url,
                "items": change_summary,
                "error": str(e),
            })

    # Create new article
    if new_article_plan:
        log.info(f"Creating new article: {new_article_plan.get('title', '?')}")
        try:
            create_prompt = """Create a help center article in HTML format based on this outline.

TITLE FORMAT RULES (for the article title — must follow exactly):
- Use imperative verb form (e.g., "Generate badges", "Create rooms and assign them to sessions")
- Do NOT use gerund form ("Generating...") or "How to..." prefix
- Use sentence case (only capitalize the first letter of the first word, e.g., "Generate badges")
- Keep it concise: 3-8 words

DESCRIPTION FORMAT RULES (for the short subtitle — must follow exactly):
- Start with "This article explains how to..." or "This article explains how you can..."
- Write exactly ONE sentence, ending with a period
- Target 60-120 characters

CONTENT RULES:
- Use clean, semantic HTML (h2, h3, p, ul/li, ol/li, b, etc.)
- Short, straight-to-the-point sentences. No filler or marketing language.
- Use bullet lists (<ul>) or numbered lists (<ol>) whenever possible instead of paragraphs.
- Step-by-step numbered instructions for how-to content (1. Go to... 2. Click...).
- Speak directly to the user ("You can...", "Click...", "Go to...").
- Keep the tone helpful and professional, not casual or overly friendly.
- Aimed at non-technical event organizers — avoid jargon.
- Present tense, current-state only. The article describes how the product works today — NOT that it was recently added or changed. Do NOT use "now", "will now", "has been added", "recently", "new feature", "from now on", or any wording that signals a change from a previous state. Write as if the feature has always existed.
- Add [SCREENSHOT: description] placeholders where screenshots would help.
- Return ONLY the HTML body"""

            outline = (
                f"Title: {new_article_plan.get('title', '')}\n"
                f"Description: {new_article_plan.get('description', '')}\n"
                f"Outline: {new_article_plan.get('outline', '')}"
            )

            body_html = strip_code_fences(call_llm_fn(create_prompt, outline, model_hint="pro"))

            # Generate French version
            fr_prompt = """Translate this help center article to French. You are given:
1. The English HTML body
2. The English title and description

Produce a JSON object with three fields:
{"title": "French title", "description": "French description", "body": "French HTML body"}

RULES:
- Naturally translate, not word-for-word
- Use "vous" form (formal)
- Keep the same HTML structure and tags
- Translate UI labels to their French equivalents used in the Fourwaves app
- FRENCH TITLE: Use imperative verb form in French, concise (3-8 words)
- FRENCH DESCRIPTION: Start with "Cet article explique comment..." — one sentence, period at end
- Présent, état actuel uniquement. L'article décrit le fonctionnement actuel du produit — pas un ajout récent. N'utilisez PAS « désormais », « maintenant », « a été ajouté », « récemment », « nouvelle fonctionnalité », « dorénavant », ni aucune formulation qui évoque un changement par rapport à un état antérieur. Rédigez comme si la fonctionnalité avait toujours existé.
- Return ONLY the JSON object"""

            fr_raw = call_llm_fn(
                fr_prompt,
                f"ENGLISH TITLE: {new_article_plan.get('title', '')}\n"
                f"ENGLISH DESCRIPTION: {new_article_plan.get('description', '')}\n"
                f"ENGLISH HTML BODY:\n{body_html}",
                model_hint="pro",
            )

            fr_data = None
            try:
                fr_cleaned = strip_code_fences(fr_raw)
                fr_data = json.loads(fr_cleaned)
            except (json.JSONDecodeError, AttributeError):
                log.warning("Failed to generate French translation for new article")

            translated_content = None
            if fr_data:
                translated_content = {
                    "fr": {
                        "title": fr_data.get("title", ""),
                        "description": fr_data.get("description", ""),
                        "body": fr_data.get("body", ""),
                        "author_id": INTERCOM_AUTHOR_ID,
                    }
                }

            result = create_intercom_article(
                title=new_article_plan["title"],
                body=body_html,
                description=new_article_plan.get("description", ""),
                state="published",
                translated_content=translated_content,
            )
            new_url = result.get("url", "")
            lang_note = "EN + FR" if fr_data else "EN only"
            results.append({
                "kind": "created",
                "title": new_article_plan["title"],
                "url": new_url,
                "lang_note": lang_note,
                "items": [],
            })

        except Exception as e:
            log.error(f"Failed to create new article: {e}")
            results.append({
                "kind": "failed_create",
                "title": new_article_plan.get("title", "(new article)"),
                "url": "",
                "items": [],
                "error": str(e),
            })

    return format_apply_report(results)


def format_apply_report(results):
    """Render the post-apply git-diff-style report as Slack mrkdwn."""
    if not results:
        return "No changes were applied."

    updated = [r for r in results if r["kind"] == "updated"]
    created = [r for r in results if r["kind"] == "created"]
    skipped = [r for r in results if r["kind"] == "skipped"]
    failed = [r for r in results if r["kind"] in ("failed", "failed_create")]

    parts = ["*Changes applied:*\n"]

    def render_item_bullets(items):
        lines = []
        for ch in items or []:
            typ = str(ch.get("type", "UPDATE")).upper()
            sec = (ch.get("section") or "").strip()
            if sec:
                lines.append(f'    • [{typ}] {sec}')
            else:
                lines.append(f'    • [{typ}]')
        return lines

    if updated:
        parts.append(f"*Articles updated ({len(updated)}):*")
        for r in updated:
            parts.append(f"*{r['title']}* — updated ({r.get('lang_note', '')})")
            if r.get("url"):
                parts.append(f"    {r['url']}")
            parts.extend(render_item_bullets(r.get("items")))
            parts.append("")

    if created:
        parts.append(f"*New articles created and published ({len(created)}):*")
        for r in created:
            parts.append(f"*{r['title']}* — created and published ({r.get('lang_note', '')})")
            if r.get("url"):
                parts.append(f"    {r['url']}")
            parts.append("")

    if skipped:
        parts.append(f"*Skipped ({len(skipped)}):*")
        for r in skipped:
            parts.append(f"*{r['title']}* — skipped: {r.get('error', '')}")
        parts.append("")

    if failed:
        parts.append(f"*Failed ({len(failed)}):*")
        for r in failed:
            parts.append(f"*{r['title']}* — failed: {r.get('error', '')}")
        parts.append("")

    parts.append("All updates are live in Intercom.")
    return "\n".join(parts).rstrip()
