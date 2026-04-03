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
    resp.raise_for_status()
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
    resp.raise_for_status()
    return resp.json()


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

Return ONLY a JSON array of ID numbers. Example: [123, 456]
If none are relevant, return: []"""

        user_prompt = f"RELEASE SUMMARY:\n{release_summary}\n\nARTICLES TO EVALUATE:\n{batch_text}"

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

Return a JSON object with: {"title": "...", "description": "...", "outline": "..."}
The outline should be a bullet-pointed structure of what the article should cover.""",
            f"Release:\n{release_summary}",
            model_hint="pro",
        )
        return (
            f"I scanned all {len(published)} published help center articles and none seem directly affected by this release.\n\n"
            f"However, I recommend *creating a new article*:\n\n{new_article_proposal}\n\n"
            f"Reply with *yes, proceed* to create the article as a draft, or tell me what to adjust."
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

CRITICAL: Only propose changes that are DIRECTLY related to this article's topic. If the release affects a different feature than what this article covers, return NO_CHANGES. Do NOT propose adding content about tangentially related features.

HEADER HIERARCHY RULES (very important):
- Look at the existing HTML headers in the article (h1, h2, h3, etc.).
- When adding new sections, MATCH the header level used by existing same-level sections. For example, if existing top-level sections use <h2>, new top-level sections must also use <h2>.
- If you detect INCONSISTENT header levels (e.g., some top-level sections use <h1> and others use <h2>), mention that you will also normalize headers to be consistent. The standard is: top-level sections = <h2>, sub-sections = <h3>.

WRITING STYLE for any new/edited text (must match existing articles):
- Short, straight-to-the-point sentences. No filler or marketing language.
- Use bullet lists or numbered lists whenever possible instead of paragraphs.
- Step-by-step instructions for how-to content (1. Go to... 2. Click... 3. Select...).
- Speak directly to the user ("You can...", "Click...", "Go to...").
- Keep the tone helpful and professional, not casual or overly friendly.

If NO changes are needed for this article, return exactly: NO_CHANGES

OUTPUT FORMAT (strict Slack mrkdwn — this will be posted in Slack):
For each change, use this exact format:

*[UPDATE/ADD/REMOVE/SCREENSHOT]* — Section: "section name"
Why: one sentence explaining why
```Before:
exact current text that will be changed```
```After:
exact new text that will replace it```

For ADD changes (new content), only show an "After" block.
For REMOVE changes (deleted content), only show a "Before" block.
For SCREENSHOT changes, just describe what screenshot to add/update.

IMPORTANT:
- Use single * for bold. NEVER use ** or ## or ### or markdown headers.
- The Before/After blocks MUST contain the actual text, not a description of it.
- Use ``` for code blocks (triple backtick). These render as distinct blocks in Slack.
- Number each change (1. 2. 3.)"""

        user_prompt = f"RELEASE SUMMARY:\n{release_summary}\n\n{article_detail}"

        try:
            proposal = call_llm_fn(proposal_prompt, user_prompt, model_hint="pro")
            if "NO_CHANGES" not in proposal:
                detailed_proposals.append({
                    "article_id": str(article["id"]),
                    "article_title": article["title"],
                    "article_url": article.get("url", ""),
                    "changes": proposal,
                })
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

If a new article is needed, return a JSON object:
{"needed": true, "title": "...", "description": "...", "outline": "full draft of the article content in plain text with section headers and bullet points — this will be shown in a Slack code block for review"}

If NOT needed, return: {"needed": false}"""

    existing_titles = "\n".join(f"- {p['article_title']}" for p in detailed_proposals)
    new_article_raw = call_llm_fn(
        new_article_prompt,
        f"Release:\n{release_summary}\n\nArticles being updated:\n{existing_titles}",
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

    # 8. Format the Slack response
    return format_proposal_message(detailed_proposals, new_article_plan, len(published), notion_pages, release_summary)


def format_proposal_message(proposals, new_article_plan, total_articles, notion_pages, release_summary):
    """Format the change proposal as a Slack mrkdwn message."""
    parts = []

    # Feature summary from QA Notes — serves as validation
    parts.append("*FEATURE SUMMARY (from QA Notes):*\n")
    parts.append(release_summary)

    feature_names = ", ".join(p["title"] for p in notion_pages)
    parts.append(f"\n_Scanned {total_articles} published help center articles for: {feature_names}_\n")

    if proposals:
        parts.append(f"*ARTICLES TO UPDATE ({len(proposals)}):*\n")

        for i, p in enumerate(proposals, 1):
            parts.append(f"*{i}. {p['article_title']}*")
            parts.append(f"    {p['article_url']}")
            parts.append(p["changes"])
            parts.append("")

    if new_article_plan:
        parts.append("*NEW ARTICLE RECOMMENDED:*\n")
        parts.append(f"*Title:* {new_article_plan.get('title', 'TBD')}")
        parts.append(f"*Description:* {new_article_plan.get('description', '')}\n")
        outline = new_article_plan.get("outline", "")
        parts.append(f"```{outline}```")
        parts.append("")

    if not proposals and not new_article_plan:
        parts.append("After detailed analysis, no changes are needed for any existing articles and no new article is recommended.")
        return "\n".join(parts)

    parts.append("\n_Note: When approved, changes will be applied to both English and French versions of each article._")
    parts.append("\n---")
    parts.append("You can ask me to revise the proposal, or reply *yes, proceed* to apply the changes.")

    return "\n".join(parts)


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
- Keep the same format as the original proposal (article titles, Before/After blocks, etc.)
- If the user asked to remove a change, remove it entirely.
- If the user corrected factual details, apply those corrections to the Before/After text.
- Show the full revised proposal, not just what changed.

HEADER HIERARCHY RULES:
- When proposing new sections, match the header level used by existing same-level sections in the article.
- If you notice inconsistent headers (e.g., mix of H1 and H2 for top-level sections), mention that you will normalize them: top-level = H2, sub-sections = H3.

WRITING STYLE for any new/edited text (must match existing Fourwaves help center articles):
- Short, straight-to-the-point sentences. No filler or marketing language.
- Use bullet lists or numbered lists whenever possible instead of paragraphs.
- Step-by-step instructions for how-to content (1. Go to... 2. Click... 3. Select...).
- Speak directly to the user ("You can...", "Click...", "Go to...").
- Keep the tone helpful and professional, not casual or overly friendly.

OUTPUT FORMAT (strict Slack mrkdwn — this will be posted in Slack):
- Use single * for bold (*bold*). NEVER use ** or ## or ### or markdown headers.
- Use ``` for code blocks (triple backtick) for Before/After text blocks.
- Number each change (1. 2. 3.)
- For each change use: *[UPDATE/ADD/REMOVE/SCREENSHOT]* — Section: "section name"
  Why: one sentence
  ```Before: ...``` and ```After: ...```

End with:
---
You can ask me to revise again, or reply *yes, proceed* to apply the changes."""

    response = call_llm_fn(
        revision_prompt,
        f"Full thread conversation:\n{thread_context}\n\nUser's latest revision request:\n{revision_request}",
        model_hint="pro",
    )
    return response


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
      "changes_description": "full description of all changes to make to this article"
    }
  ],
  "new_article": null or {
    "title": "...",
    "description": "...",
    "outline": "..."
  }
}

If the user's approval message specifies only certain changes to apply (e.g., "only article 1"), include only those.
Return ONLY the JSON object, nothing else."""

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

        # Find article by title (case-insensitive)
        article = articles_by_title.get(article_title.lower())
        if not article:
            # Try partial match
            for title, a in articles_by_title.items():
                if article_title.lower() in title or title in article_title.lower():
                    article = a
                    break

        if not article:
            results.append(f"SKIPPED: *{article_title}* — could not find article in Intercom")
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
- HEADER HIERARCHY: When adding new sections, use the same header level as existing same-level sections. If top-level sections use <h2>, new top-level sections must use <h2>. If you see inconsistent headers (mix of <h1> and <h2> for same-level sections), normalize them: top-level = <h2>, sub-sections = <h3>.
- If adding new content, match the style of the existing content exactly:
  * Short, straight-to-the-point sentences. No filler or marketing language.
  * Use <ul>/<ol> lists whenever possible instead of long paragraphs.
  * Step-by-step numbered instructions for how-to content.
  * Speak directly to the user ("You can...", "Click...", "Go to...").
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
- Return ONLY the updated French HTML body, nothing else"""

                fr_user_prompt = (
                    f"CHANGES (English):\n{changes_description}\n\n"
                    f"UPDATED ENGLISH HTML:\n{new_en_body[:3000]}\n\n"
                    f"CURRENT FRENCH TITLE: {fr_title}\n"
                    f"CURRENT FRENCH DESCRIPTION: {fr_description}\n"
                    f"CURRENT FRENCH HTML BODY:\n{fr_body}"
                )

                new_fr_body = strip_code_fences(call_llm_fn(fr_update_prompt, fr_user_prompt, model_hint="pro"))

            # Apply the updates (both languages in one API call)
            if new_fr_body:
                update_intercom_article(
                    article_id,
                    body=new_en_body,
                    translated_content={"fr": {"body": new_fr_body}},
                )
                lang_note = " (EN + FR)"
            else:
                update_intercom_article(article_id, body=new_en_body)
                lang_note = " (EN only — no French version found)"

            url = article.get("url", article_url)
            results.append(f"Updated: *{article_title}*{lang_note}\n   {url}")

        except Exception as e:
            log.error(f"Failed to update article {article_id}: {e}")
            results.append(f"FAILED: *{article_title}* — {e}")

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
                state="draft",
                translated_content=translated_content,
            )
            new_url = result.get("url", "")
            lang_note = " (EN + FR)" if fr_data else " (EN only)"
            results.append(f"Created (as draft): *{new_article_plan['title']}*{lang_note}\n   {new_url}")

        except Exception as e:
            log.error(f"Failed to create new article: {e}")
            results.append(f"FAILED to create new article: {e}")

    # Format response
    if not results:
        return "No changes were applied."

    summary = "*Changes applied:*\n\n" + "\n\n".join(results)
    summary += "\n\nAll updates are live. New articles are saved as drafts — publish them when ready."
    return summary
