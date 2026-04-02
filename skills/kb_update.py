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
    """Fetch a Notion page's properties and block content."""
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

    # Get block children (the actual content)
    blocks_text = fetch_notion_blocks(page_id)

    return {"id": page_id, "title": title, "content": blocks_text}


def fetch_notion_blocks(block_id, depth=0):
    """Recursively fetch all block content from a Notion page."""
    if depth > 5:
        return ""

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

    parts = []
    for block in all_blocks:
        block_type = block.get("type", "")
        block_data = block.get(block_type, {})

        # Extract text from rich_text arrays
        rich_text = block_data.get("rich_text", [])
        text = " ".join(t.get("plain_text", "") for t in rich_text)

        if block_type.startswith("heading"):
            level = block_type[-1]  # heading_1 → 1
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
            child_text = fetch_notion_blocks(block["id"], depth + 1)
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


def update_intercom_article(article_id, title=None, body=None, description=None):
    """Update an Intercom article."""
    payload = {}
    if title:
        payload["title"] = title
    if body:
        payload["body"] = body
    if description:
        payload["description"] = description

    resp = requests.put(
        f"https://api.intercom.io/articles/{article_id}",
        headers=INTERCOM_HEADERS,
        json=payload,
    )
    resp.raise_for_status()
    return resp.json()


def create_intercom_article(title, body, description="", parent_id=None, parent_type=None, state="draft"):
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

CHANGES_CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pending_changes.json")


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

    # 3. Understand the release scope
    release_summary_prompt = """Analyze these product release/feature pages and create a clear summary of:
1. What new features or changes were released
2. What functionality they add or modify
3. Key details a technical writer would need to update documentation

Be specific and thorough. This summary will be used to identify which help center articles need updating."""

    pages_content = "\n\n===\n\n".join(
        f"PAGE: {p['title']}\n\n{p['content']}" for p in notion_pages
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
- It covers a feature or workflow affected by the release
- It describes functionality that has changed
- It could benefit from mentioning the new feature
- It's closely related to the topic area of the release

Be INCLUSIVE — if an article might be relevant, include it. We'll do detailed analysis next.

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
        article_detail = (
            f"ARTICLE: {article['title']}\n"
            f"ID: {article['id']}\n"
            f"URL: {article.get('url', 'N/A')}\n"
            f"Description: {article.get('description', '')}\n"
            f"Current content:\n{body_text[:3000]}"
        )

        proposal_prompt = """You are a technical writer updating help center documentation after a product release.

Given the release details and an existing help center article, determine what specific changes should be made.

For each change, specify:
1. The type: UPDATE (modify existing content), ADD (add a new section), REMOVE (delete outdated content), or SCREENSHOT (recommend adding/updating a screenshot)
2. The specific location in the article (section title or paragraph reference)
3. What exactly should be changed — be specific about what text to add, modify, or remove
4. Why this change is needed

If NO changes are needed for this article, return exactly: NO_CHANGES

Format your response as a numbered list of changes. Be specific and actionable."""

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

If a new article is needed, return a JSON object:
{"needed": true, "title": "...", "description": "...", "outline": "bullet-pointed structure"}

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

    # 8. Save pending changes for execution phase
    pending = {
        "notion_pages": [{"id": p["id"], "title": p["title"]} for p in notion_pages],
        "release_summary": release_summary,
        "article_updates": detailed_proposals,
        "new_article": new_article_plan,
    }
    with open(CHANGES_CACHE_FILE, "w") as f:
        json.dump(pending, f, indent=2)

    # 9. Format the Slack response
    return format_proposal_message(detailed_proposals, new_article_plan, len(published), notion_pages)


def format_proposal_message(proposals, new_article_plan, total_articles, notion_pages):
    """Format the change proposal as a Slack mrkdwn message."""
    parts = []

    feature_names = ", ".join(p["title"] for p in notion_pages)
    parts.append(f"I've analyzed the release ({feature_names}) against all {total_articles} published help center articles.\n")

    if proposals:
        parts.append(f"*ARTICLES TO UPDATE ({len(proposals)}):*\n")

        for i, p in enumerate(proposals, 1):
            parts.append(f"*{i}. {p['article_title']}*")
            parts.append(f"   {p['article_url']}\n")
            parts.append(p["changes"])
            parts.append("")  # blank line

    if new_article_plan:
        parts.append("\n*NEW ARTICLE RECOMMENDED:*\n")
        parts.append(f"*Title:* {new_article_plan.get('title', 'TBD')}")
        parts.append(f"*Description:* {new_article_plan.get('description', '')}")
        parts.append(f"*Outline:*\n{new_article_plan.get('outline', '')}")
        parts.append("")

    if not proposals and not new_article_plan:
        parts.append("After detailed analysis, no changes are needed for any existing articles and no new article is recommended.")
        return "\n".join(parts)

    parts.append("\n---")
    parts.append("Reply with *yes, proceed with all changes* to apply everything.")
    parts.append("Or specify which changes to apply (e.g., *only update articles 1 and 3* or *skip the new article*).")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Skill handler: execute approved changes
# ---------------------------------------------------------------------------

def execute_approved_changes(original_query, approval_text, classification, thread_context, call_llm_fn):
    """
    Phase 2: Apply the approved changes to Intercom.
    Returns a Slack mrkdwn summary of what was done.
    """
    # Load pending changes
    if not os.path.exists(CHANGES_CACHE_FILE):
        return "I can't find the pending changes. Please start over by posting the Notion links again."

    with open(CHANGES_CACHE_FILE, "r") as f:
        pending = json.load(f)

    article_updates = pending.get("article_updates", [])
    new_article_plan = pending.get("new_article", None)
    release_summary = pending.get("release_summary", "")

    # Determine which changes to apply
    apply_updates = []
    apply_new_article = False

    if classification == "approve_all":
        apply_updates = list(range(len(article_updates)))
        apply_new_article = new_article_plan is not None
    else:
        # Partial approval — use LLM to figure out which ones
        filter_prompt = """The user partially approved a set of proposed knowledge base changes.
Given the original proposals and the user's response, determine which changes to apply.

Return a JSON object:
{
  "update_indices": [0, 2],  // 0-based indices of article updates to apply
  "create_new_article": true/false
}"""
        proposals_summary = "\n".join(
            f"{i}. {p['article_title']}" for i, p in enumerate(article_updates)
        )
        if new_article_plan:
            proposals_summary += f"\n{len(article_updates)}. NEW ARTICLE: {new_article_plan.get('title', '')}"

        raw = call_llm_fn(
            filter_prompt,
            f"Proposals:\n{proposals_summary}\n\nUser's response: {approval_text}",
            model_hint="flash",
        )

        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
            parsed = json.loads(cleaned)
            apply_updates = parsed.get("update_indices", [])
            apply_new_article = parsed.get("create_new_article", False)
        except (json.JSONDecodeError, AttributeError):
            return "I couldn't understand which changes to apply. Please be more specific (e.g., 'only articles 1 and 3')."

    results = []

    # Apply article updates
    for idx in apply_updates:
        if idx >= len(article_updates):
            continue
        update = article_updates[idx]
        article_id = update["article_id"]
        article_title = update["article_title"]
        changes_description = update["changes"]

        log.info(f"Updating article {article_id}: {article_title}")

        try:
            # Fetch current article content
            resp = requests.get(
                f"https://api.intercom.io/articles/{article_id}",
                headers=INTERCOM_HEADERS,
            )
            resp.raise_for_status()
            current = resp.json()
            current_body = current.get("body", "")
            current_title = current.get("title", "")
            current_description = current.get("description", "")

            # Generate updated HTML body
            update_prompt = """You are updating a help center article's HTML content based on approved changes.

Given the current HTML body and the specific changes to make, produce the updated HTML body.

RULES:
- Preserve the existing HTML structure and formatting exactly
- Only make the specified changes — do not rewrite unrelated sections
- Keep the same HTML tags and CSS classes
- If adding new content, match the style of the existing content
- Return ONLY the updated HTML body, nothing else"""

            user_prompt = (
                f"CHANGES TO APPLY:\n{changes_description}\n\n"
                f"RELEASE CONTEXT:\n{release_summary[:1000]}\n\n"
                f"CURRENT TITLE: {current_title}\n"
                f"CURRENT DESCRIPTION: {current_description}\n"
                f"CURRENT HTML BODY:\n{current_body}"
            )

            new_body = call_llm_fn(update_prompt, user_prompt, model_hint="pro")

            # Clean up LLM output (remove markdown code fences if present)
            if new_body.startswith("```"):
                new_body = new_body.split("\n", 1)[1]
                if new_body.endswith("```"):
                    new_body = new_body[:-3]
                new_body = new_body.strip()

            # Apply the update
            update_intercom_article(article_id, body=new_body)
            results.append(f"Updated: *{article_title}*\n   {update.get('article_url', '')}")

        except Exception as e:
            log.error(f"Failed to update article {article_id}: {e}")
            results.append(f"FAILED: *{article_title}* — {e}")

    # Create new article
    if apply_new_article and new_article_plan:
        log.info(f"Creating new article: {new_article_plan.get('title', '?')}")
        try:
            create_prompt = """Create a help center article in HTML format based on this outline.

RULES:
- Use clean, semantic HTML (h2, h3, p, ul/li, ol/li, b, etc.)
- Write clear, concise documentation aimed at non-technical event organizers
- Include practical step-by-step instructions where appropriate
- Add [SCREENSHOT: description] placeholders where screenshots would help
- Match the tone of a friendly SaaS help center
- Return ONLY the HTML body"""

            outline = (
                f"Title: {new_article_plan.get('title', '')}\n"
                f"Description: {new_article_plan.get('description', '')}\n"
                f"Outline: {new_article_plan.get('outline', '')}\n\n"
                f"Release context:\n{release_summary[:2000]}"
            )

            body_html = call_llm_fn(create_prompt, outline, model_hint="pro")

            if body_html.startswith("```"):
                body_html = body_html.split("\n", 1)[1]
                if body_html.endswith("```"):
                    body_html = body_html[:-3]
                body_html = body_html.strip()

            result = create_intercom_article(
                title=new_article_plan["title"],
                body=body_html,
                description=new_article_plan.get("description", ""),
                state="draft",
            )
            new_url = result.get("url", "")
            results.append(f"Created (as draft): *{new_article_plan['title']}*\n   {new_url}")

        except Exception as e:
            log.error(f"Failed to create new article: {e}")
            results.append(f"FAILED to create new article: {e}")

    # Clean up
    try:
        os.remove(CHANGES_CACHE_FILE)
    except OSError:
        pass

    # Format response
    if not results:
        return "No changes were applied."

    summary = "*Changes applied:*\n\n" + "\n\n".join(results)
    summary += "\n\nAll updates are live. New articles are saved as drafts — publish them when ready."
    return summary
