"""
Product Brain Update Skill — Update the Notion product brain database from
Notion product release pages.

Flow:
  1. Extract Notion page URLs from the Slack message (the released feature pages)
  2. Fetch each released page's QA Notes content
  3. Build a release summary
  4. Fetch ALL cards from the product brain database (paginated)
  5. LLM scores which cards are relevant to the release
  6. For each relevant card: LLM proposes specific UPDATE/ADD/REMOVE changes
  7. LLM proposes a new card if a brand-new feature isn't covered anywhere
  8. Format the proposal as Slack mrkdwn (one message per card)
  9. On approval, regenerate each card's body and create new pages in the DB

The product brain is English-only (no FR translation logic).

Properties on new pages:
  - Name      → required (title)
  - Module    → derived by the LLM from existing values in the DB
  - Sub-module → derived by the LLM from existing values in the DB
  - Status    → always "Published" on creation (existing pages keep their status)
"""

import os
import re
import json
import logging
from collections import Counter

import requests

# Reuse generic Notion helpers from kb_update — they handle pagination,
# block rendering, URL parsing, code-fence stripping the same way the rest
# of this codebase does, so duplicating them would just create drift.
from skills.kb_update import (
    NOTION_HEADERS,
    fetch_child_blocks,
    render_blocks,
    strip_code_fences,
    notion_url_to_page_id,
    extract_notion_urls,
    fetch_notion_blocks_qa_only,
)

log = logging.getLogger("oracle.product_brain_update")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PRODUCT_BRAIN_DB = "2b98b055-517b-80ef-9733-d05ed8018f61"

# Logical names — looked up case-insensitively against the DB schema at runtime
# so the code keeps working if a property gets renamed in Notion.
TITLE_PROPERTY = "Name"
MODULE_PROPERTY = "Module"
SUBMODULE_PROPERTY = "Sub-module"
STATUS_PROPERTY = "Status"
PUBLISHED_STATUS = "Published"


# ---------------------------------------------------------------------------
# Property helpers
# ---------------------------------------------------------------------------

def _extract_property_text(prop):
    """Render a single page property to a plain-text value for prompts."""
    if not prop:
        return ""
    t = prop.get("type", "")
    if t == "title":
        return " ".join(x.get("plain_text", "") for x in prop.get("title", []))
    if t == "rich_text":
        return " ".join(x.get("plain_text", "") for x in prop.get("rich_text", []))
    if t == "select":
        sel = prop.get("select")
        return sel.get("name", "") if sel else ""
    if t == "multi_select":
        return ", ".join(o.get("name", "") for o in prop.get("multi_select", []) or [])
    if t == "status":
        st = prop.get("status")
        return st.get("name", "") if st else ""
    if t == "people":
        return ", ".join(p.get("name", "") for p in prop.get("people", []) or [])
    if t == "date":
        d = prop.get("date")
        return (d.get("start", "") if d else "") or ""
    if t == "checkbox":
        return "true" if prop.get("checkbox") else "false"
    if t == "number":
        n = prop.get("number")
        return "" if n is None else str(n)
    if t == "url":
        return prop.get("url", "") or ""
    return ""


def _page_property(page, name):
    """Look up a page property case-insensitively. Returns the property dict or None."""
    target = (name or "").strip().lower()
    for k, v in (page.get("properties") or {}).items():
        if k.strip().lower() == target:
            return v
    return None


def _page_text(page, name):
    return _extract_property_text(_page_property(page, name))


def _schema_property(schema, name):
    """Look up a property in a database schema, case-insensitively."""
    target = (name or "").strip().lower()
    for k, v in (schema.get("properties") or {}).items():
        if k.strip().lower() == target:
            v = dict(v)
            v["_actual_name"] = k
            return v
    return None


def _build_property_value(prop_schema, value):
    """Build a Notion API property value matching the schema's type.

    Handles select / multi_select / status / rich_text / title. Notion accepts
    new select option names inline (it auto-creates them), so we don't need to
    PATCH the schema first to use a brand-new module/sub-module value.
    """
    if not prop_schema or value in (None, ""):
        return None
    t = prop_schema.get("type")
    if t == "title":
        return {"title": [{"type": "text", "text": {"content": str(value)}}]}
    if t == "rich_text":
        return {"rich_text": [{"type": "text", "text": {"content": str(value)}}]}
    if t == "select":
        return {"select": {"name": str(value)}}
    if t == "multi_select":
        if isinstance(value, list):
            return {"multi_select": [{"name": str(v)} for v in value if v]}
        return {"multi_select": [{"name": str(value)}]}
    if t == "status":
        return {"status": {"name": str(value)}}
    if t == "checkbox":
        return {"checkbox": bool(value)}
    if t == "number":
        try:
            return {"number": float(value)}
        except (TypeError, ValueError):
            return None
    if t == "url":
        return {"url": str(value)}
    return None


# ---------------------------------------------------------------------------
# Notion: database schema + page fetching
# ---------------------------------------------------------------------------

def fetch_database_schema(database_id):
    """Fetch the database schema (property definitions and select options)."""
    resp = requests.get(
        f"https://api.notion.com/v1/databases/{database_id}",
        headers=NOTION_HEADERS,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_all_brain_pages(database_id):
    """Query the product brain DB and return ALL pages (paginated)."""
    url = f"https://api.notion.com/v1/databases/{database_id}/query"
    pages = []
    has_more = True
    cursor = None
    body = {"page_size": 100}
    while has_more:
        if cursor:
            body["start_cursor"] = cursor
        elif "start_cursor" in body:
            del body["start_cursor"]
        resp = requests.post(url, headers=NOTION_HEADERS, json=body)
        resp.raise_for_status()
        data = resp.json()
        pages.extend(data.get("results", []))
        has_more = data.get("has_more", False)
        cursor = data.get("next_cursor")
    return pages


def fetch_page_full_content(page_id):
    """Fetch all top-level blocks of a page rendered as plain text."""
    blocks = fetch_child_blocks(page_id)
    return render_blocks(blocks)


def notion_page_url(page_id):
    return f"https://www.notion.so/{page_id.replace('-', '')}"


def summarize_brain_card(page):
    """Build a compact dict view of a brain card for prompts and matching."""
    return {
        "id": page["id"],
        "title": _page_text(page, TITLE_PROPERTY),
        "module": _page_text(page, MODULE_PROPERTY),
        "submodule": _page_text(page, SUBMODULE_PROPERTY),
        "status": _page_text(page, STATUS_PROPERTY),
        "url": notion_page_url(page["id"]),
    }


def collect_property_value_usage(pages, property_name):
    """Return a Counter of values used for a given property across pages.

    For multi_select properties, each option counts independently. Empty values
    are skipped.
    """
    counts = Counter()
    for page in pages:
        prop = _page_property(page, property_name)
        if not prop:
            continue
        t = prop.get("type")
        if t == "multi_select":
            for o in prop.get("multi_select") or []:
                name = (o.get("name") or "").strip()
                if name:
                    counts[name] += 1
        else:
            val = (_extract_property_text(prop) or "").strip()
            if val:
                counts[val] += 1
    return counts


# ---------------------------------------------------------------------------
# Notion: applying changes
# ---------------------------------------------------------------------------

def _markdown_line_to_block(line):
    """Convert a single markdown-ish line into a Notion block dict."""
    stripped = line.rstrip()
    if not stripped:
        return None

    # Headings
    m = re.match(r"^(#{1,3})\s+(.*)$", stripped)
    if m:
        level = len(m.group(1))
        text = m.group(2).strip()
        block_type = f"heading_{level}"
        return {
            "object": "block",
            "type": block_type,
            block_type: {"rich_text": [{"type": "text", "text": {"content": text}}]},
        }

    # Numbered list
    m = re.match(r"^\s*\d+\.\s+(.*)$", stripped)
    if m:
        text = m.group(1).strip()
        return {
            "object": "block",
            "type": "numbered_list_item",
            "numbered_list_item": {
                "rich_text": [{"type": "text", "text": {"content": text}}]
            },
        }

    # Bulleted list (allow -, *, +)
    m = re.match(r"^\s*[-*+]\s+(.*)$", stripped)
    if m:
        text = m.group(1).strip()
        return {
            "object": "block",
            "type": "bulleted_list_item",
            "bulleted_list_item": {
                "rich_text": [{"type": "text", "text": {"content": text}}]
            },
        }

    # Quote
    if stripped.startswith("> "):
        return {
            "object": "block",
            "type": "quote",
            "quote": {
                "rich_text": [{"type": "text", "text": {"content": stripped[2:].strip()}}]
            },
        }

    # Default: paragraph
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [{"type": "text", "text": {"content": stripped}}]
        },
    }


def markdown_to_notion_blocks(text):
    """Convert simple markdown (headings, lists, paragraphs) into Notion blocks.

    Notion's API caps rich_text content at 2000 chars per block, so paragraph
    blocks longer than that are split.
    """
    blocks = []
    for raw_line in (text or "").splitlines():
        block = _markdown_line_to_block(raw_line)
        if not block:
            continue
        # Split long content per Notion's 2000-char limit per rich_text item.
        block_type = block["type"]
        rt = block[block_type]["rich_text"]
        if rt and len(rt[0]["text"]["content"]) > 1900:
            content = rt[0]["text"]["content"]
            chunks = [content[i:i + 1900] for i in range(0, len(content), 1900)]
            for chunk in chunks:
                clone = json.loads(json.dumps(block))
                clone[block_type]["rich_text"] = [
                    {"type": "text", "text": {"content": chunk}}
                ]
                blocks.append(clone)
        else:
            blocks.append(block)
    return blocks


def replace_page_body(page_id, markdown_text):
    """Delete all children blocks of a page and append fresh ones from markdown.

    Notion has no atomic 'replace children' operation, so this is delete-then-
    append. If a delete fails partway through we still try to append, leaving
    the page in a recoverable (if a bit messy) state rather than empty.
    """
    # Fetch current top-level blocks
    existing = fetch_child_blocks(page_id)
    for block in existing:
        bid = block.get("id")
        if not bid:
            continue
        try:
            resp = requests.delete(
                f"https://api.notion.com/v1/blocks/{bid}",
                headers=NOTION_HEADERS,
            )
            resp.raise_for_status()
        except Exception as e:
            log.warning(f"Failed to delete block {bid} on page {page_id}: {e}")

    # Append new blocks (Notion caps at 100 children per call)
    new_blocks = markdown_to_notion_blocks(markdown_text)
    for i in range(0, len(new_blocks), 100):
        batch = new_blocks[i:i + 100]
        resp = requests.patch(
            f"https://api.notion.com/v1/blocks/{page_id}/children",
            headers=NOTION_HEADERS,
            json={"children": batch},
        )
        resp.raise_for_status()


def create_brain_page(database_id, db_schema, title, module, submodule, body_markdown):
    """Create a new page in the product brain DB with given properties and body.

    Status is forced to PUBLISHED_STATUS for new pages (per user spec).
    """
    properties = {}

    title_schema = _schema_property(db_schema, TITLE_PROPERTY)
    if title_schema:
        val = _build_property_value(title_schema, title)
        if val is not None:
            properties[title_schema["_actual_name"]] = val

    module_schema = _schema_property(db_schema, MODULE_PROPERTY)
    if module_schema and module:
        val = _build_property_value(module_schema, module)
        if val is not None:
            properties[module_schema["_actual_name"]] = val

    submodule_schema = _schema_property(db_schema, SUBMODULE_PROPERTY)
    if submodule_schema and submodule:
        val = _build_property_value(submodule_schema, submodule)
        if val is not None:
            properties[submodule_schema["_actual_name"]] = val

    status_schema = _schema_property(db_schema, STATUS_PROPERTY)
    if status_schema:
        val = _build_property_value(status_schema, PUBLISHED_STATUS)
        if val is not None:
            properties[status_schema["_actual_name"]] = val

    children = markdown_to_notion_blocks(body_markdown or "")
    payload = {
        "parent": {"database_id": database_id},
        "properties": properties,
        # Notion caps page-create children at 100; we trim and append the
        # remainder below if needed.
        "children": children[:100],
    }
    resp = requests.post(
        "https://api.notion.com/v1/pages",
        headers=NOTION_HEADERS,
        json=payload,
    )
    resp.raise_for_status()
    page = resp.json()

    if len(children) > 100:
        page_id = page["id"]
        for i in range(100, len(children), 100):
            batch = children[i:i + 100]
            r = requests.patch(
                f"https://api.notion.com/v1/blocks/{page_id}/children",
                headers=NOTION_HEADERS,
                json={"children": batch},
            )
            r.raise_for_status()

    return page


# ---------------------------------------------------------------------------
# Slack mrkdwn rendering
# ---------------------------------------------------------------------------

def _sanitize_block_content(text):
    """Strip literal triple-backticks from text destined for a Slack code block."""
    if not text:
        return text
    return text.replace("```", "'''")


def render_changes_as_mrkdwn(changes_list):
    """Render a list of change dicts (UPDATE/ADD/REMOVE) as Slack mrkdwn."""
    if not changes_list:
        return ""

    parts = []
    for i, change in enumerate(changes_list, 1):
        type_ = str(change.get("type", "UPDATE")).upper()
        section = change.get("section", "")
        why = change.get("why", "")

        header = f"{i}. *[{type_}]*"
        if section:
            header += f' — Section: "{section}"'
        parts.append(header)
        if why:
            parts.append(f"Why: {why}")

        before = _sanitize_block_content((change.get("before") or "").strip())
        after = _sanitize_block_content((change.get("after") or "").strip())

        if before:
            parts.append("")
            parts.append("*Before:*")
            parts.append("```")
            parts.append(before)
            parts.append("```")
        if after:
            parts.append("")
            parts.append("*After:*")
            parts.append("```")
            parts.append(after)
            parts.append("```")
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def _render_card_message(index, proposal):
    """Build a single Slack message for one card's proposed changes."""
    lines = [f"*{index}. {proposal['card_title']}*"]
    if proposal.get("card_url"):
        lines.append(f"    {proposal['card_url']}")
    meta_bits = []
    if proposal.get("module"):
        meta_bits.append(f"Module: {proposal['module']}")
    if proposal.get("submodule"):
        meta_bits.append(f"Sub-module: {proposal['submodule']}")
    if meta_bits:
        lines.append(f"    _{' • '.join(meta_bits)}_")
    lines.append(proposal["changes"])
    return "\n".join(lines).rstrip()


def _render_new_card_message(new_card_plan):
    """Build a single Slack message for a recommended new product brain card."""
    lines = ["*NEW CARD RECOMMENDED:*\n"]
    lines.append(f"*Title:* {new_card_plan.get('title', 'TBD')}")
    if new_card_plan.get("module"):
        lines.append(f"*Module:* {new_card_plan['module']}")
    if new_card_plan.get("submodule"):
        lines.append(f"*Sub-module:* {new_card_plan['submodule']}")
    lines.append(f"*Status:* {PUBLISHED_STATUS} _(applied automatically on creation)_")
    lines.append("")
    body = _sanitize_block_content((new_card_plan.get("body") or "").strip())
    if body:
        lines.append("*Body:*")
        lines.append("```")
        lines.append(body)
        lines.append("```")
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Phase 1: propose changes
# ---------------------------------------------------------------------------

def handle_product_brain_update(message_text, call_llm_fn):
    """Analyze the released feature against the product brain and propose updates.

    Returns a list[str] — one Slack message per element so the parent thread
    can render each card cleanly without code fences spanning Slack messages.
    """
    # 1. Extract Notion URLs (released feature pages)
    urls = extract_notion_urls(message_text)
    if not urls:
        return [
            "I couldn't find any Notion page URLs in your message. Please share "
            "the Notion links for the features you released."
        ]

    log.info(f"Found {len(urls)} Notion URL(s): {urls}")

    # 2. Fetch released feature pages (QA Notes only — same source as kb_update)
    feature_pages = []
    for url in urls:
        page_id = notion_url_to_page_id(url)
        if not page_id:
            log.warning(f"Could not extract page ID from: {url}")
            continue
        try:
            qa_text = fetch_notion_blocks_qa_only(page_id)
            # Get the page title for display
            resp = requests.get(
                f"https://api.notion.com/v1/pages/{page_id}",
                headers=NOTION_HEADERS,
            )
            resp.raise_for_status()
            page_obj = resp.json()
            title = ""
            for prop in (page_obj.get("properties") or {}).values():
                if prop.get("type") == "title":
                    title = " ".join(t.get("plain_text", "") for t in prop.get("title", []))
                    break
            feature_pages.append({"id": page_id, "title": title, "content": qa_text})
            log.info(f"  Fetched feature page: {title}")
        except Exception as e:
            log.error(f"  Failed to fetch feature page {page_id}: {e}")

    if not feature_pages:
        return [
            "I couldn't fetch any of the Notion pages. Please check the URLs and "
            "make sure the Notion integration has access to those pages."
        ]

    empty_pages = [p for p in feature_pages if not (p["content"] or "").strip()]
    if empty_pages and len(empty_pages) == len(feature_pages):
        names = ", ".join(f"*{p['title']}*" for p in empty_pages)
        return [
            f"I couldn't find any content under the *QA Notes* section for: {names}. "
            "I read the QA Notes section of Notion feature pages. Please make sure "
            "the section exists and has content."
        ]

    # 3. Build the release summary
    summary_prompt = """You are reading QA Notes from product feature pages. These notes describe what was built and how the feature works.

Create a concise bullet-point summary of the key features and changes released. Each bullet should be one clear sentence.

OUTPUT FORMAT (strict Slack mrkdwn):
- Use single * for bold (*bold*). NEVER use ** or ## or ### or any markdown headers.
- Use plain - for bullet points.
- Keep it to one section: a flat bullet list of key changes. No sub-sections, no numbered lists, no headers.
- Be specific but concise. Each bullet = one feature or change."""

    pages_content = "\n\n===\n\n".join(
        f"PAGE: {p['title']}\n\nQA NOTES:\n{p['content']}" for p in feature_pages
    )
    release_summary = call_llm_fn(summary_prompt, pages_content, model_hint="pro")
    log.info(f"Release summary generated ({len(release_summary)} chars).")

    # 4. Fetch all product brain cards
    db_schema = fetch_database_schema(PRODUCT_BRAIN_DB)
    pages = fetch_all_brain_pages(PRODUCT_BRAIN_DB)
    log.info(f"Fetched {len(pages)} product brain page(s).")

    summaries = []
    for p in pages:
        s = summarize_brain_card(p)
        if not s["title"]:
            continue
        summaries.append((p, s))

    log.info(f"Analyzing against {len(summaries)} product brain card(s).")

    # 5. Score relevance in batches
    BATCH_SIZE = 15
    relevant = []
    for i in range(0, len(summaries), BATCH_SIZE):
        batch = summaries[i:i + BATCH_SIZE]

        batch_blobs = []
        for page, s in batch:
            body_text = fetch_page_full_content(page["id"])[:600]
            batch_blobs.append(
                f"[ID:{page['id']}] {s['title']}\n"
                f"  Module: {s['module']} / Sub-module: {s['submodule']}\n"
                f"  Status: {s['status']}\n"
                f"  Content: {body_text}\n"
                f"  URL: {s['url']}"
            )
        batch_text = "\n\n".join(batch_blobs)

        scoring_prompt = """You are identifying which product brain cards need to be updated based on a product release.

Given the release summary and a batch of product brain cards, return a JSON array of card IDs (the value after [ID:...]) that are relevant and likely need updates.

A card is RELEVANT if:
- It directly covers the SAME feature, area, or workflow that was changed in this release
- It describes specific functionality, behavior, customer pain point, or knowledge that was modified, added, or removed by this release
- The card's existing content would be INCORRECT or INCOMPLETE without an update

A card is NOT relevant if:
- It merely mentions a related concept but covers a different feature
- It's in the same general area but the release doesn't change what the card describes
- The connection is only tangential or thematic

Be PRECISE — only include cards whose content is directly affected by the release. Quality over quantity.

If the user's request includes explicit exclusions or scoping constraints (e.g., "ignore X", "skip Y", "except for Z"), honor them.

Return ONLY a JSON array of ID strings. Example: ["abc-123", "def-456"]
If none are relevant, return: []"""

        user_prompt = (
            f"RELEASE SUMMARY:\n{release_summary}\n\n"
            f"CARDS TO EVALUATE:\n{batch_text}\n\n"
            f"USER REQUEST (verbatim — honor any explicit exclusions or scoping constraints):\n{message_text}"
        )

        try:
            raw = call_llm_fn(scoring_prompt, user_prompt, model_hint="flash")
            cleaned = strip_code_fences(raw)
            ids = json.loads(cleaned)
            if isinstance(ids, list):
                id_set = {str(x) for x in ids}
                for page, s in batch:
                    if str(page["id"]) in id_set:
                        relevant.append((page, s))
        except Exception as e:
            log.warning(f"Batch scoring failed: {e}")

    log.info(f"Found {len(relevant)} potentially relevant card(s).")

    # 6. Detailed proposal for each relevant card
    detailed_proposals = []
    for page, s in relevant:
        body_text = fetch_page_full_content(page["id"])
        card_detail = (
            f"CARD TITLE: {s['title']}\n"
            f"MODULE: {s['module']} / SUB-MODULE: {s['submodule']}\n"
            f"STATUS: {s['status']}\n"
            f"URL: {s['url']}\n"
            f"CURRENT CONTENT (plain text):\n{body_text[:3500]}"
        )

        proposal_prompt = """You are a product knowledge writer updating an internal product brain card after a product release.

Given the release details and an existing product brain card, determine what specific changes should be made to the card's body content.

CRITICAL: Only propose changes DIRECTLY related to this card's topic. If the release affects a different feature than what this card covers, return an empty list []. Do NOT propose adding content about tangentially related features.

SCOPE — what NOT to propose:
- Do NOT propose changes whose only purpose is to fix existing inconsistencies in the card (formatting, tone, structure, typos). Drive-by cleanup of pre-existing content makes the approval process painful.
- Only propose a change if the release content itself requires it.
- If the user's request includes explicit exclusions (e.g., "ignore X", "skip Y"), honor them.

WRITING STYLE for any new/edited text:
- Short, factual sentences. No filler or marketing language.
- Use bullet lists or numbered lists wherever they fit.
- Speak about the product / customer / workflow directly. Avoid second-person help-center phrasing ("you can…") unless the existing card already uses it.
- Present tense, current state. The product brain captures how things ARE today, not what was just changed. Do NOT use "now", "will now", "has been updated", "recently", "from now on", or any wording that signals a change from a previous state.

OUTPUT FORMAT: Return ONLY a JSON array of change objects. No prose, no markdown, no code fences. If no changes are needed, return: []

Each change object has these fields:
{
  "type": "UPDATE" | "ADD" | "REMOVE",
  "section": "name of the section in the card (or short description of where in the card)",
  "why": "one sentence explaining why this change is needed",
  "before": "exact current text that will be changed (REQUIRED for UPDATE and REMOVE; omit or empty for ADD)",
  "after": "exact new text that will replace it (REQUIRED for UPDATE and ADD; omit or empty for REMOVE)"
}

RULES:
- "before" and "after" must contain actual prose, NOT a description of it.
- Plain text only inside "before" and "after" — no markdown, no code fences, no HTML.
- Keep each change narrowly scoped (one paragraph or one bullet per change)."""

        user_prompt = (
            f"RELEASE SUMMARY:\n{release_summary}\n\n"
            f"{card_detail}\n\n"
            f"USER REQUEST (verbatim — honor any explicit exclusions or scoping constraints):\n{message_text}"
        )

        try:
            raw = call_llm_fn(proposal_prompt, user_prompt, model_hint="pro")
            cleaned = strip_code_fences(raw)
            if cleaned.strip().upper().startswith("NO_CHANGES"):
                continue
            changes_list = json.loads(cleaned)
            if not isinstance(changes_list, list):
                log.warning(f"Card {page['id']}: proposal was not a JSON list, skipping.")
                continue
            if changes_list:
                detailed_proposals.append({
                    "card_id": page["id"],
                    "card_title": s["title"],
                    "card_url": s["url"],
                    "module": s["module"],
                    "submodule": s["submodule"],
                    "status": s["status"],
                    "changes_list": changes_list,
                    "changes": render_changes_as_mrkdwn(changes_list),
                })
        except json.JSONDecodeError as e:
            log.warning(f"Failed to parse proposal JSON for card {page['id']}: {e}")
        except Exception as e:
            log.warning(f"Failed to analyze card {page['id']}: {e}")

    # 7. Should a NEW card be created?
    module_usage = collect_property_value_usage(pages, MODULE_PROPERTY)
    submodule_usage = collect_property_value_usage(pages, SUBMODULE_PROPERTY)

    def _format_usage(counter, top=30):
        if not counter:
            return "(none)"
        items = counter.most_common(top)
        return ", ".join(f"{name} ({n})" for name, n in items)

    existing_titles = "\n".join(
        f"- {p['card_title']} [Module: {p['module']} / Sub-module: {p['submodule']}]"
        for p in detailed_proposals
    ) or "(none)"

    new_card_prompt = f"""Based on this product release and the cards already being updated, determine if a NEW product brain card should be created.

A new card is needed if:
- The release introduces a feature, customer pain point, or product concept not covered by any existing card
- The scope is significant enough to deserve its own card
- Splitting the knowledge into a dedicated card will make the brain easier to navigate

PROPERTY RULES (must follow exactly):
- "title": short, specific, sentence case (capitalize only the first word). Names a feature, concept, or workflow — NOT a marketing phrase.
- "module": MUST match one of these existing module values whenever a reasonable fit exists — only invent a new value if NONE of these fit. Existing modules (with usage count): {_format_usage(module_usage)}
- "submodule": MUST match one of these existing sub-module values whenever a reasonable fit exists — only invent a new value if NONE fit. Existing sub-modules: {_format_usage(submodule_usage)}
- The "status" property is set automatically to "{PUBLISHED_STATUS}" — do NOT include it in your response.

BODY RULES:
- Plain markdown body using #/##/### headings, - for bullets, 1. for numbered steps, blank lines between blocks.
- Short, factual sentences. No filler.
- Present tense, current state. The brain captures how the product IS, not what just changed. Do NOT use "now", "will now", "has been added", "recently", "new feature", "from now on".
- Aim for a card that another teammate can read cold and understand the feature.

If the user's request includes explicit exclusions, honor them.

If a new card is needed, return a JSON object:
{{"needed": true, "title": "...", "module": "...", "submodule": "...", "body": "markdown body content"}}

If NOT needed, return: {{"needed": false}}"""

    new_card_raw = call_llm_fn(
        new_card_prompt,
        (
            f"Release summary:\n{release_summary}\n\n"
            f"Cards already being updated:\n{existing_titles}\n\n"
            f"USER REQUEST (verbatim — honor any explicit exclusions or scoping constraints):\n{message_text}"
        ),
        model_hint="pro",
    )

    new_card_plan = None
    try:
        cleaned = strip_code_fences(new_card_raw)
        parsed = json.loads(cleaned)
        if parsed.get("needed"):
            new_card_plan = parsed
    except (json.JSONDecodeError, AttributeError):
        pass

    # 8. Format the Slack response
    return format_proposal_message(
        detailed_proposals, new_card_plan, len(summaries), feature_pages, release_summary
    )


def format_proposal_message(proposals, new_card_plan, total_cards, feature_pages, release_summary):
    """Build the proposal as a list of Slack messages — one per card."""
    messages = []

    feature_names = ", ".join(p["title"] for p in feature_pages if p.get("title"))
    header = (
        "*FEATURE SUMMARY (from QA Notes):*\n\n"
        f"{release_summary}\n\n"
        f"_Scanned {total_cards} product brain card(s) for: {feature_names}_"
    )
    messages.append(header)

    if not proposals and not new_card_plan:
        messages.append(
            "After detailed analysis, no changes are needed for any existing cards "
            "and no new card is recommended."
        )
        return messages

    if proposals:
        messages.append(f"*CARDS TO UPDATE ({len(proposals)}):*")
        for i, p in enumerate(proposals, 1):
            messages.append(_render_card_message(i, p))

    if new_card_plan:
        messages.append(_render_new_card_message(new_card_plan))

    trailer = (
        "_Note: When approved, existing cards keep their current Status; new cards "
        f"are created with Status = *{PUBLISHED_STATUS}*. The product brain is English-only._\n\n"
        "---\n"
        "You can ask me to revise the proposal, or reply *yes, proceed* to apply the changes."
    )
    messages.append(trailer)
    return messages


# ---------------------------------------------------------------------------
# Phase 1.5: revise based on user feedback
# ---------------------------------------------------------------------------

def handle_product_brain_revision(thread_context, revision_request, call_llm_fn):
    """Revise the product brain proposal based on user feedback in the thread."""
    revision_prompt = """You are revising a product brain update proposal based on user feedback.

You have the full Slack thread with the original proposal and the user's corrections. Generate a REVISED proposal that incorporates ALL of the user's feedback.

IMPORTANT:
- Read the full thread carefully to understand what was originally proposed.
- Apply ALL corrections the user asked for. Do not ignore any feedback.
- If the user asked to remove a change, remove it entirely from the output.
- If the user corrected factual details, apply those corrections to the before/after text.
- Show the FULL revised proposal (all cards and all changes that still apply), not just what changed.

SCOPE — what NOT to propose:
- Do NOT propose changes whose only purpose is to fix existing inconsistencies in a card (formatting, tone, structure). Only propose a change if the release content requires it.

WRITING STYLE for any new/edited text:
- Short, factual sentences. No filler or marketing language.
- Bullet lists wherever they fit.
- Present tense, current state. The brain captures how the product IS today, not what just changed. Do NOT use "now", "will now", "has been updated", "recently", "from now on".

OUTPUT FORMAT: Return ONLY a JSON object with this shape. No prose, no markdown, no code fences:

{
  "cards": [
    {
      "card_title": "exact title of the card",
      "card_url": "url of the card (copy from thread if available, else empty string)",
      "module": "module value (copy from thread if available)",
      "submodule": "sub-module value (copy from thread if available)",
      "changes": [
        {
          "type": "UPDATE" | "ADD" | "REMOVE",
          "section": "name of the section in the card",
          "why": "one sentence explaining why this change is needed",
          "before": "exact current text (REQUIRED for UPDATE and REMOVE; omit or empty for ADD)",
          "after": "exact new text (REQUIRED for UPDATE and ADD; omit or empty for REMOVE)"
        }
      ]
    }
  ],
  "new_card": null | {
    "title": "card title",
    "module": "module value",
    "submodule": "sub-module value",
    "body": "markdown body content"
  }
}

RULES:
- "before" and "after" must contain actual prose, NOT a description of it.
- Plain text only inside "before" and "after" — no markdown, no code fences, no HTML.
- If a card has no remaining changes after the revision, omit it from "cards".
- If no new card is needed, set "new_card" to null."""

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
        return [
            "I couldn't parse my own revised proposal. Please ask me to revise again, "
            "or describe the corrections in a different way."
        ]

    revised_proposals = []
    for c in data.get("cards", []) or []:
        changes_list = c.get("changes") or []
        if not changes_list:
            continue
        revised_proposals.append({
            "card_title": c.get("card_title", ""),
            "card_url": c.get("card_url", ""),
            "module": c.get("module", ""),
            "submodule": c.get("submodule", ""),
            "changes_list": changes_list,
            "changes": render_changes_as_mrkdwn(changes_list),
        })

    new_card_plan = data.get("new_card") or None

    if not revised_proposals and not new_card_plan:
        return [
            "*REVISED PROPOSAL:*\n\n"
            "No changes remain after your revisions. Let me know if you'd like to start over."
        ]

    messages = ["*REVISED PROPOSAL:*"]
    if revised_proposals:
        messages.append(f"*CARDS TO UPDATE ({len(revised_proposals)}):*")
        for i, p in enumerate(revised_proposals, 1):
            messages.append(_render_card_message(i, p))
    if new_card_plan:
        messages.append(_render_new_card_message(new_card_plan))
    messages.append(
        "_Note: When approved, existing cards keep their current Status; new cards "
        f"are created with Status = *{PUBLISHED_STATUS}*._\n\n"
        "---\n"
        "You can ask me to revise again, or reply *yes, proceed* to apply the changes."
    )
    return messages


# ---------------------------------------------------------------------------
# Phase 2: execute approved changes
# ---------------------------------------------------------------------------

def execute_approved_product_brain_changes(original_query, approval_text, thread_context, call_llm_fn):
    """Apply the approved changes to the Notion product brain.

    Uses the thread context to recover the final approved proposal — same
    pattern as kb_update so user revisions in the thread are honored.
    """
    extract_prompt = """You are reading a Slack thread where a product brain update was proposed, possibly revised, and then approved.

Extract the FINAL version of the changes to apply. Look at the MOST RECENT proposal in the thread (the user may have asked for revisions — use the last revised version, not the original).

Return a JSON object:
{
  "card_updates": [
    {
      "card_title": "...",
      "card_url": "...",
      "changes_description": "full plain-text description of all changes to make to this card — include the Before/After text for each change so a downstream LLM can apply them",
      "change_summary": [
        {"type": "UPDATE" | "ADD" | "REMOVE", "section": "exact section name"}
      ]
    }
  ],
  "new_card": null | {
    "title": "...",
    "module": "...",
    "submodule": "...",
    "body": "markdown body content"
  }
}

RULES:
- "change_summary" must contain ONE entry per change shown in the most recent proposal for that card, in the same order. Use the exact type and section name shown.
- "changes_description" is the human-readable description used by the next step to actually rewrite the card body. Keep it complete and unambiguous.
- If the user's approval message specifies only certain changes to apply (e.g., "only card 1", "only changes 1 and 3"), include only those.
- Return ONLY the JSON object, nothing else."""

    raw = call_llm_fn(
        extract_prompt,
        f"Thread conversation:\n{thread_context}\n\nApproval message:\n{approval_text}",
        model_hint="pro",
    )

    try:
        cleaned = strip_code_fences(raw)
        pending = json.loads(cleaned)
    except (json.JSONDecodeError, AttributeError):
        return "I couldn't parse the changes from our conversation. Could you clarify which changes to apply?"

    card_updates = pending.get("card_updates", []) or []
    new_card_plan = pending.get("new_card") or None

    db_schema = fetch_database_schema(PRODUCT_BRAIN_DB)
    all_pages = fetch_all_brain_pages(PRODUCT_BRAIN_DB)
    pages_by_title = {}
    for page in all_pages:
        title = _page_text(page, TITLE_PROPERTY)
        if title:
            pages_by_title[title.strip().lower()] = page

    results = []

    # --- Apply updates ---
    for update in card_updates:
        card_title = (update.get("card_title") or "").strip()
        card_url = update.get("card_url", "")
        changes_description = update.get("changes_description", "")
        change_summary = update.get("change_summary", []) or []

        page = pages_by_title.get(card_title.lower())
        if not page:
            for title_l, p in pages_by_title.items():
                if card_title.lower() in title_l or title_l in card_title.lower():
                    page = p
                    break

        if not page:
            results.append({
                "kind": "skipped",
                "title": card_title,
                "url": card_url,
                "items": change_summary,
                "error": "could not find card in product brain",
            })
            continue

        page_id = page["id"]
        log.info(f"Updating brain card {page_id}: {card_title}")

        try:
            current_body = fetch_page_full_content(page_id)
            current_status = _page_text(page, STATUS_PROPERTY)

            update_prompt = """You are updating a product brain card's body content based on approved changes.

Given the current body content (plain-text rendering of the existing Notion blocks) and the specific changes to make, produce the FULL updated body in simple markdown.

OUTPUT FORMAT (the markdown will be parsed back into Notion blocks):
- Use # / ## / ### for headings
- Use - for bullet points
- Use 1. for numbered list items
- Blank line between blocks
- No code fences in the output, no HTML, no inline formatting (bold/italic/links won't be preserved)

RULES:
- Preserve sections and content that aren't affected by the changes.
- Apply ALL the specified changes — do not skip any.
- Match the structure and tone of the existing card (don't restructure unaffected sections).
- Present tense, current state. The card describes how things ARE today, not what just changed. Do NOT use "now", "will now", "has been updated", "recently", "from now on".
- Return ONLY the markdown body, nothing else (no preamble, no code fences around the whole output)."""

            user_prompt = (
                f"CHANGES TO APPLY:\n{changes_description}\n\n"
                f"CARD TITLE: {card_title}\n"
                f"CURRENT BODY (plain text rendering of existing Notion blocks):\n{current_body}"
            )

            new_body_md = strip_code_fences(call_llm_fn(update_prompt, user_prompt, model_hint="pro"))

            replace_page_body(page_id, new_body_md)

            results.append({
                "kind": "updated",
                "title": card_title,
                "url": notion_page_url(page_id),
                "status_note": f"Status preserved: {current_status}" if current_status else "",
                "items": change_summary,
            })

        except Exception as e:
            log.error(f"Failed to update brain card {page_id}: {e}")
            results.append({
                "kind": "failed",
                "title": card_title,
                "url": card_url or notion_page_url(page_id),
                "items": change_summary,
                "error": str(e),
            })

    # --- Create new card ---
    if new_card_plan:
        log.info(f"Creating new brain card: {new_card_plan.get('title', '?')}")
        try:
            create_prompt = """You are writing the body of a brand-new product brain card in simple markdown.

OUTPUT FORMAT (the markdown will be parsed back into Notion blocks):
- Use # / ## / ### for headings
- Use - for bullet points
- Use 1. for numbered list items
- Blank line between blocks
- No code fences in the output, no HTML, no inline formatting

RULES:
- Short, factual sentences. No filler or marketing language.
- Use bullet lists wherever they fit.
- Present tense, current state. The card describes how the product IS today, not what was just added or changed. Do NOT use "now", "will now", "has been added", "recently", "new feature", "from now on". Write as if the feature has always existed.
- Cover the topic well enough that a teammate reading cold can understand it.
- Return ONLY the markdown body, nothing else."""

            outline_prompt = (
                f"TITLE: {new_card_plan.get('title', '')}\n"
                f"MODULE: {new_card_plan.get('module', '')}\n"
                f"SUB-MODULE: {new_card_plan.get('submodule', '')}\n"
                f"DRAFT BODY (use as the base — refine, don't restart from scratch):\n"
                f"{new_card_plan.get('body', '')}"
            )

            body_md = strip_code_fences(call_llm_fn(create_prompt, outline_prompt, model_hint="pro"))

            new_page = create_brain_page(
                PRODUCT_BRAIN_DB,
                db_schema,
                title=new_card_plan.get("title", "Untitled"),
                module=new_card_plan.get("module", ""),
                submodule=new_card_plan.get("submodule", ""),
                body_markdown=body_md,
            )
            new_url = notion_page_url(new_page["id"])
            results.append({
                "kind": "created",
                "title": new_card_plan.get("title", "(new card)"),
                "url": new_url,
                "status_note": f"Status set to: {PUBLISHED_STATUS}",
                "items": [],
            })

        except Exception as e:
            log.error(f"Failed to create new brain card: {e}")
            results.append({
                "kind": "failed_create",
                "title": new_card_plan.get("title", "(new card)"),
                "url": "",
                "items": [],
                "error": str(e),
            })

    return format_apply_report(results)


def format_apply_report(results):
    """Render the post-apply report as Slack mrkdwn."""
    if not results:
        return "No changes were applied."

    updated = [r for r in results if r["kind"] == "updated"]
    created = [r for r in results if r["kind"] == "created"]
    skipped = [r for r in results if r["kind"] == "skipped"]
    failed = [r for r in results if r["kind"] in ("failed", "failed_create")]

    parts = ["*Product brain changes applied:*\n"]

    def render_item_bullets(items):
        lines = []
        for ch in items or []:
            typ = str(ch.get("type", "UPDATE")).upper()
            sec = (ch.get("section") or "").strip()
            if sec:
                lines.append(f"    • [{typ}] {sec}")
            else:
                lines.append(f"    • [{typ}]")
        return lines

    if updated:
        parts.append(f"*Cards updated ({len(updated)}):*")
        for r in updated:
            parts.append(f"*{r['title']}* — updated")
            if r.get("url"):
                parts.append(f"    {r['url']}")
            if r.get("status_note"):
                parts.append(f"    _{r['status_note']}_")
            parts.extend(render_item_bullets(r.get("items")))
            parts.append("")

    if created:
        parts.append(f"*New cards created ({len(created)}):*")
        for r in created:
            parts.append(f"*{r['title']}* — created")
            if r.get("url"):
                parts.append(f"    {r['url']}")
            if r.get("status_note"):
                parts.append(f"    _{r['status_note']}_")
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

    parts.append("All updates are live in the product brain.")
    return "\n".join(parts).rstrip()
