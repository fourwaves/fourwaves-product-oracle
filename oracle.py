#!/usr/bin/env python3
"""
Fourwaves Product Oracle — Slack bot with pluggable skills.

Polls #product-oracle for new messages, classifies which skill to route to,
and handles the full request lifecycle (ack → process → respond) in threads.

Skills:
  - kb_update: Update Intercom knowledge base articles based on Notion product release pages

Usage:
    python oracle.py --slack-poll    # Poll Slack (for GitHub Actions cron)
"""

import os
import sys
import json
import time
import random
import argparse
import logging
from datetime import datetime

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ORACLE_CHANNEL_ID = os.environ.get("ORACLE_CHANNEL_ID") or "C0ACXJ4RNJ0"
SLACK_PROCESSED_FILE = os.path.join(os.path.dirname(__file__), "oracle_processed_messages.json")
LAST_POLL_FILE = os.path.join(os.path.dirname(__file__), "oracle_last_poll_ts.txt")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("oracle")

# ---------------------------------------------------------------------------
# LLM backend
# ---------------------------------------------------------------------------

try:
    from google import genai
    HAS_GEMINI_SDK = True
except (ImportError, Exception):
    HAS_GEMINI_SDK = False


def _extract_quota_details(exc):
    """Pull quotaId and details out of a Gemini 429 for diagnostic logging."""
    try:
        # google.genai raises errors.ClientError with .details (dict)
        details = getattr(exc, "details", None) or {}
        if isinstance(details, str):
            try:
                details = json.loads(details)
            except Exception:
                return str(details)[:500]
        err = details.get("error", {}) if isinstance(details, dict) else {}
        violations = []
        for d in err.get("details", []) or []:
            if d.get("@type", "").endswith("QuotaFailure"):
                for v in d.get("violations", []) or []:
                    violations.append(v.get("quotaId") or v.get("quotaMetric") or str(v))
        if violations:
            return "quotas=" + ",".join(violations)
        return str(details)[:500]
    except Exception:
        return "<no details>"


def call_llm(system_prompt, user_prompt, model_hint="flash", max_attempts=4):
    """Call LLM with exponential backoff on 429/transient errors.

    model_hint: 'flash' for fast tasks, 'pro' for synthesis.
    Retries 429 RESOURCE_EXHAUSTED with backoff 2s, 4s, 8s.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not (HAS_GEMINI_SDK and gemini_key):
        raise RuntimeError("No LLM backend available. Set GEMINI_API_KEY.")

    client = genai.Client(api_key=gemini_key)
    model = "gemini-2.0-flash" if model_hint == "flash" else "gemini-2.5-flash"

    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=f"{system_prompt}\n\n{user_prompt}",
            )
            return resp.text.strip()
        except Exception as e:
            last_exc = e
            msg = str(e)
            is_429 = "429" in msg or "RESOURCE_EXHAUSTED" in msg
            is_503 = "503" in msg or "UNAVAILABLE" in msg
            if not (is_429 or is_503) or attempt == max_attempts:
                if is_429:
                    log.error(f"  LLM 429 after {attempt} attempt(s). {_extract_quota_details(e)}")
                raise
            # Backoff with jitter: 2s, 4s, 8s (+/- 25%)
            base = 2 ** attempt
            sleep = base + random.uniform(-base * 0.25, base * 0.25)
            log.warning(f"  LLM {'429' if is_429 else '503'} attempt {attempt}/{max_attempts}, retrying in {sleep:.1f}s... {_extract_quota_details(e) if is_429 else ''}")
            time.sleep(sleep)
    raise last_exc  # unreachable, but keeps type-checkers happy


# ---------------------------------------------------------------------------
# Slack helpers
# ---------------------------------------------------------------------------

def slack_api(method, http_method="POST", **kwargs):
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if not token:
        raise RuntimeError("SLACK_BOT_TOKEN not set.")
    url = f"https://slack.com/api/{method}"
    headers = {"Authorization": f"Bearer {token}"}
    if http_method == "GET":
        resp = requests.get(url, headers=headers, params=kwargs)
    else:
        headers["Content-Type"] = "application/json; charset=utf-8"
        resp = requests.post(url, headers=headers, json=kwargs)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"Slack API error ({method}): {data.get('error', 'unknown')}")
    return data


def slack_post_message(channel, text, thread_ts=None):
    kwargs = {"channel": channel, "text": text, "unfurl_links": False, "unfurl_media": False}
    if thread_ts:
        kwargs["thread_ts"] = thread_ts
    return slack_api("chat.postMessage", **kwargs)


def slack_join_channel(channel):
    try:
        slack_api("conversations.join", channel=channel)
        log.info(f"Bot joined channel {channel}.")
    except Exception as e:
        log.warning(f"Could not join channel {channel}: {e}")


def slack_get_channel_messages(channel, oldest=None, limit=200):
    """Fetch channel messages with automatic pagination to get ALL messages since oldest."""
    all_messages = []
    cursor = None
    while True:
        kwargs = {"channel": channel, "limit": limit}
        if oldest:
            kwargs["oldest"] = oldest
        if cursor:
            kwargs["cursor"] = cursor
        data = slack_api("conversations.history", http_method="GET", **kwargs)
        messages = data.get("messages", [])
        all_messages.extend(messages)
        # Check for more pages
        next_cursor = data.get("response_metadata", {}).get("next_cursor", "")
        if not next_cursor or not data.get("has_more"):
            break
        cursor = next_cursor
        log.info(f"  Fetching next page (have {len(all_messages)} messages so far)...")
    return {"messages": all_messages, "ok": True}


def slack_get_thread_replies(channel, ts, limit=100):
    data = slack_api("conversations.replies", http_method="GET", channel=channel, ts=ts, limit=limit)
    return data.get("messages", [])


def get_bot_user_id():
    try:
        data = slack_api("auth.test", http_method="GET")
        return data.get("user_id", "")
    except Exception:
        return ""


def load_processed_messages():
    if os.path.exists(SLACK_PROCESSED_FILE):
        with open(SLACK_PROCESSED_FILE, "r") as f:
            return json.load(f)
    return {}


def save_processed_messages(processed):
    with open(SLACK_PROCESSED_FILE, "w") as f:
        json.dump(processed, f, indent=2)


FALLBACK_ACKS = [
    ":crystal_ball: Consulting the oracle... hold tight!",
    ":mag: Digging through the archives... be right back.",
    ":hourglass_flowing_sand: Processing your question — this may take a moment.",
    ":robot_face: Beep boop, on it!",
    ":brain: Thinking really hard... give me a sec.",
    ":books: Summoning the user insights, one moment...",
    ":telescope: Scanning the feedback universe for answers...",
]


def generate_processing_message():
    """Return a funny ack. Falls back to a hardcoded message if the LLM is unavailable."""
    system_prompt = """Generate a funny, creative message telling the user their question is being processed.
Use emojis. Make it 1-3 sentences. Be creative and different each time.
Output ONLY the message, nothing else."""
    seed = f"Seed: {datetime.now().isoformat()} {random.randint(1, 99999)}"
    try:
        # Cheap + fast — don't burn too many retries on an ack
        return call_llm(system_prompt, seed, model_hint="flash", max_attempts=2)
    except Exception as e:
        log.warning(f"  Ack LLM failed, using fallback: {e}")
        return random.choice(FALLBACK_ACKS)


def split_response(text, max_len=39000):
    """Split text into chunks that fit Slack's limit, never cutting inside code blocks."""
    chunks = []
    while len(text) > max_len:
        # Find a safe split point: a double newline that's NOT inside a code block
        best_split = -1
        search_end = max_len

        # Count open/close ``` pairs to know if we're inside a code block
        while search_end > max_len // 4:
            pos = text.rfind("\n\n", 0, search_end)
            if pos < 0:
                break
            # Count ``` occurrences before this position
            prefix = text[:pos]
            backtick_count = prefix.count("```")
            if backtick_count % 2 == 0:
                # Even count = we're outside a code block, safe to split
                best_split = pos
                break
            # Inside a code block — try an earlier split point
            search_end = pos

        if best_split < 0:
            # Fallback: force split at max_len (shouldn't happen often)
            best_split = max_len

        chunks.append(text[:best_split])
        text = text[best_split:].lstrip("\n")
    if text:
        chunks.append(text)
    return chunks


def post_long_message(channel, text, thread_ts):
    """Post a message, splitting into multiple if too long for Slack."""
    max_len = 39000
    if len(text) <= max_len:
        slack_post_message(channel, text, thread_ts=thread_ts)
    else:
        for chunk in split_response(text, max_len):
            slack_post_message(channel, chunk, thread_ts=thread_ts)
            time.sleep(0.5)


# ---------------------------------------------------------------------------
# Skill routing
# ---------------------------------------------------------------------------

SKILL_DESCRIPTIONS = """Available skills:
- transcripts: Search through full call transcripts (sales calls, support calls, demos, onboarding, feedback sessions). Triggered when the user explicitly mentions "call transcripts", "transcripts", "calls", or wants to search through what was said in actual calls/meetings.
- insights: Answer product questions using the Fourwaves user insights database. Triggered when the user mentions "user insights", "insights", or asks about user feedback, feature requests, pain points, or product topics WITHOUT mentioning call transcripts.
- kb_update: Update the Intercom knowledge base (help center articles) based on Notion product release pages. Triggered when the user mentions updating the knowledge base, help center, or provides Notion page URLs describing new features or product updates.
"""


def classify_skill(text):
    """Classify a Slack message into a skill name or 'none'.

    Uses keyword pre-checks for unambiguous signals before falling back to LLM.
    """
    text_lower = text.lower()

    # Keyword pre-checks — deterministic routing for clear-cut cases
    transcript_keywords = ["call transcript", "call transcripts", "transcripts", "in calls", "in the calls", "in meetings", "said in calls"]
    if any(kw in text_lower for kw in transcript_keywords):
        log.info("  Keyword pre-check → transcripts")
        return "transcripts"

    if ("notion.so/" in text_lower or "notion.site/" in text_lower) and any(
        kw in text_lower for kw in ["knowledge base", "help center", "kb", "article"]
    ):
        log.info("  Keyword pre-check → kb_update")
        return "kb_update"

    # Fall back to LLM for ambiguous messages
    system_prompt = f"""You route Slack messages to the correct skill handler.

{SKILL_DESCRIPTIONS}

Rules:
- If the message explicitly mentions "call transcript(s)", "transcripts", "calls", "what was said in calls/meetings", or wants to search through actual call recordings/transcripts → return "transcripts"
- If the message contains one or more Notion page URLs (notion.so or notion.site) and talks about features released, product updates, or knowledge base/help center updates → return "kb_update"
- If the message mentions "user insights", "insights", or asks about user feedback, feature requests, pain points, what users think (without specifically mentioning call transcripts) → return "insights"
- If the message is a system notification (e.g., "X was added to the channel"), casual chat, or doesn't match any skill → return "none"

Reply with ONLY the skill name or "none". Nothing else."""

    result = call_llm(system_prompt, f"Message: {text}", model_hint="flash")
    return result.strip().lower().split()[0]  # Take first word only


def classify_followup(thread_context, followup_text, skill):
    """Classify a follow-up message in a thread. Works for any skill."""
    system_prompt = f"""You are evaluating a follow-up message in a Slack thread where the bot previously responded. The thread skill is: {skill}.

Classify the follow-up into one of:
- "approve" — The user explicitly approves and wants the bot to execute/apply changes (e.g., "yes proceed", "go ahead", "looks good apply it", "ok I'm satisfied now apply those updates", "create the drafts")
- "revise" — The user wants the bot to adjust, correct, or refine its previous response (e.g., "actually change X to Y", "remove the part about Z", "make those adjustments and show me again", feedback with corrections)
- "reject" — The user explicitly cancels (e.g., "cancel", "nevermind", "stop")
- "followup" — The user is asking a follow-up question, requesting additional action, or continuing the conversation in any other way (e.g., "tell me more about X", "can you also draft an email", "what about Y?")

IMPORTANT: "revise" means the user is giving feedback on the bot's previous response and wants a corrected version. "followup" means the user is asking something new or additional. "approve" ONLY when the user explicitly says to proceed with execution.

Reply with ONLY one of: approve, revise, reject, followup"""

    result = call_llm(system_prompt, f"Thread so far:\n{thread_context}\n\nFollow-up message:\n{followup_text}", model_hint="flash")
    return result.strip().lower().split()[0]


# ---------------------------------------------------------------------------
# Main polling loop
# ---------------------------------------------------------------------------

def load_last_poll_ts():
    """Load the timestamp of the last successful poll. Falls back to 24h ago.

    Includes a safety check: if the stored timestamp is more than 7 days old
    or in the future, reset to 24h ago to prevent stuck polls.
    """
    fallback = str(time.time() - 86400)
    if os.path.exists(LAST_POLL_FILE):
        try:
            with open(LAST_POLL_FILE, "r") as f:
                ts_str = f.read().strip()
            ts_val = float(ts_str)
            now = time.time()
            if ts_val > now + 60:
                log.warning(f"Poll timestamp {ts_val} is in the future! Resetting to 24h ago.")
                return fallback
            if now - ts_val > 7 * 86400:
                log.warning(f"Poll timestamp {ts_val} is >7 days old. Resetting to 24h ago.")
                return fallback
            return ts_str
        except Exception:
            pass
    return fallback


def save_last_poll_ts(ts):
    """Save the last poll timestamp. Requires an explicit timestamp."""
    with open(LAST_POLL_FILE, "w") as f:
        f.write(str(ts))


def run_slack_poll():
    """Poll #product-oracle for new messages and follow-ups."""
    processed = load_processed_messages()
    slack_join_channel(ORACLE_CHANNEL_ID)
    bot_user_id = get_bot_user_id()
    if bot_user_id:
        log.info(f"Bot user ID: {bot_user_id}")

    oldest = load_last_poll_ts()
    log.info(f"Polling #{ORACLE_CHANNEL_ID} for messages since ts={oldest}...")

    try:
        data = slack_get_channel_messages(ORACLE_CHANNEL_ID, oldest=oldest)
    except Exception as e:
        log.error(f"Failed to fetch Slack messages: {e}")
        return

    messages = data.get("messages", [])
    log.info(f"Found {len(messages)} raw message(s) from Slack API since ts={oldest}.")

    # Debug: log each raw message
    for m in messages:
        mts = m.get("ts", "?")
        muser = m.get("user", m.get("bot_id", "?"))
        mtype = m.get("subtype", "normal")
        mthread = m.get("thread_ts", "")
        mtext = (m.get("text") or "")[:80]
        log.info(f"  RAW msg ts={mts} user={muser} subtype={mtype} thread_ts={mthread} text={mtext!r}")

    # --- Top-level messages ---
    new_queries = []
    for msg in messages:
        ts = msg.get("ts", "")
        if ts in processed and processed[ts].get("status") != "error":
            log.info(f"  SKIP {ts}: already processed (status={processed[ts].get('status')})")
            continue
        if msg.get("bot_id") or msg.get("subtype"):
            log.info(f"  SKIP {ts}: bot_id={msg.get('bot_id')} subtype={msg.get('subtype')}")
            continue
        if msg.get("thread_ts") and msg.get("thread_ts") != ts:
            log.info(f"  SKIP {ts}: thread reply (thread_ts={msg.get('thread_ts')})")
            continue
        text = msg.get("text", "").strip()
        if not text:
            log.info(f"  SKIP {ts}: empty text")
            continue
        new_queries.append((ts, text, msg.get("user", "")))

    if not new_queries:
        log.info("No new top-level messages to process.")
    else:
        log.info(f"{len(new_queries)} new message(s) to evaluate.")

    for ts, text, user_id in new_queries:
        log.info(f"Evaluating message {ts}: {text[:100]}...")

        # Classify into skill
        try:
            skill = classify_skill(text)
        except Exception as e:
            log.error(f"Classification failed for {ts}: {e}")
            # Tell the user instead of staying silent. Keep status=error so we retry next poll.
            try:
                err_msg = (
                    ":warning: I couldn't process this question right now "
                    "(LLM is temporarily unavailable — likely a rate limit). "
                    "I'll automatically retry on the next poll."
                )
                slack_post_message(ORACLE_CHANNEL_ID, err_msg, thread_ts=ts)
            except Exception:
                pass
            processed[ts] = {"status": "error", "error": str(e)[:500], "date": datetime.now().isoformat()}
            save_processed_messages(processed)
            continue

        if skill == "none":
            log.info(f"  No skill matched, skipping.")
            processed[ts] = {"status": "no_skill", "date": datetime.now().isoformat()}
            save_processed_messages(processed)
            continue

        # Check if bot already replied
        try:
            replies = slack_get_thread_replies(ORACLE_CHANNEL_ID, ts)
            bot_already_replied = any(
                r.get("user") == bot_user_id or r.get("bot_id")
                for r in replies if r.get("ts") != ts
            )
            if bot_already_replied:
                log.info(f"  Bot already replied, skipping.")
                processed[ts] = {"status": "already_replied", "date": datetime.now().isoformat()}
                save_processed_messages(processed)
                continue
        except Exception:
            pass

        log.info(f"  Routing to skill: {skill}")

        # Send ack
        try:
            ack = generate_processing_message()
            slack_post_message(ORACLE_CHANNEL_ID, ack, thread_ts=ts)
        except Exception as e:
            log.warning(f"  Failed to send ack: {e}")

        # Dispatch to skill
        try:
            if skill == "kb_update":
                from skills.kb_update import handle_kb_update
                response = handle_kb_update(text, call_llm)
            elif skill == "insights":
                from skills.insights import handle_insights_query
                response = handle_insights_query(text, call_llm)
            elif skill == "transcripts":
                from skills.transcripts import handle_transcript_query
                response = handle_transcript_query(text, call_llm)
            else:
                response = f"Skill '{skill}' is not yet implemented."
        except Exception as e:
            log.error(f"  Skill execution failed: {e}")
            try:
                slack_post_message(ORACLE_CHANNEL_ID, f"Something went wrong while processing your request: {e}", thread_ts=ts)
            except Exception:
                pass
            processed[ts] = {"status": "error", "skill": skill, "error": str(e), "date": datetime.now().isoformat()}
            save_processed_messages(processed)
            continue

        # Post response
        try:
            post_long_message(ORACLE_CHANNEL_ID, response, thread_ts=ts)
        except Exception as e:
            log.error(f"  Failed to post response: {e}")
            processed[ts] = {"status": "post_error", "skill": skill, "error": str(e), "date": datetime.now().isoformat()}
            save_processed_messages(processed)
            continue

        log.info(f"  Skill '{skill}' response posted ({len(response)} chars).")
        # All threads start as "active" — the bot will keep monitoring for follow-ups
        processed[ts] = {
            "status": "active",
            "skill": skill,
            "query": text[:500],
            "response_length": len(response),
            "date": datetime.now().isoformat(),
        }
        save_processed_messages(processed)

    # --- Follow-up scanning: check active threads ---
    log.info("Scanning threads for follow-ups...")

    now = datetime.now()
    active_threads = []
    for thread_ts, entry in processed.items():
        if entry.get("status") not in ("active", "awaiting_approval", "answered"):
            continue
        if ":" in thread_ts:
            continue
        try:
            entry_date = datetime.fromisoformat(entry["date"])
            if (now - entry_date).total_seconds() < 86400:
                active_threads.append((thread_ts, entry))
        except (KeyError, ValueError):
            continue

    log.info(f"Found {len(active_threads)} active thread(s) to check.")

    for thread_ts, entry in active_threads:
        try:
            replies = slack_get_thread_replies(ORACLE_CHANNEL_ID, thread_ts)
        except Exception as e:
            log.warning(f"Failed to fetch replies for {thread_ts}: {e}")
            continue

        if len(replies) < 2:
            continue

        # Find bot's last reply
        bot_last_ts = None
        for r in reversed(replies):
            if r.get("user") == bot_user_id or r.get("bot_id"):
                bot_last_ts = r.get("ts")
                break
        if not bot_last_ts:
            continue

        # Find new user messages after bot's last reply
        new_followups = []
        for r in replies:
            if float(r.get("ts", "0")) <= float(bot_last_ts):
                continue
            if r.get("user") == bot_user_id or r.get("bot_id"):
                continue
            followup_key = f"{thread_ts}:{r['ts']}"
            if followup_key in processed:
                continue
            text = r.get("text", "").strip()
            if not text:
                continue
            new_followups.append((r["ts"], text, followup_key))

        if not new_followups:
            continue

        # Build thread context
        thread_context_parts = []
        for r in replies:
            sender = "Bot" if (r.get("user") == bot_user_id or r.get("bot_id")) else "User"
            thread_context_parts.append(f"{sender}: {r.get('text', '')[:3000]}")
        thread_context = "\n\n".join(thread_context_parts)

        for followup_ts, followup_text, followup_key in new_followups:
            log.info(f"  Processing follow-up {followup_ts}: {followup_text[:100]}...")

            try:
                ack = generate_processing_message()
                slack_post_message(ORACLE_CHANNEL_ID, ack, thread_ts=thread_ts)
            except Exception:
                pass

            skill = entry.get("skill", "")
            classification = classify_followup(thread_context, followup_text, skill)
            log.info(f"  Follow-up classified as: {classification} (skill: {skill})")

            try:
                if classification.startswith("reject"):
                    response = "Got it, cancelled. Let me know if you'd like to start over."
                    processed[thread_ts]["status"] = "rejected"

                elif classification.startswith("approve"):
                    if skill == "kb_update":
                        log.info(f"  Approval detected, executing KB changes...")
                        from skills.kb_update import execute_approved_changes
                        response = execute_approved_changes(
                            entry.get("query", ""),
                            followup_text,
                            thread_context,
                            call_llm,
                        )
                        processed[thread_ts]["status"] = "completed"
                    else:
                        # For non-actionable skills, treat approve as a followup
                        response = call_llm(
                            """You are the Fourwaves Oracle. The user approved or confirmed something. Respond helpfully.
Use Slack mrkdwn: single * for bold, > for quotes. NEVER use ** or #.""",
                            f"Thread:\n{thread_context}\n\nMessage:\n{followup_text}",
                            model_hint="pro",
                        )

                elif classification.startswith("revise"):
                    if skill == "kb_update":
                        from skills.kb_update import handle_kb_revision
                        response = handle_kb_revision(thread_context, followup_text, call_llm)
                        # Keep status as active — user is still iterating
                        processed[thread_ts]["status"] = "active"
                    elif skill == "insights":
                        from skills.insights import handle_insights_followup
                        response = handle_insights_followup(thread_context, followup_text, call_llm)
                    elif skill == "transcripts":
                        from skills.transcripts import handle_transcript_followup
                        response = handle_transcript_followup(thread_context, followup_text, call_llm)
                    else:
                        response = call_llm(
                            """You are the Fourwaves Oracle. Revise your previous response based on the user's feedback.
Use Slack mrkdwn: single * for bold, > for quotes. NEVER use ** or #.""",
                            f"Thread:\n{thread_context}\n\nRevision request:\n{followup_text}",
                            model_hint="pro",
                        )

                else:  # "followup" or anything else
                    if skill == "insights":
                        from skills.insights import handle_insights_followup
                        response = handle_insights_followup(thread_context, followup_text, call_llm)
                    elif skill == "transcripts":
                        from skills.transcripts import handle_transcript_followup
                        response = handle_transcript_followup(thread_context, followup_text, call_llm)
                    else:
                        response = call_llm(
                            """You are the Fourwaves Oracle. Answer the user's follow-up based on the thread context.
Use Slack mrkdwn: single * for bold, > for quotes. NEVER use ** or #.""",
                            f"Thread:\n{thread_context}\n\nFollow-up:\n{followup_text}",
                            model_hint="pro",
                        )

            except Exception as e:
                log.error(f"  Follow-up processing failed: {e}")
                processed[followup_key] = {"status": "error", "error": str(e), "date": now.isoformat()}
                save_processed_messages(processed)
                continue

            try:
                post_long_message(ORACLE_CHANNEL_ID, response, thread_ts=thread_ts)
            except Exception as e:
                log.error(f"  Failed to post follow-up response: {e}")

            processed[followup_key] = {
                "status": f"followup_{classification}",
                "thread_ts": thread_ts,
                "text": followup_text[:200],
                "date": now.isoformat(),
            }
            # Keep the parent thread's date fresh so it stays in the active window
            processed[thread_ts]["date"] = now.isoformat()
            save_processed_messages(processed)

    # Advance poll timestamp based on what Slack actually returned.
    # Errors have already been reported to the user (with retry-with-backoff
    # inside call_llm), so we advance past them — we don't want infinite re-processing
    # of a message that keeps erroring. If the LLM was genuinely down, the user
    # was told and can re-ask.
    if messages:
        latest_msg_ts = max(float(m.get("ts", 0)) for m in messages)
        save_last_poll_ts(latest_msg_ts)
        log.info(f"Poll timestamp advanced to latest message: {latest_msg_ts}")
    else:
        # No messages returned — don't advance, keep old timestamp
        log.info(f"No messages returned, keeping poll timestamp at {oldest}")
    log.info("Polling complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fourwaves Product Oracle")
    parser.add_argument("--slack-poll", action="store_true", help="Poll Slack for new messages")
    args = parser.parse_args()

    if args.slack_poll:
        run_slack_poll()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
