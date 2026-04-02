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

ORACLE_CHANNEL_ID = os.environ.get("ORACLE_CHANNEL_ID", "C0ACXJ4RNJ0")
SLACK_PROCESSED_FILE = os.path.join(os.path.dirname(__file__), "oracle_processed_messages.json")

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


def call_llm(system_prompt, user_prompt, model_hint="flash"):
    """Call LLM. model_hint: 'flash' for fast tasks, 'pro' for synthesis."""
    gemini_key = os.environ.get("GEMINI_API_KEY", "")

    if HAS_GEMINI_SDK and gemini_key:
        client = genai.Client(api_key=gemini_key)
        model = "gemini-2.0-flash" if model_hint == "flash" else "gemini-2.5-flash"
        resp = client.models.generate_content(
            model=model,
            contents=f"{system_prompt}\n\n{user_prompt}",
        )
        return resp.text.strip()

    raise RuntimeError("No LLM backend available. Set GEMINI_API_KEY.")


# ---------------------------------------------------------------------------
# Slack helpers
# ---------------------------------------------------------------------------

def slack_api(method, **kwargs):
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if not token:
        raise RuntimeError("SLACK_BOT_TOKEN not set.")
    resp = requests.post(
        f"https://slack.com/api/{method}",
        headers={"Authorization": f"Bearer {token}"},
        json=kwargs,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"Slack API error ({method}): {data.get('error', 'unknown')}")
    return data


def slack_post_message(channel, text, thread_ts=None):
    kwargs = {"channel": channel, "text": text}
    if thread_ts:
        kwargs["thread_ts"] = thread_ts
    return slack_api("chat.postMessage", **kwargs)


def slack_join_channel(channel):
    try:
        slack_api("conversations.join", channel=channel)
        log.info(f"Bot joined channel {channel}.")
    except Exception as e:
        log.warning(f"Could not join channel {channel}: {e}")


def slack_get_channel_messages(channel, oldest=None, limit=20):
    kwargs = {"channel": channel, "limit": limit}
    if oldest:
        kwargs["oldest"] = oldest
    return slack_api("conversations.history", **kwargs)


def slack_get_thread_replies(channel, ts, limit=100):
    data = slack_api("conversations.replies", channel=channel, ts=ts, limit=limit)
    return data.get("messages", [])


def get_bot_user_id():
    try:
        data = slack_api("auth.test")
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


def generate_processing_message():
    system_prompt = """Generate a funny, creative message telling the user their question is being processed.
Use emojis. Make it 1-3 sentences. Be creative and different each time.
Output ONLY the message, nothing else."""
    seed = f"Seed: {datetime.now().isoformat()} {random.randint(1, 99999)}"
    return call_llm(system_prompt, seed, model_hint="flash")


def split_response(text, max_len=39000):
    chunks = []
    while len(text) > max_len:
        split_at = text.rfind("\n\n", 0, max_len)
        if split_at < max_len // 2:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
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
- kb_update: Update the Intercom knowledge base (help center articles) based on Notion product release pages. Triggered when the user provides Notion page URLs describing new features or product updates.
"""


def classify_skill(text):
    """Classify a Slack message into a skill name or 'none'."""
    system_prompt = f"""You route Slack messages to the correct skill handler.

{SKILL_DESCRIPTIONS}

Rules:
- If the message contains one or more Notion page URLs (notion.so or notion.site) and talks about features released, product updates, or knowledge base updates → return "kb_update"
- If the message is a system notification, casual chat, or doesn't match any skill → return "none"

Reply with ONLY the skill name or "none". Nothing else."""

    result = call_llm(system_prompt, f"Message: {text}", model_hint="flash")
    return result.strip().lower().split()[0]  # Take first word only


def classify_followup(thread_context, followup_text):
    """Determine if a follow-up is an approval, a correction, or a new request."""
    system_prompt = """You are evaluating a follow-up message in a Slack thread where the bot proposed knowledge base changes.

Classify the follow-up into one of:
- "approve_all" — The user approves all proposed changes (e.g., "yes proceed", "go ahead", "looks good, do it")
- "approve_partial" — The user approves only some changes or gives specific instructions (e.g., "only update article X", "skip the new article", "proceed with 1 and 3 only")
- "reject" — The user rejects the changes (e.g., "no don't do that", "cancel", "let me rethink")
- "question" — The user is asking a clarifying question (e.g., "what about article Y?", "can you show me the full content?")

Reply with ONLY one of: approve_all, approve_partial, reject, question"""

    result = call_llm(system_prompt, f"Thread so far:\n{thread_context}\n\nFollow-up message:\n{followup_text}", model_hint="flash")
    return result.strip().lower()


# ---------------------------------------------------------------------------
# Main polling loop
# ---------------------------------------------------------------------------

def run_slack_poll():
    """Poll #product-oracle for new messages and follow-ups."""
    processed = load_processed_messages()
    slack_join_channel(ORACLE_CHANNEL_ID)
    bot_user_id = get_bot_user_id()
    if bot_user_id:
        log.info(f"Bot user ID: {bot_user_id}")

    ten_min_ago = str(time.time() - 600)
    log.info(f"Polling #{ORACLE_CHANNEL_ID} for new messages...")

    try:
        data = slack_get_channel_messages(ORACLE_CHANNEL_ID, oldest=ten_min_ago, limit=20)
    except Exception as e:
        log.error(f"Failed to fetch Slack messages: {e}")
        return

    messages = data.get("messages", [])
    log.info(f"Found {len(messages)} message(s) in the last 10 minutes.")

    # --- Top-level messages ---
    new_queries = []
    for msg in messages:
        ts = msg.get("ts", "")
        if ts in processed:
            continue
        if msg.get("bot_id") or msg.get("subtype"):
            continue
        if msg.get("thread_ts") and msg.get("thread_ts") != ts:
            continue
        text = msg.get("text", "").strip()
        if not text:
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
            processed[ts] = {"status": "error", "error": str(e), "date": datetime.now().isoformat()}
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
        processed[ts] = {
            "status": "awaiting_approval",
            "skill": skill,
            "query": text[:500],
            "response_length": len(response),
            "date": datetime.now().isoformat(),
        }
        save_processed_messages(processed)

    # --- Follow-up scanning: check threads awaiting approval ---
    log.info("Scanning threads for follow-up responses...")

    now = datetime.now()
    active_threads = []
    for thread_ts, entry in processed.items():
        if entry.get("status") not in ("awaiting_approval",):
            continue
        if ":" in thread_ts:
            continue
        try:
            entry_date = datetime.fromisoformat(entry["date"])
            if (now - entry_date).total_seconds() < 86400:
                active_threads.append((thread_ts, entry))
        except (KeyError, ValueError):
            continue

    log.info(f"Found {len(active_threads)} thread(s) awaiting approval.")

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

            # Classify the follow-up
            try:
                classification = classify_followup(thread_context, followup_text)
            except Exception as e:
                log.error(f"  Follow-up classification failed: {e}")
                processed[followup_key] = {"status": "error", "error": str(e), "date": now.isoformat()}
                save_processed_messages(processed)
                continue

            skill = entry.get("skill", "")

            if classification.startswith("approve"):
                log.info(f"  Approval detected ({classification}), executing changes...")
                try:
                    if skill == "kb_update":
                        from skills.kb_update import execute_approved_changes
                        response = execute_approved_changes(
                            entry.get("query", ""),
                            followup_text,
                            classification,
                            thread_context,
                            call_llm,
                        )
                    else:
                        response = "Skill execution not available."
                except Exception as e:
                    log.error(f"  Execution failed: {e}")
                    response = f"Something went wrong while applying changes: {e}"

                processed[thread_ts]["status"] = "completed"

            elif classification.startswith("reject"):
                response = "Got it, changes cancelled. Let me know if you'd like to try again with different instructions."
                processed[thread_ts]["status"] = "rejected"

            else:
                # question or other — answer from context
                try:
                    response = call_llm(
                        """You are the Fourwaves Oracle. Answer the user's follow-up question based on the thread context.
Use Slack mrkdwn: single * for bold, > for quotes. NEVER use ** or #.""",
                        f"Thread:\n{thread_context}\n\nFollow-up:\n{followup_text}",
                        model_hint="pro",
                    )
                except Exception as e:
                    response = f"Error generating response: {e}"

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
            save_processed_messages(processed)

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
