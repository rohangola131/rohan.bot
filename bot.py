"""
=============================================================================
  NON-LIMITOR GROUP BOT  —  Production-ready Telegram Bot
  Stack: python-telegram-bot v20+, APScheduler, OpenAI-compatible API client
=============================================================================

HOW TO FIND YOUR GROUP'S INTEGER CHAT ID
─────────────────────────────────────────
Option A (easiest):
  1. Add @userinfobot to your group.
  2. It will immediately post the group's integer ID (e.g., -1001234567890).
  3. Copy that number, paste it into your .env as TARGET_GROUP_ID.
  4. Remove @userinfobot from the group.

Option B (using this bot):
  1. Add this bot to the group as an admin.
  2. Temporarily add a print(update.effective_chat.id) line to any handler.
  3. Send any message in the group, read your console output.
  4. Remove the debug line and restart.

Note: Group/supergroup IDs are always NEGATIVE integers.
      Example: TARGET_GROUP_ID=-1001234567890
=============================================================================
"""

import asyncio
import logging
import os
import random
from collections import deque
from functools import wraps

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatType
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ─────────────────────────────────────────────────────────────────────────────
#  BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()  # Reads .env from the current working directory

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  (loaded once at startup — fail fast if anything is missing)
# ─────────────────────────────────────────────────────────────────────────────

BOT_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"]
GITHUB_MODELS_API_KEY: str = os.environ["GITHUB_MODELS_API_KEY"]
TARGET_GROUP_ID: int = int(os.environ["TARGET_GROUP_ID"])  # negative integer
OWNER_USERNAME: str = "lost_herobrine"  # hardcoded, no leading @

# GitHub Models endpoint (OpenAI-compatible)
GITHUB_MODELS_BASE_URL = "https://models.inference.ai.azure.com"
GITHUB_MODELS_MODEL = "gpt-4o"  # change to any model you have access to

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL STATE
# ─────────────────────────────────────────────────────────────────────────────

# Controls whether the 5-minute question loop broadcasts or stays silent.
# Only @lost_herobrine can flip this via "start" / "stop" DMs.
interval_active: bool = False

# APScheduler instance — created once, jobs added/removed dynamically
scheduler = AsyncIOScheduler(timezone="UTC")

# We keep a reference to the PTB Application so the scheduler job can use it
_app: "Application | None" = None

# Question categories and simple in-memory history to avoid repetition
CATEGORIES = [
    "philosophy",
    "science",
    "psychology",
    "existentialism",
    "quantum",
    "consciousness",
    "ethics",
    "reality",
]

# last chosen category (avoids same category back-to-back)
last_category: str | None = None

# keep last N questions to avoid exact repeats during runtime
recent_questions: "deque[str]" = deque(maxlen=10)

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def is_owner(update: Update) -> bool:
    """Return True if the sender is @lost_herobrine (case-insensitive)."""
    username = (update.effective_user.username or "").lower().lstrip("@")
    return username == OWNER_USERNAME.lower()


def is_target_group(update: Update) -> bool:
    """Return True if the message is in the designated group."""
    return update.effective_chat.id == TARGET_GROUP_ID


def is_dm(update: Update) -> bool:
    """Return True if the message is a private/DM chat."""
    return update.effective_chat.type == ChatType.PRIVATE


async def call_ai(
    system_prompt: str,
    user_content: str,
    *,
    max_tokens: int = 400,
) -> str:
    """
    Call the GitHub Models (OpenAI-compatible) API asynchronously.

    Returns the model's reply as a plain string.
    Traps HTTP 429 (rate limit) and other errors gracefully — never crashes
    the bot; returns a user-facing error string instead.
    """
    headers = {
        "Authorization": f"Bearer {GITHUB_MODELS_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GITHUB_MODELS_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.85,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{GITHUB_MODELS_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "a moment")
                logger.warning("AI API rate-limited. Retry-After: %s", retry_after)
                return f"⚠️ AI is rate-limited right now. Try again in {retry_after}s."

            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

    except httpx.TimeoutException:
        logger.error("AI API request timed out.")
        return "⚠️ AI request timed out. Please try again."
    except httpx.HTTPStatusError as exc:
        logger.error("AI API HTTP error: %s", exc)
        return f"⚠️ AI API error ({exc.response.status_code}). Try later."
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected AI API error: %s", exc)
        return "⚠️ Something went wrong with the AI. Try again later."


# ─────────────────────────────────────────────────────────────────────────────
#  INTERVAL ENGINE  — 5-minute question broadcaster
# ─────────────────────────────────────────────────────────────────────────────

QUESTION_SYSTEM_PROMPT = """
You are the moderator of an edgy, intellectually sharp debate group on Telegram.
Your job: generate ONE short, punchy question that sparks debate or deep thought.
Topics rotate between: philosophy, science curiosity, human psychology, existentialism,
quantum weirdness, consciousness, ethics, or the nature of reality.

Language rules:
- Alternate naturally between simple English, Hindi, or fluid Hinglish.
- Keep it under 2 sentences. No emojis overload — 1 max.
- Make it feel like a real person from Delhi asking, not a textbook.
- Do NOT explain the question or add context. Just the question, raw.
""".strip()


async def broadcast_question() -> None:
    """
    Called by APScheduler every 5 minutes.
    Checks the toggle; if active, generates a question and sends it to the group.
    """
    global interval_active, _app

    if not interval_active:
        logger.info("Interval tick — broadcasting is paused (interval_active=False).")
        return

    if _app is None:
        logger.error("Interval tick — _app not set yet. Skipping.")
        return

    logger.info("Interval tick — generating question for group %d", TARGET_GROUP_ID)
    # Choose a category different from the last one to avoid repetition
    def choose_category() -> str:
        global last_category
        options = [c for c in CATEGORIES if c != last_category]
        if not options:
            options = CATEGORIES
        return random.choice(options)

    category = choose_category()

    # Build a focused user prompt that instructs the model about the category
    user_content = (
        f"Category: {category}\n"
        "Generate ONE short, punchy question that sparks debate or deep thought. "
        "Keep it under 2 sentences, use at most 1 emoji, and match the tone rules in the system prompt. "
        "Do NOT add explanation — return only the question text."
    )

    # Try a few times to avoid exact repeats from recent_questions
    question = ""
    max_tries = 4
    for attempt in range(max_tries):
        ai_reply = await call_ai(
            system_prompt=QUESTION_SYSTEM_PROMPT,
            user_content=user_content,
            max_tokens=120,
        )

        candidate = (ai_reply or "").strip()
        if not candidate:
            logger.warning("AI returned empty question on attempt %d", attempt + 1)
            continue

        if candidate in recent_questions:
            logger.info("AI produced recently-used question; retrying (attempt %d)", attempt + 1)
            # nudge model to rephrase
            user_content += "\nNote: Avoid repeating previous questions; rephrase strongly."
            continue

        # Accept candidate
        question = candidate
        break

    if not question:
        question = "⚠️ Couldn't generate a unique question right now. Try again later."

    # Update history and last category if we produced a real question
    if question and not question.startswith("⚠️"):
        recent_questions.append(question)
        last_category = category

    try:
        await _app.bot.send_message(chat_id=TARGET_GROUP_ID, text=question)
        logger.info("Question sent (category=%s): %s", category, question[:80])
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to send interval question: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
#  DM BLOCKER  (runs before any DM handler)
# ─────────────────────────────────────────────────────────────────────────────


async def dm_blocker(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Catches every private message NOT from the owner and immediately responds
    with the required rejection text. No further processing happens.
    """
    if not is_owner(update):
        await update.message.reply_text(
            "You don't own me buddy> you dot have permissions."
        )
        # Returning here stops PTB from calling further handlers in the chain.
        # The handler is registered with group=-1 so it runs first.
        return


# ─────────────────────────────────────────────────────────────────────────────
#  OWNER DM HANDLERS
# ─────────────────────────────────────────────────────────────────────────────


async def handle_owner_dm_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles plain text messages (not commands) from the owner in DMs.
    Recognises: "start" → enable interval, "stop" → disable interval.
    """
    global interval_active

    if not is_owner(update) or not is_dm(update):
        return  # Safety guard; should not reach here for non-owners due to blocker

    text = (update.message.text or "").strip().lower()

    if text == "start":
        interval_active = True
        logger.info("Owner started interval broadcasting.")
        await update.message.reply_text("Interval questions started.")

    elif text == "stop":
        interval_active = False
        logger.info("Owner stopped interval broadcasting.")
        await update.message.reply_text("Interval questions stopped.")

    # Any other DM text from owner is silently ignored (no reply).


async def handle_send_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /send <message>  — Owner DM only.
    Broadcasts the given text directly into TARGET_GROUP_ID.
    """
    if not is_owner(update) or not is_dm(update):
        return

    # context.args is a list of words after /send; join them back
    if not context.args:
        await update.message.reply_text(
            "Usage: /send <your message here>\nExample: /send Kal raat 9 baje debate session hai!"
        )
        return

    broadcast_text = " ".join(context.args)

    try:
        await context.bot.send_message(chat_id=TARGET_GROUP_ID, text=broadcast_text)
        await update.message.reply_text("✅ Message sent to the group.")
        logger.info("Owner broadcast: %s", broadcast_text[:100])
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to broadcast owner message: %s", exc)
        await update.message.reply_text(f"❌ Failed to send: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
#  GROUP VERIFY HANDLER
# ─────────────────────────────────────────────────────────────────────────────

VERIFY_SYSTEM_PROMPT = """
You are a sharp, no-nonsense fact-checker and logical analyst embedded in a debate group.
When given a statement or argument:
1. Check it for scientific accuracy and factual correctness.
2. Identify any logical fallacies (name them explicitly).
3. Assess philosophical soundness if relevant.
4. Be direct, conversational, and snappy — like a well-read friend, not a professor.
5. CRITICAL: Match the language/tone of the input. If it's Hinglish, reply in Hinglish.
   If it's Hindi, reply in Hindi. If it's English, reply in English.
6. Keep the total response under 200 words.
""".strip()


async def handle_verify_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Triggered when a user in TARGET_GROUP_ID sends /verify as a REPLY
    to another message. The replied-to message's text is analysed by the AI.
    """
    # Only respond inside the exact target group
    if not is_target_group(update):
        return  # Silently ignore /verify in any other chat

    message = update.message

    # /verify must be a reply to another message
    if message.reply_to_message is None:
        await message.reply_text(
            "⚠️ Yaar, /verify ko kisi message ke reply mein bhejo. "
            "Akele /verify ka koi kaam nahi."
        )
        return

    replied_text = message.reply_to_message.text or message.reply_to_message.caption

    if not replied_text:
        await message.reply_text(
            "⚠️ Jis message ko verify karna hai usmein text hona chahiye. "
            "Images/stickers verify nahi ho sakte abhi."
        )
        return

    logger.info(
        "Verify requested in group %d for text: %s",
        TARGET_GROUP_ID,
        replied_text[:80],
    )

    # Send a "thinking" placeholder to acknowledge quickly
    thinking_msg = await message.reply_text("🔍 Analyzing...")

    ai_response = await call_ai(
        system_prompt=VERIFY_SYSTEM_PROMPT,
        user_content=replied_text,
        max_tokens=300,
    )

    # Edit the placeholder with the real response
    try:
        await thinking_msg.edit_text(ai_response)
    except Exception:
        # Fallback: send a fresh reply if edit fails
        await message.reply_text(ai_response)


# ─────────────────────────────────────────────────────────────────────────────
#  GROUP MESSAGE FILTER  — ignore everything else in groups
# ─────────────────────────────────────────────────────────────────────────────


async def ignore_group_messages(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Catch-all handler for group messages that aren't /verify.
    Does absolutely nothing — just prevents PTB from logging unhandled updates.
    """
    return


# ─────────────────────────────────────────────────────────────────────────────
#  APPLICATION SETUP & MAIN
# ─────────────────────────────────────────────────────────────────────────────


def build_application() -> Application:
    """Wire up all handlers and return a configured Application instance."""

    app = Application.builder().token(BOT_TOKEN).build()

    # ── DM BLOCKER (handler group -1 → runs before all other handlers) ──────
    # This intercepts ALL private messages from non-owners first.
    app.add_handler(
        MessageHandler(
            filters.ChatType.PRIVATE,
            dm_blocker,
        ),
        group=-1,
    )

    # ── OWNER DM: /send command ───────────────────────────────────────────────
    app.add_handler(
        CommandHandler("send", handle_send_command, filters=filters.ChatType.PRIVATE),
        group=0,
    )

    # ── OWNER DM: plain text "start" / "stop" ────────────────────────────────
    app.add_handler(
        MessageHandler(
            filters.ChatType.PRIVATE & filters.TEXT & ~filters.COMMAND,
            handle_owner_dm_text,
        ),
        group=0,
    )

    # ── GROUP: /verify (reply-based fact check) ───────────────────────────────
    # We use a custom filter: message must be in TARGET_GROUP_ID chat
    target_group_filter = filters.Chat(chat_id=TARGET_GROUP_ID)
    app.add_handler(
        CommandHandler(
            "verify",
            handle_verify_command,
            filters=target_group_filter,
        ),
        group=0,
    )

    # ── GROUP: swallow everything else silently ───────────────────────────────
    app.add_handler(
        MessageHandler(
            filters.ChatType.GROUPS & ~filters.COMMAND,
            ignore_group_messages,
        ),
        group=1,
    )

    return app


async def main() -> None:
    global _app

    app = build_application()
    _app = app  # Expose to the scheduler job

    # ── APScheduler: 5-minute interval job ───────────────────────────────────
    scheduler.add_job(
        broadcast_question,
        trigger="interval",
        minutes=5,
        id="question_loop",
        replace_existing=True,
        misfire_grace_time=30,  # seconds; fires even if slightly late
    )
    scheduler.start()
    logger.info("APScheduler started. 5-minute question loop registered (paused by default).")

    # ── Start polling ─────────────────────────────────────────────────────────
    logger.info("Bot starting. Owner: @%s | Target group: %d", OWNER_USERNAME, TARGET_GROUP_ID)

    async with app:
        await app.initialize()
        await app.start()
        await app.updater.start_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)

        logger.info("Bot is live. Press Ctrl+C to stop.")

        # Keep running until interrupted
        try:
            await asyncio.Event().wait()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Shutdown signal received.")
        finally:
            scheduler.shutdown(wait=False)
            await app.updater.stop()
            await app.stop()
            await app.shutdown()
            logger.info("Bot shut down cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
