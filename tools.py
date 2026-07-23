"""
tools.py  —  Two tools the AI agent can call.

VideoSearchTool   → SerpAPI YouTube Engine → returns top YouTube URL
TranscriptionTool → youtube-transcript-api (captions) → Gemini cleanup
                  → saves to  transcripts/<video_id>.txt

Why youtube-transcript-api instead of yt-dlp:
  yt-dlp gets HTTP 403 on cloud servers (Streamlit Cloud / AWS / GCP)
  because YouTube blocks audio downloads from datacenter IP ranges.
  youtube-transcript-api fetches built-in captions via plain HTTP —
  no download, no 403, works everywhere.

Gemini model fallback chain (tried in order until one succeeds):
    gemini-3.5-flash  →  gemini-3.5-flash-lite  →  gemini-3.1-flash-lite
"""

import os
import re
import requests
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# Secret helper — Streamlit Cloud first, then .env
# ─────────────────────────────────────────────────────────────────────────────

def _get_secret(key: str) -> str:
    try:
        import streamlit as st
        try:
            return st.secrets[key]
        except Exception:
            pass
    except ImportError:
        pass
    return os.getenv(key, "")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

CLEANUP_PROMPT = (
    "Below is a raw auto-generated caption transcript from a YouTube video. "
    "Clean it up into a readable, verbatim transcript. "
    "Fix obvious caption errors, run-on sentences, and missing punctuation. "
    "Do NOT summarise, paraphrase, or remove any content. "
    "Return ONLY the cleaned transcript text and nothing else.\n\n"
    "RAW CAPTIONS:\n{raw}"
)


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _extract_video_id(url: str) -> str:
    """Extract YouTube video ID from any URL format."""
    parsed = urlparse(url)
    if parsed.hostname in ("youtu.be",):
        return parsed.path.lstrip("/").split("?")[0]
    if parsed.hostname in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [""])[0]
        if parsed.path.startswith(("/shorts/", "/embed/", "/v/")):
            return parsed.path.split("/")[2]
    # Last resort: grab 11-char ID from anywhere in the URL
    match = re.search(r"[?&/]([a-zA-Z0-9_-]{11})(?:[?&/]|$)", url)
    return match.group(1) if match else ""


def _captions_to_text(entries) -> str:
    """
    Join caption entries into a single plain-text string.
    Handles both:
      - youtube-transcript-api v1.x: FetchedTranscriptSnippet dataclass (.text attr)
      - youtube-transcript-api v0.x: plain dict (["text"] key)
    """
    texts = []
    for e in entries:
        # v1.x dataclass
        if hasattr(e, "text"):
            t = e.text.strip()
        # v0.x dict fallback
        elif isinstance(e, dict):
            t = e.get("text", "").strip()
        else:
            continue
        if t:
            texts.append(t)
    return " ".join(texts)


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1 — Video Search
# ─────────────────────────────────────────────────────────────────────────────

def video_search_tool(query: str) -> dict:
    """
    Search YouTube with SerpAPI and return the single top result.

    Args:
        query: natural-language search string

    Returns:
        {url, title, channel, duration, views}  or  {error: str}
    """
    serpapi_key = _get_secret("SERPAPI_KEY")
    if not serpapi_key:
        return {"error": "SERPAPI_KEY is not set."}

    try:
        resp = requests.get(
            "https://serpapi.com/search",
            params={
                "engine":       "youtube",
                "search_query": query,
                "api_key":      serpapi_key,
            },
            timeout=20,
        )
        resp.raise_for_status()
        results = resp.json().get("video_results", [])

        if not results:
            return {"error": f"No YouTube results for: '{query}'"}

        v = results[0]
        return {
            "url":      v.get("link", ""),
            "title":    v.get("title", "Unknown"),
            "channel":  v.get("channel", {}).get("name", "Unknown"),
            "duration": v.get("length", "N/A"),
            "views":    v.get("views",  "N/A"),
        }

    except requests.RequestException as exc:
        return {"error": f"SerpAPI error: {exc}"}
    except Exception as exc:
        return {"error": f"video_search_tool: {exc}"}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2 — Transcription
# ─────────────────────────────────────────────────────────────────────────────

def transcription_tool(video_url: str) -> dict:
    """
    Pipeline (cloud-safe — no audio download):
      1. Extract video ID from URL
      2. Fetch captions with youtube-transcript-api
         (tries manual captions first, then auto-generated)
      3. Pass raw captions to Gemini for cleanup/formatting
         (model fallback: gemini-3.5-flash → lite → 3.1-flash-lite)
      4. Save cleaned transcript to transcripts/<video_id>.txt
      5. Return transcript text + metadata

    Args:
        video_url: full YouTube URL

    Returns:
        {transcription, source_url, saved_path, model_used}  or  {error: str}
    """
    gemini_key = _get_secret("GEMINI_API_KEY")
    if not gemini_key:
        return {"error": "GEMINI_API_KEY is not set."}

    # ── 1. Extract video ID ───────────────────────────────────────────────
    video_id = _extract_video_id(video_url)
    if not video_id:
        return {"error": f"Could not extract video ID from URL: {video_url}"}

    # ── 2. Fetch captions (youtube-transcript-api v1.0+ API) ─────────────
    # v1.0 changed from class methods to instance methods:
    #   OLD: YouTubeTranscriptApi.list_transcripts(id)
    #   NEW: YouTubeTranscriptApi().list(id)
    # Entries are now FetchedTranscriptSnippet dataclass objects with
    # a .text attribute, not plain dicts.
    try:
        ytt = YouTubeTranscriptApi()
        transcript_list = ytt.list(video_id)

        fetched = None
        try:
            # Manual English captions first
            t = transcript_list.find_manually_created_transcript(
                ["en", "en-US", "en-GB"]
            )
            fetched = t.fetch()
        except NoTranscriptFound:
            pass

        if fetched is None:
            try:
                # Auto-generated English captions
                t = transcript_list.find_generated_transcript(
                    ["en", "en-US", "en-GB"]
                )
                fetched = t.fetch()
            except NoTranscriptFound:
                pass

        if fetched is None:
            # Any language as last resort
            t = next(iter(transcript_list))
            fetched = t.fetch()

        if not fetched:
            return {"error": "Captions fetched but were empty."}

        raw_text = _captions_to_text(fetched)

    except TranscriptsDisabled:
        return {"error": "Transcripts are disabled for this video."}
    except Exception as exc:
        return {"error": f"Caption fetch failed: {exc}"}

    # ── 3. Gemini cleanup — try each model until one succeeds ─────────────
    client     = genai.Client(api_key=gemini_key)
    final_text = None
    model_used = None
    last_error = None

    for model_name in GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model    = model_name,
                contents = CLEANUP_PROMPT.format(raw=raw_text),
                config   = types.GenerateContentConfig(
                    temperature       = 0.1,
                    max_output_tokens = 8192,
                ),
            )
            text = (response.text or "").strip()
            if text:
                final_text = text
                model_used = model_name
                break
        except Exception as exc:
            last_error = f"{model_name}: {exc}"
            continue

    # If all Gemini models fail, still return the raw captions
    if not final_text:
        final_text = raw_text
        model_used = f"raw captions (Gemini failed: {last_error})"

    # ── 4. Save to transcripts/ ───────────────────────────────────────────
    os.makedirs("transcripts", exist_ok=True)
    saved_path = os.path.join("transcripts", f"{video_id}.txt")

    with open(saved_path, "w", encoding="utf-8") as fh:
        fh.write(f"Source: {video_url}\n")
        fh.write("=" * 60 + "\n\n")
        fh.write(final_text)
        fh.write("\n")

    return {
        "transcription": final_text,
        "source_url":    video_url,
        "saved_path":    saved_path,
        "model_used":    model_used,
    }