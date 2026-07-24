"""
tools.py  —  Two tools the AI agent can call.

VideoSearchTool   → SerpAPI YouTube Engine            → returns top YouTube URL
TranscriptionTool → SerpAPI YouTube Transcript Engine  → verbatim captions
                    → saves to  transcripts/<video_id>.txt

No audio download, no yt-dlp, no Gemini. Pulls YouTube's own manual/auto
captions directly, so there's no per-video processing time and no risk of
the host's IP getting 403'd by YouTube (a common yt-dlp-on-cloud problem).
"""

import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

TRANSCRIPT_ENGINE = "youtube_video_transcript"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_video_id(video_url: str) -> str:
    """Pull the 11-char YouTube video ID out of any common URL shape."""
    patterns = [
        r"[?&]v=([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"/shorts/([A-Za-z0-9_-]{11})",
        r"/embed/([A-Za-z0-9_-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, video_url)
        if m:
            return m.group(1)
    return ""


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
    serpapi_key = os.getenv("SERPAPI_KEY")
    if not serpapi_key:
        return {"error": "SERPAPI_KEY is not set in .env"}

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

        v = results[0]          # top result only
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
    Pull YouTube's own transcript (manual captions, falling back to
    auto-generated) via SerpAPI's youtube_video_transcript engine, and save
    it to transcripts/<video_id>.txt.

    Args:
        video_url: full YouTube URL

    Returns:
        {transcription, source_url, saved_path, model_used}  or  {error: str}
    """
    serpapi_key = os.getenv("SERPAPI_KEY")
    if not serpapi_key:
        return {"error": "SERPAPI_KEY is not set in .env"}

    video_id = _extract_video_id(video_url)
    if not video_id:
        return {"error": f"Could not extract a YouTube video ID from: {video_url}"}

    try:
        resp = requests.get(
            "https://serpapi.com/search",
            params={
                "engine":  TRANSCRIPT_ENGINE,
                "v":       video_id,
                "api_key": serpapi_key,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            return {"error": f"SerpAPI transcript error: {data['error']}"}

        segments = data.get("transcript", [])
        if not segments:
            return {"error": "No captions available for this video (none manual or auto-generated)."}

        transcript_text = " ".join(
            seg.get("snippet", "").strip() for seg in segments if seg.get("snippet")
        ).strip()

        if not transcript_text:
            return {"error": "SerpAPI returned an empty transcript for this video."}

        # ── Save to transcripts/ ────────────────────────────────────────────
        os.makedirs("transcripts", exist_ok=True)
        saved_path = os.path.join("transcripts", f"{video_id}.txt")

        with open(saved_path, "w", encoding="utf-8") as fh:
            fh.write(f"Source: {video_url}\n")
            fh.write("=" * 60 + "\n\n")
            fh.write(transcript_text)
            fh.write("\n")

        return {
            "transcription": transcript_text,
            "source_url":    video_url,
            "saved_path":    saved_path,
            "model_used":    "youtube_captions (serpapi)",
        }

    except requests.RequestException as exc:
        return {"error": f"SerpAPI error: {exc}"}
    except Exception as exc:
        return {"error": f"transcription_tool: {exc}"}