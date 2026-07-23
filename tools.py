"""
tools.py  —  Two tools the AI agent can call.

VideoSearchTool   → SerpAPI YouTube Engine → returns top YouTube URL
TranscriptionTool → yt-dlp download → Gemini Files API → verbatim transcript
                    → saves to  transcripts/<video_id>.txt

Model fallback chain (tried in order until one succeeds):
    gemini-3.5-flash  →  gemini-3.5-flash-lite  →  gemini-3.1-flash-lite
"""

import os
import glob
import requests
from google import genai
from google.genai import types
import yt_dlp
from dotenv import load_dotenv

load_dotenv()

# Fallback chain — first available model wins
GEMINI_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

AUDIO_MIME = {
    "m4a":  "audio/mp4",
    "webm": "audio/webm",
    "opus": "audio/opus",
    "ogg":  "audio/ogg",
    "mp3":  "audio/mpeg",
    "wav":  "audio/wav",
    "mp4":  "audio/mp4",
}

TRANSCRIPTION_PROMPT = (
    "You are a professional transcription service. "
    "Transcribe every spoken word in this audio exactly as said. "
    "Include all dialogue, narration, and speech. "
    "Do NOT summarise. Do NOT add commentary. Do NOT add timestamps. "
    "Return ONLY the verbatim transcript text and nothing else."
)


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
    Full pipeline:
      1. Download best audio with yt-dlp  (no ffmpeg / no postprocessors)
      2. Upload audio bytes to Gemini Files API
      3. Generate verbatim transcript with model fallback chain
      4. Save transcript to  transcripts/<video_id>.txt
      5. Delete local audio + remote Gemini file

    Args:
        video_url: full YouTube URL

    Returns:
        {transcription, source_url, saved_path, model_used}  or  {error: str}
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        return {"error": "GEMINI_API_KEY is not set in .env"}

    audio_path    = None
    uploaded_file = None
    client        = genai.Client(api_key=gemini_key)

    try:
        # ── 1. Download audio ─────────────────────────────────────────────
        # Avoid DASH fragmented containers — they produce binary garbage when
        # passed to the Gemini Files API.  Non-DASH m4a → webm → anything.
        ydl_opts = {
            "format": (
                "bestaudio[ext=m4a][protocol!*=dash]"
                "/bestaudio[ext=m4a]"
                "/bestaudio[ext=webm]"
                "/bestaudio"
            ),
            "outtmpl":     "%(id)s.%(ext)s",
            "quiet":       True,
            "no_warnings": True,
            "noplaylist":  True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info     = ydl.extract_info(video_url, download=True)
            video_id = info["id"]

        candidates = glob.glob(f"{video_id}.*")
        if not candidates:
            return {"error": "yt-dlp completed but no audio file was found."}
        audio_path = candidates[0]

        # ── 2. Upload to Gemini Files API ─────────────────────────────────
        # Pass file PATH (str), not raw bytes.
        # Passing bytes caused the SDK to embed binary MP4 content in the
        # exception message — the b'\x00\x00\x00\x18ftypdash...' error.
        ext       = os.path.splitext(audio_path)[1].lower().lstrip(".")
        mime_type = AUDIO_MIME.get(ext, "audio/mp4")

        with open(audio_path, "rb") as fh:
            uploaded_file = client.files.upload(
                file   = fh,
                config = types.UploadFileConfig(mime_type=mime_type),
            )

        # ── 3. Transcribe — try each model until one succeeds ─────────────
        transcript_text = None
        model_used      = None
        last_error      = None

        for model_name in GEMINI_MODELS:
            try:
                response = client.models.generate_content(
                    model    = model_name,
                    contents = [
                        types.Content(parts=[
                            types.Part(file_data=types.FileData(
                                file_uri  = uploaded_file.uri,
                                mime_type = mime_type,
                            )),
                            types.Part(text=TRANSCRIPTION_PROMPT),
                        ])
                    ],
                    config = types.GenerateContentConfig(
                        temperature      = 0.1,
                        max_output_tokens = 8192,
                    ),
                )
                text = (response.text or "").strip()
                if text:
                    transcript_text = text
                    model_used      = model_name
                    break
            except Exception as exc:
                last_error = f"{model_name}: {exc}"
                continue

        if not transcript_text:
            return {"error": f"All Gemini models failed. Last error → {last_error}"}

        # ── 4. Save to transcripts/ ───────────────────────────────────────
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
            "model_used":    model_used,
        }

    except Exception as exc:
        return {"error": f"transcription_tool: {exc}"}

    finally:
        # ── 5. Clean up local audio and remote Gemini file ────────────────
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass
        if uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass