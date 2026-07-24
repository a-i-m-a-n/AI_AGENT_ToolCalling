"""
agent.py  —  AI Agent powered by Groq / LLaMA 3.3-70B with two tools.

HOW TOOL CALLING WORKS (under the hood)
────────────────────────────────────────
Step 1  User message + JSON tool schemas → Groq LLM
Step 2  LLM returns tool_calls JSON  (it does NOT run code)
Step 3  OUR CODE reads the JSON, calls the real Python function
Step 4  Tool result appended as {"role": "tool"} message
Step 5  LLM called again with the full updated conversation
Step 6  Repeat until LLM replies with plain text (no tool_calls) → done
"""

import json
import os
from typing import Callable, Optional

from groq import Groq
from dotenv import load_dotenv
from tools import video_search_tool, transcription_tool

load_dotenv()

_groq   = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL   = "llama-3.3-70b-versatile"

# ─────────────────────────────────────────────────────────────────────────────
# Tool schemas  (what the LLM sees)
# ─────────────────────────────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "video_search_tool",
            "description": (
                "Search YouTube via SerpAPI and return the single top video: "
                "URL, title, channel, duration, view count."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "YouTube search query."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transcription_tool",
            "description": (
                "Transcribe a YouTube video with Gemini multimodal AI. "
                "Downloads audio via yt-dlp, uploads to Gemini Files API, "
                "returns the complete verbatim transcript and saves it to disk."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "video_url": {"type": "string", "description": "Full YouTube URL."}
                },
                "required": ["video_url"],
            },
        },
    },
]

TOOL_MAP: dict[str, Callable] = {
    "video_search_tool": video_search_tool,
    "transcription_tool": transcription_tool,
}

# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI agent that finds and transcribes YouTube videos.

MANDATORY SEQUENCE — no deviations allowed:
1. Call video_search_tool ONCE with the user's topic as the search query.
2. Take the "url" value from the result and call transcription_tool with it.
3. Take the "transcription" value from the tool result and return it as your reply — unchanged, verbatim.
4. On the very last line write:  Source: <video URL>

HARD RULES:
- NEVER call video_search_tool more than once.
- NEVER write or invent a transcript. Only return what transcription_tool gives you.
- NEVER summarise the video. Return the raw transcript text, unchanged.
- If a tool returns an "error" field, report the error and stop."""

# ─────────────────────────────────────────────────────────────────────────────
# Agent loop
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(
    user_query: str,
    status_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Run the agentic loop for user_query.

    Returns:
        response    — LLM final text
        video_info  — dict from video_search_tool
        transcript  — raw transcript string from transcription_tool
        saved_path  — path to the saved .txt file
        model_used  — which Gemini model succeeded
        steps       — list of {tool, args, result}
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_query},
    ]

    video_info:  dict      = {}
    transcript:  str       = ""
    saved_path:  str       = ""
    model_used:  str       = ""
    steps:       list[dict] = []
    search_count            = 0

    for step_num in range(1, 6):           # max 5 LLM calls

        if status_callback:
            status_callback(f"Agent thinking... (step {step_num})")

        # ── LLM call ──────────────────────────────────────────────────────
        resp = _groq.chat.completions.create(
            model      = MODEL,
            messages   = messages,
            tools      = TOOL_DEFINITIONS,
            tool_choice= "auto",
            max_tokens = 4096,
        )
        msg = resp.choices[0].message

        # Serialise to plain dict (keeps future messages list clean)
        msg_dict: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id":   tc.id,
                    "type": "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(msg_dict)

        # ── Done? ─────────────────────────────────────────────────────────
        if not msg.tool_calls:
            return {
                "response":   msg.content or "",
                "video_info": video_info,
                "transcript": transcript,
                "saved_path": saved_path,
                "model_used": model_used,
                "steps":      steps,
            }

        # ── Execute tool calls ─────────────────────────────────────────────
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)

            if status_callback:
                status_callback(f"Tool called: {fn_name}  |  {json.dumps(fn_args)}")

            # Hard block on repeated searches
            if fn_name == "video_search_tool":
                search_count += 1
                if search_count > 1:
                    result = {
                        "error": (
                            "video_search_tool already called once. "
                            f"Use transcription_tool with url: {video_info.get('url', '')}"
                        )
                    }
                    steps.append({"tool": fn_name, "args": fn_args, "result": result})
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": json.dumps(result)})
                    continue

            # HARD OVERRIDE: never trust the LLM to correctly copy the URL
            # from the search result into this call. Models frequently pass
            # a hallucinated placeholder (e.g. "{result of ...}.url") or a
            # mangled string instead of the real value. We already have the
            # real URL in video_info from step 1 — use that instead of
            # whatever the model typed.
            if fn_name == "transcription_tool":
                if not video_info.get("url"):
                    result = {"error": "transcription_tool called before a successful video_search_tool result."}
                    steps.append({"tool": fn_name, "args": fn_args, "result": result})
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": json.dumps(result)})
                    continue
                fn_args["video_url"] = video_info["url"]

            # Run the real function
            fn = TOOL_MAP.get(fn_name)
            result = fn(**fn_args) if fn else {"error": f"Unknown tool: {fn_name}"}
            steps.append({"tool": fn_name, "args": fn_args, "result": result})

            # Cache values for UI
            if fn_name == "video_search_tool" and "error" not in result:
                video_info = result
                if status_callback:
                    status_callback(
                        f"Video found: {result['title']}  [{result['channel']}]"
                    )

            elif fn_name == "transcription_tool" and "error" not in result:
                transcript = result.get("transcription", "")
                saved_path = result.get("saved_path", "")
                model_used = result.get("model_used", "")
                if status_callback:
                    status_callback(
                        f"Transcription done — {len(transcript):,} chars  "
                        f"| model: {model_used}  "
                        f"| saved: {saved_path}"
                    )

            elif "error" in result:
                if status_callback:
                    status_callback(f"Tool error: {result['error']}")

            # Feed result back to the LLM
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      json.dumps(result),
            })

    return {
        "response":   "Agent hit iteration limit.",
        "video_info": video_info,
        "transcript": transcript,
        "saved_path": saved_path,
        "model_used": model_used,
        "steps":      steps,
    }