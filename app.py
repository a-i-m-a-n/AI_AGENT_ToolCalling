"""
app.py  —  Streamlit UI for the AI Video Search & Transcription Agent
"""

import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from agent import run_agent

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "AI Video Transcription Agent",
    page_icon  = "🎬",
    layout     = "wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .block-container { padding-top: 1.8rem; }

  .page-title {
    font-size: 2rem; font-weight: 800;
    background: linear-gradient(135deg,#6366f1,#8b5cf6,#a855f7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .page-sub { color:#94a3b8; font-size:.95rem; margin-bottom:1rem; }

  .video-card {
    background: linear-gradient(135deg,#1e1b4b,#312e81);
    border: 1px solid #4f46e5; border-radius:12px;
    padding: 1.1rem 1.4rem; margin-bottom:.8rem;
  }
  .video-card h3 { color:#c7d2fe; margin:0 0 .35rem; font-size:1rem; }
  .video-card p  { color:#94a3b8; margin:.12rem 0; font-size:.85rem; }
  .video-card a  { color:#818cf8; text-decoration:none; font-weight:600; }

  .transcript-box {
    background:#0f172a; border:1px solid #334155; border-radius:10px;
    padding:1.1rem 1.4rem; max-height:460px; overflow-y:auto;
    font-family:'Courier New',monospace; font-size:.88rem;
    color:#e2e8f0; line-height:1.75; white-space:pre-wrap;
  }

  .badge-ok  { color:#4ade80; font-size:.83rem; }
  .badge-err { color:#f87171; font-size:.83rem; }
  .meta-row  { color:#64748b; font-size:.8rem; margin-top:.4rem; }

  .workflow-box {
    background:#0f172a; border:1px solid #334155; border-radius:10px;
    padding:1rem 1.2rem; font-size:.8rem; color:#94a3b8;
    font-family:'Courier New',monospace; line-height:1.8;
  }
  hr { border-color:#1e293b; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def youtube_embed(url: str, height: int = 280) -> str:
    """Convert any YouTube URL to an embeddable iframe."""
    vid_id = ""
    for part in [url.split("v="), url.split("youtu.be/"), url.split("/shorts/")]:
        if len(part) > 1:
            vid_id = part[1].split("&")[0].split("?")[0]
            break
    if not vid_id:
        return ""
    return (
        f'<iframe width="100%" height="{height}" '
        f'src="https://www.youtube.com/embed/{vid_id}" '
        f'frameborder="0" allowfullscreen style="border-radius:10px;"></iframe>'
    )

def build_download_text(query: str, video_info: dict, transcript: str) -> str:
    url = video_info.get("url", "")
    sep = "=" * 60
    return (
        f"VIDEO TRANSCRIPTION\n{sep}\n"
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Query     : {query}\n"
        f"Title     : {video_info.get('title', 'N/A')}\n"
        f"Channel   : {video_info.get('channel', 'N/A')}\n"
        f"Source    : {url}\n"
        f"{sep}\n\n"
        f"{transcript}\n\n"
        f"{sep}\n"
        f"Source: {url}\n"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## AI Video Agent")
    st.caption("Search  →  Transcribe  →  Download")
    st.divider()

    # API key status
    st.markdown("### API Keys")
    KEYS = {
        "GROQ_API_KEY":   "Groq  (LLaMA agent)",
        "SERPAPI_KEY":    "SerpAPI  (search + transcript)",
    }
    all_ok = True
    for k, label in KEYS.items():
        if os.getenv(k):
            st.markdown(f'<p class="badge-ok">✓ {label}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="badge-err">✗ {label}  — missing</p>', unsafe_allow_html=True)
            all_ok = False

    if not all_ok:
        st.warning("Set missing keys in your .env file.")

    st.divider()

    # Workflow
    st.markdown("### Workflow")
    st.markdown("""
<div class="workflow-box">
User Prompt<br>
  ↓<br>
Agent (Groq / LLaMA)<br>
  ↓  1. search query<br>
VideoSearchTool<br>
  → SerpAPI YouTube Engine<br>
  → YouTube URL<br>
  ↓  2. video URL<br>
TranscriptionTool<br>
  → SerpAPI Transcript Engine<br>
  → YouTube captions<br>
  → transcript text<br>
  → transcripts/id.txt<br>
  ↓<br>
Agent Final Reply<br>
+ Video Embed in UI
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.caption("Stack: Groq · SerpAPI · Streamlit")

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<h1 class="page-title">AI Video Search & Transcription Agent</h1>',
            unsafe_allow_html=True)
st.markdown('<p class="page-sub">Groq (LLaMA 3.3-70B) · SerpAPI Search + Transcript</p>',
            unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Input
# ─────────────────────────────────────────────────────────────────────────────

user_query = st.text_area(
    "What video would you like to transcribe?",
    placeholder=(
        "e.g.  'Python tutorial for beginners'\n"
        "      'How do black holes form'\n"
        "      'Denzel Washington motivational speech'"
    ),
    height=100,
    key="user_query",
)

run_btn = st.button(
    "Search & Transcribe",
    type             = "primary",
    use_container_width = True,
    disabled         = not all_ok,
)

# ─────────────────────────────────────────────────────────────────────────────
# Run agent
# ─────────────────────────────────────────────────────────────────────────────

if run_btn:
    if not user_query.strip():
        st.warning("Enter a topic first.")
        st.stop()

    st.divider()

    # Live status panel
    with st.status("Agent running...", expanded=True) as status_box:
        result = run_agent(user_query.strip(), status_callback=st.write)

        if result.get("transcript"):
            status_box.update(label="Done.", state="complete", expanded=False)
        else:
            status_box.update(label="Finished with issues.", state="error", expanded=True)

    st.divider()

    # ── Layout: left = transcript | right = video + meta ──────────────────
    left, right = st.columns([3, 2], gap="large")

    transcript  = result.get("transcript", "")
    video_info  = result.get("video_info", {})
    saved_path  = result.get("saved_path", "")
    model_used  = result.get("model_used", "")
    video_url   = video_info.get("url", "")

    # ── RIGHT: video embed + metadata ─────────────────────────────────────
    with right:
        if video_info and "error" not in video_info:

            # YouTube embed
            iframe = youtube_embed(video_url)
            if iframe:
                st.markdown(iframe, unsafe_allow_html=True)
                st.markdown("")     # spacer

            # Metadata card
            st.markdown(f"""
<div class="video-card">
  <h3>{video_info.get('title', 'Unknown')}</h3>
  <p>Channel : {video_info.get('channel', 'N/A')}</p>
  <p>Duration: {video_info.get('duration', 'N/A')}</p>
  <p>Views   : {video_info.get('views', 'N/A')}</p>
  <p style="margin-top:.7rem;">
    <a href="{video_url}" target="_blank">Open on YouTube ↗</a>
  </p>
</div>
""", unsafe_allow_html=True)

            # Source line
            st.info(f"**Source:** {video_url}", icon="🔗")

            # Model used + saved path
            if model_used:
                st.markdown(
                    f'<p class="meta-row">Transcript source: {model_used}</p>',
                    unsafe_allow_html=True,
                )
            if saved_path:
                st.markdown(
                    f'<p class="meta-row">Saved to: {saved_path}</p>',
                    unsafe_allow_html=True,
                )

        # Tool call log
        steps = result.get("steps", [])
        if steps:
            st.markdown("#### Tool Call Log")
            for i, step in enumerate(steps, 1):
                with st.expander(f"Step {i} — {step['tool']}"):
                    st.markdown("**Args:**")
                    st.json(step["args"])
                    st.markdown("**Result:**")
                    display = dict(step["result"])
                    if "transcription" in display:
                        t = display["transcription"]
                        display["transcription"] = (
                            t[:400] + " … [truncated]" if len(t) > 400 else t
                        )
                    st.json(display)

    # ── LEFT: transcript ───────────────────────────────────────────────────
    with left:
        st.markdown("### Transcript")

        if transcript:
            # Scrollable monospace box
            st.markdown(
                f'<div class="transcript-box">{transcript}</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<p class="meta-row">'
                f'{len(transcript):,} characters &nbsp;·&nbsp; '
                f'{len(transcript.split()):,} words</p>',
                unsafe_allow_html=True,
            )

            # Download button — this is where the transcript goes into the file
            fname = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            st.download_button(
                label            = "Download Transcript (.txt)",
                data             = build_download_text(user_query, video_info, transcript),
                file_name        = fname,
                mime             = "text/plain",
                use_container_width = True,
            )

            # Note about knowledge-base file
            if saved_path:
                st.caption(f"Also saved locally at: {saved_path}")

        else:
            st.error(
                "No transcript returned. Check the Tool Call Log for the error details."
            )

    # Final LLM text (collapsible)
    final = result.get("response", "")
    if final:
        with st.expander("Full agent reply (raw LLM output)", expanded=False):
            st.markdown(final)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.caption("AI Video Agent · Groq LLaMA 3.3-70B · SerpAPI Search + Transcript · Streamlit")