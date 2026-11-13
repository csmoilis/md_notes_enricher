"""Streamlit UI: streamlit run app.py"""
from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from enricher import (DEFAULT_DELAY, DEFAULT_MAX_WORDS, DEFAULT_MIN_WORDS, DEFAULT_MODEL,
                      DailyQuotaExceeded, Enricher, Settings, load_api_key, make_client)

OUT_DIR = Path(__file__).parent / "outputs"

st.set_page_config(page_title="MD Notes Enricher", page_icon="📝", layout="wide")
st.title("📝 MD Notes Enricher")
st.caption("Global context → semantic chunks → Gemini enrichment → enriched .md")

# ---------------------------------------------------------------- sidebar: key + settings
with st.sidebar:
    st.header("API key")
    env_key = load_api_key()
    if env_key:
        st.success(f"GEMINI_API_KEY found (…{env_key[-4:]})")
        typed_key = ""
    else:
        st.warning("No GEMINI_API_KEY in environment or .env")
        typed_key = st.text_input("Paste a key (used only for this session)", type="password")
        st.markdown("[Get a free key](https://aistudio.google.com/apikey)")
    api_key = env_key or typed_key

    st.header("Settings")
    model = st.text_input("Model", DEFAULT_MODEL,
                          help="e.g. gemini-3.8-flash, gemini-3.5-flash-lite (bigger free daily quota)")
    delay = st.number_input("Seconds between requests", 0.0, 60.0, DEFAULT_DELAY, 0.5,
                            help="5s keeps you under 12 requests/minute")
    max_words = st.number_input("Max words per chunk", 200, 6000, DEFAULT_MAX_WORDS, 100)
    min_words = st.number_input("Target min words per chunk", 0, 6000, DEFAULT_MIN_WORDS, 100)
    language = st.text_input("Language of additions (blank = same as notes)", "")
    include_summary = st.checkbox("Put the global summary at the top of the output", False)

settings = Settings(model=model.strip() or DEFAULT_MODEL, delay=float(delay),
                    max_words=int(max_words), min_words=int(min_words),
                    language=language.strip() or None, include_summary=include_summary)

# ---------------------------------------------------------------- input
uploaded = st.file_uploader("Upload your study notes (.md)", type=["md", "markdown", "txt"])
if not uploaded:
    st.info("Upload a Markdown file to start.")
    st.stop()

text = uploaded.getvalue().decode("utf-8", errors="replace")
stem = re.sub(r"[^\w\-]+", "_", Path(uploaded.name).stem) or "notes"
out_path = OUT_DIR / f"{stem}_enriched.md"
chunks = Enricher(None, settings).chunks(text)

c1, c2, c3 = st.columns(3)
c1.metric("Words", f"{len(text.split()):,}")
c2.metric("Chunks", len(chunks))
c3.metric("API requests", len(chunks) + 1)
with st.expander("Chunk plan"):
    for c in chunks:
        st.write(f"**{c.index + 1}.** {c.label} — {c.words} words")

state_file = out_path.with_name(out_path.name + ".progress.json")
can_resume = state_file.exists() and out_path.exists()
b1, b2 = st.columns(2)
start = b1.button("🚀 Enrich notes", type="primary", disabled=not api_key)
resume = b2.button("⏯ Resume previous run", disabled=not (api_key and can_resume))

# ---------------------------------------------------------------- run
if start or resume:
    bar = st.progress(0.0, text="Building global context…")
    status = st.empty()
    log_box = st.expander("Log", expanded=False)
    preview = st.container()

    def log(msg: str):
        log_box.text(msg)

    def progress(done, total, chunk, payload):
        bar.progress(done / max(total, 1), text=f"Chunk {done}/{total}")
        if chunk is None:
            with preview.expander("Global context", expanded=False):
                st.markdown(payload)
        else:
            status.info(f"Enriched: {chunk.label}")

    try:
        enricher = Enricher(make_client(api_key), settings, log=log)
        enricher.run(text, out_path, resume=bool(resume), progress=progress)
        bar.progress(1.0, text="Done ✅")
        status.success(f"Done! Saved to {out_path}")
    except DailyQuotaExceeded as e:
        status.error(f"{e}\n\nPartial output kept — press **Resume** later.")
    except Exception as e:  # noqa: BLE001
        status.error(f"Error: {e}\n\nPartial output kept — press **Resume** to continue.")

# ---------------------------------------------------------------- result
if out_path.exists() and out_path.stat().st_size:
    result = out_path.read_text(encoding="utf-8")
    st.subheader("Enriched notes")
    st.download_button("⬇️ Download enriched .md", result, file_name=out_path.name,
                       mime="text/markdown")
    tab1, tab2 = st.tabs(["Preview", "Raw Markdown"])
    with tab1:
        st.markdown(result)
    with tab2:
        st.code(result, language="markdown")
