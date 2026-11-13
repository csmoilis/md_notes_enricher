"""Core pipeline: global context -> semantic chunks -> enrich each chunk -> append to output.md."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

from chunker import Chunk, chunk_markdown

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_DELAY = 5.0          # seconds between requests -> stays under 12 requests/minute
DEFAULT_MAX_WORDS = 2000
DEFAULT_MIN_WORDS = 1000

SUMMARY_PROMPT = """You are helping a student enrich their study notes.
Read the ENTIRE document below and produce a "global context" brief in Markdown with:
1. **Subject & theme** - one or two sentences on what these notes are about.
2. **Summary** - a concise summary of the whole document (max ~250 words).
3. **Key points** - the most important concepts, definitions and ideas as a bullet list.
4. **How it fits together** - how the main concepts/definitions relate to each other.
Keep it under ~700 words.{language_line}

DOCUMENT:
\"\"\"
{document}
\"\"\""""

SYSTEM_TEMPLATE = """You are an expert tutor who enriches a student's study notes.
Below is the GLOBAL CONTEXT of the whole document the student is studying. Use it to
understand the overarching theme and how each subsection relates to the whole.

=== GLOBAL CONTEXT ===
{context}
=== END GLOBAL CONTEXT ===

Rules for every answer:
- Return ONLY Markdown (no preamble like "Here is...", no wrapping ``` fence).
- Keep ALL of the original content and keep the original headings exactly (same text, same # level).
- Add clear definitions for key terms, explanations, intuition, short examples and relevant
  subject information where helpful. Connect ideas to the rest of the document when relevant.
- You may add sub-headings only at a deeper level than the original headings.
- Do not invent facts; be accurate. Keep code blocks, formulas and tables intact.{language_line}"""

CHUNK_PROMPT = ("Using the global context provided, enrich the following specific subsection "
                "with definitions and relevant subject information:\n\n{chunk}")


# --------------------------------------------------------------------------- errors / retry
class DailyQuotaExceeded(RuntimeError):
    """The free-tier per-day quota is used up; retrying now will not help. Resume later."""


RETRYABLE_CODES = {429, 500, 502, 503, 504}
RETRYABLE_STATUS = {"RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED"}


def _error_code(exc: BaseException) -> Optional[int]:
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def _error_details(exc: BaseException) -> list:
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        details = details.get("error", details).get("details", [])
    return details if isinstance(details, list) else []


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if _error_code(exc) in RETRYABLE_CODES:
        return True
    status = str(getattr(exc, "status", "") or "")
    if status in RETRYABLE_STATUS:
        return True
    return "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc)[:20]


def retry_delay_from_error(exc: BaseException) -> Optional[float]:
    """Return the server-suggested retry delay in seconds (google.rpc.RetryInfo), if any."""
    for d in _error_details(exc):
        if isinstance(d, dict) and "RetryInfo" in str(d.get("@type", "")):
            m = re.match(r"^\s*([\d.]+)s\s*$", str(d.get("retryDelay", "")))
            if m:
                return float(m.group(1))
    m = re.search(r"retry in ([\d.]+)\s*s", str(exc), re.IGNORECASE)
    return float(m.group(1)) if m else None


def is_daily_quota(exc: BaseException) -> bool:
    for d in _error_details(exc):
        if isinstance(d, dict) and "QuotaFailure" in str(d.get("@type", "")):
            for v in d.get("violations", []) or []:
                if "PerDay" in str(v.get("quotaId", "")) or "per_day" in str(v.get("quotaMetric", "")).lower():
                    return True
    return False


def call_with_backoff(fn: Callable[[], object], *, max_retries: int = 6, base_delay: float = 2.0,
                      max_delay: float = 120.0, sleep: Callable[[float], None] = time.sleep,
                      on_retry: Optional[Callable[[int, float, BaseException], None]] = None):
    """Call fn(); on 429/503-type errors retry with exponential backoff + jitter.

    Honors the server's suggested retry delay when given. Raises DailyQuotaExceeded
    immediately if the per-day quota is exhausted (no point waiting hours).
    """
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we re-raise non-retryable errors
            if not is_retryable(exc):
                raise
            if is_daily_quota(exc):
                raise DailyQuotaExceeded(
                    "Gemini free-tier DAILY quota exhausted for this model. Run again tomorrow "
                    "with --resume, or switch to a model with a bigger free quota "
                    "(e.g. --model gemini-3.5-flash-lite).") from exc
            if attempt >= max_retries:
                raise
            delay = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0, 1)
            hinted = retry_delay_from_error(exc)
            if hinted is not None:
                delay = max(delay, min(hinted + 1, max_delay * 3))
            attempt += 1
            if on_retry:
                on_retry(attempt, delay, exc)
            sleep(delay)


# --------------------------------------------------------------------------- client
def load_api_key() -> Optional[str]:
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).with_name(".env"))
        load_dotenv()
    except ImportError:
        pass
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def make_client(api_key: Optional[str] = None):
    from google import genai
    key = api_key or load_api_key()
    if not key:
        raise RuntimeError("No API key. Set GEMINI_API_KEY in your environment or in a .env file.")
    return genai.Client(api_key=key)


def _strip_wrapping_fence(text: str) -> str:
    m = re.match(r"^\s*```(?:markdown|md)?\s*\n(.*)\n```\s*$", text, re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else text


def _language_line(language: Optional[str]) -> str:
    return f"\n- Write your additions in {language}." if language else \
        "\n- Write in the same language as the original notes."


# --------------------------------------------------------------------------- pipeline
@dataclass
class Settings:
    model: str = DEFAULT_MODEL
    delay: float = DEFAULT_DELAY
    max_words: int = DEFAULT_MAX_WORDS
    min_words: int = DEFAULT_MIN_WORDS
    language: Optional[str] = None
    include_summary: bool = False
    max_retries: int = 6
    temperature: float = 0.4


class Enricher:
    def __init__(self, client, settings: Settings | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 log: Callable[[str], None] = print):
        self.client = client
        self.s = settings or Settings()
        self.sleep = sleep
        self.log = log
        self._last_call: Optional[float] = None

    # -- low level
    def _throttle(self):
        if self._last_call is not None and self.s.delay > 0:
            wait = self.s.delay - (time.monotonic() - self._last_call)
            if wait > 0:
                self.sleep(wait)

    def _generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        from google.genai import types
        config = types.GenerateContentConfig(system_instruction=system_instruction,
                                             temperature=self.s.temperature)

        def once():
            self._throttle()
            try:
                return self.client.models.generate_content(model=self.s.model, contents=prompt,
                                                           config=config)
            finally:
                self._last_call = time.monotonic()

        def on_retry(n, delay, exc):
            self.log(f"  rate limited / unavailable ({_error_code(exc) or type(exc).__name__}); "
                     f"retry {n}/{self.s.max_retries} in {delay:.0f}s")

        resp = call_with_backoff(once, max_retries=self.s.max_retries, sleep=self.sleep,
                                 on_retry=on_retry)
        text = getattr(resp, "text", None)
        return _strip_wrapping_fence(text.strip()) if text else ""

    # -- steps
    def global_context(self, document: str) -> str:
        prompt = SUMMARY_PROMPT.format(document=document,
                                       language_line=_language_line(self.s.language))
        ctx = self._generate(prompt)
        if not ctx:
            raise RuntimeError("Gemini returned an empty global summary.")
        return ctx

    def system_instruction(self, context: str) -> str:
        return SYSTEM_TEMPLATE.format(context=context, language_line=_language_line(self.s.language))

    def enrich_chunk(self, chunk: Chunk, system_instruction: str) -> str:
        body = chunk.text.strip()
        if chunk.continuation_of:
            body = (f"(This subsection continues the section \"{chunk.continuation_of}\"; "
                    f"do not repeat that heading.)\n\n{body}")
        out = self._generate(CHUNK_PROMPT.format(chunk=body), system_instruction)
        if not out:
            self.log(f"  WARNING: empty response for chunk {chunk.index + 1}; keeping original text")
            return chunk.text.strip()
        return out

    def chunks(self, document: str) -> list[Chunk]:
        return chunk_markdown(document, max_words=self.s.max_words, min_words=self.s.min_words)

    # -- full run
    def run(self, input_text: str, output_path: str | Path, resume: bool = False,
            progress: Optional[Callable[[int, int, Chunk, str], None]] = None) -> Path:
        """Enrich input_text and append results to output_path chunk by chunk.

        Progress is saved in ``<output>.progress.json`` so an interrupted run can
        continue with resume=True (same input + chunk settings).
        """
        output_path = Path(output_path)
        state_path = output_path.with_name(output_path.name + ".progress.json")
        chunks = self.chunks(input_text)
        fingerprint = hashlib.sha256(
            f"{input_text}|{self.s.max_words}|{self.s.min_words}".encode()).hexdigest()

        state = None
        if resume and state_path.exists() and output_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                state = None
            if state and state.get("fingerprint") != fingerprint:
                self.log("Input or chunk settings changed since last run - starting over.")
                state = None
        if state:
            done = int(state.get("done", 0))
            context = state["context"]
            self.log(f"Resuming: {done}/{len(chunks)} chunks already done.")
        else:
            done = 0
            self.log(f"Step 1/3: building global context with {self.s.model} ...")
            context = self.global_context(input_text)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                if self.s.include_summary:
                    f.write("# Global context (generated)\n\n" + context.strip() + "\n\n---\n\n")
            self._save_state(state_path, fingerprint, context, 0, len(chunks))
        if progress:
            progress(done, len(chunks), None, context)

        system = self.system_instruction(context)
        self.log(f"Step 2/3: {len(chunks)} chunk(s) to enrich.")
        for chunk in chunks[done:]:
            self.log(f"  chunk {chunk.index + 1}/{len(chunks)}: {chunk.label} ({chunk.words} words)")
            enriched = self.enrich_chunk(chunk, system)
            with open(output_path, "a", encoding="utf-8") as f:
                if chunk.index > 0 or self.s.include_summary:
                    f.write("\n\n")
                f.write(enriched.strip() + "\n")
                f.flush()
                os.fsync(f.fileno())
            self._save_state(state_path, fingerprint, context, chunk.index + 1, len(chunks))
            if progress:
                progress(chunk.index + 1, len(chunks), chunk, enriched)
        self.log(f"Step 3/3: done -> {output_path}")
        return output_path

    @staticmethod
    def _save_state(path: Path, fingerprint: str, context: str, done: int, total: int):
        path.write_text(json.dumps({"fingerprint": fingerprint, "context": context,
                                    "done": done, "total": total}, indent=2), encoding="utf-8")
