"""Command line: python enrich.py notes.md -o output.md"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from enricher import (DEFAULT_DELAY, DEFAULT_MAX_WORDS, DEFAULT_MIN_WORDS, DEFAULT_MODEL,
                      DailyQuotaExceeded, Enricher, Settings, make_client)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Enrich Markdown study notes with Gemini.")
    p.add_argument("input", help="input .md file")
    p.add_argument("-o", "--output", help="output .md file (default: <input>_enriched.md)")
    p.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"Gemini model (default {DEFAULT_MODEL})")
    p.add_argument("-d", "--delay", type=float, default=DEFAULT_DELAY,
                   help=f"seconds between requests (default {DEFAULT_DELAY}; 5s keeps you under 12 RPM)")
    p.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS, help="max words per chunk (default 2000)")
    p.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS, help="target min words per chunk (default 1000)")
    p.add_argument("-l", "--language", default=None,
                   help="language for the additions, e.g. English or Spanish (default: same as notes)")
    p.add_argument("--include-summary", action="store_true", help="put the global summary at the top of the output")
    p.add_argument("--resume", action="store_true", help="continue an interrupted run")
    p.add_argument("--retries", type=int, default=6, help="max retries per request on 429/503 (default 6)")
    p.add_argument("--dry-run", action="store_true", help="only show how the notes would be chunked (no API calls)")
    a = p.parse_args(argv)

    src = Path(a.input)
    if not src.exists():
        p.error(f"{src} not found")
    text = src.read_text(encoding="utf-8")
    out = Path(a.output) if a.output else src.with_name(src.stem + "_enriched.md")
    settings = Settings(model=a.model, delay=a.delay, max_words=a.max_words, min_words=a.min_words,
                        language=a.language, include_summary=a.include_summary, max_retries=a.retries)

    if a.dry_run:
        chunks = Enricher(None, settings).chunks(text)
        print(f"{len(chunks)} chunk(s), {len(chunks) + 1} API requests in total:")
        for c in chunks:
            print(f"  {c.index + 1:>3}. {c.words:>5} words  {c.label}")
        return 0

    try:
        client = make_client()
        Enricher(client, settings).run(text, out, resume=a.resume)
    except DailyQuotaExceeded as e:
        print(f"\nSTOPPED: {e}\nPartial output kept in {out}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(f"\nInterrupted. Partial output kept in {out}. Re-run with --resume to continue.",
              file=sys.stderr)
        return 130
    except Exception as e:  # noqa: BLE001
        print(f"\nERROR: {e}\nPartial output (if any) kept in {out}. Re-run with --resume to continue.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
