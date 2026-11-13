"""Semantic chunking of Markdown notes.

Splits a Markdown document at headers (#, ##, ### by default) - never in the
middle of a paragraph, list, table or fenced code block - and then packs the
sections into chunks of roughly ``min_words``-``max_words`` words.

Rules:
* Lines starting with ``#`` inside fenced code blocks (``` or ~~~) are NOT
  treated as headers.
* Text before the first header becomes its own section (the "preamble").
* Small neighbouring sections are merged until a chunk reaches the target size.
* A section bigger than ``max_words`` is split at deeper headers (####...),
  then at blank-line paragraph boundaries (outside code blocks). A single
  paragraph is never cut, even if it is huge.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

HEADER_RE = re.compile(r"^ {0,3}(#{1,6})(?!#)(?:[ \t]+(.*?))?[ \t]*#*[ \t]*$")
FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def count_words(text: str) -> int:
    return len(text.split())


@dataclass
class Section:
    level: int          # 0 = preamble (text before the first header)
    title: str          # heading text without the #'s ("" for preamble)
    text: str           # full markdown of the section, heading line included

    @property
    def words(self) -> int:
        return count_words(self.text)


@dataclass
class Chunk:
    index: int
    text: str
    headings: list[str] = field(default_factory=list)
    continuation_of: str | None = None  # heading this chunk continues, if split mid-section

    @property
    def words(self) -> int:
        return count_words(self.text)

    @property
    def label(self) -> str:
        if self.continuation_of:
            return f"{self.continuation_of} (continued)"
        if self.headings:
            return self.headings[0]
        return "(intro)"


def _iter_lines_with_fence_state(lines: list[str]):
    """Yield (line, in_fence) where in_fence is True for lines inside/opening/closing a fence."""
    fence: str | None = None
    for line in lines:
        m = FENCE_RE.match(line)
        if fence is None:
            if m:
                fence = m.group(1)
                yield line, True
                continue
            yield line, False
        else:
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence) \
                    and line.strip() == m.group(1):
                fence = None
            yield line, True


def split_sections(text: str, max_level: int = 3) -> list[Section]:
    """Split markdown into sections at headers of level <= max_level."""
    lines = text.splitlines(keepends=True)
    sections: list[Section] = []
    cur_lines: list[str] = []
    cur_level, cur_title = 0, ""

    def flush():
        body = "".join(cur_lines)
        if body.strip():
            sections.append(Section(cur_level, cur_title, body))

    for line, in_fence in _iter_lines_with_fence_state(lines):
        m = None if in_fence else HEADER_RE.match(line.rstrip("\r\n"))
        if m and len(m.group(1)) <= max_level:
            flush()
            cur_lines = [line]
            cur_level, cur_title = len(m.group(1)), (m.group(2) or "").strip()
        else:
            cur_lines.append(line)
    flush()
    return sections


def _split_paragraphs(text: str) -> list[str]:
    """Split at blank lines that are outside fenced code blocks."""
    blocks: list[str] = []
    cur: list[str] = []
    for line, in_fence in _iter_lines_with_fence_state(text.splitlines(keepends=True)):
        cur.append(line)
        if not in_fence and not line.strip():
            blocks.append("".join(cur))
            cur = []
    if cur:
        blocks.append("".join(cur))
    return [b for b in blocks if b.strip()] or [text]


def _pack(pieces: list[str], max_words: int) -> list[str]:
    out: list[str] = []
    cur = ""
    for p in pieces:
        if cur and count_words(cur) + count_words(p) > max_words:
            out.append(cur)
            cur = p
        else:
            cur += p
    if cur:
        out.append(cur)
    return out


def _split_oversized(section: Section, max_words: int) -> list[str]:
    """Split a too-big section at deeper headers, then at paragraphs."""
    if section.words <= max_words:
        return [section.text]
    parts: list[str] = []
    for level in range(max(section.level, 3) + 1, 7):
        subs = split_sections(section.text, max_level=level)
        # first sub contains the section's own heading (+ intro text)
        if len(subs) > 1:
            parts = [s.text for s in subs]
            break
    if not parts:
        parts = [section.text]
    result: list[str] = []
    for p in _pack(parts, max_words):
        if count_words(p) > max_words:
            result.extend(_pack(_split_paragraphs(p), max_words))
        else:
            result.append(p)
    return result


def _headings_in(text: str) -> list[str]:
    return [s.title or "(intro)" for s in split_sections(text, max_level=6) if s.level > 0]


def chunk_markdown(text: str, max_words: int = 2000, min_words: int = 1000,
                   split_level: int = 3) -> list[Chunk]:
    """Return chunks of ~min_words..max_words words split at header boundaries.

    A new chunk is started when adding the next section would exceed
    ``max_words`` or when the current chunk already has ``min_words`` and the
    next section is a top-level topic (# or ##).
    """
    if max_words <= 0:
        raise ValueError("max_words must be > 0")
    min_words = min(min_words, max_words)
    text = text.replace("\r\n", "\n")
    sections = split_sections(text, max_level=split_level)

    # 1) break oversized sections into pieces (keeping track of their parent heading)
    pieces: list[tuple[Section, str, bool]] = []  # (section, text, is_continuation)
    for s in sections:
        for i, part in enumerate(_split_oversized(s, max_words)):
            pieces.append((s, part, i > 0))

    # 2) greedily pack pieces into chunks
    raw: list[tuple[str, str | None]] = []
    cur, cur_cont = "", None
    for sec, part, is_cont in pieces:
        if cur:
            cur_w, part_w = count_words(cur), count_words(part)
            too_big = cur_w + part_w > max_words
            new_topic = (not is_cont) and 1 <= sec.level <= 2 and cur_w >= min_words
            if too_big or new_topic:
                raw.append((cur, cur_cont))
                cur, cur_cont = "", None
        if not cur:
            cur_cont = sec.title if is_cont else None
        if cur and not cur.endswith("\n"):
            cur += "\n"
        cur += part
    if cur.strip():
        raw.append((cur, cur_cont))

    return [Chunk(index=i, text=t, headings=_headings_in(t), continuation_of=c)
            for i, (t, c) in enumerate(raw)]
