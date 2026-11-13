from chunker import chunk_markdown, count_words, split_sections


def words(n, w="word"):
    return " ".join([w] * n)


def test_preamble_and_headers():
    md = "Intro text here.\n\n# Title\n\nabc\n\n## A\n\naaa\n\n### A1\n\nbbb\n\n#### deep\n\nccc\n"
    secs = split_sections(md)
    assert [s.level for s in secs] == [0, 1, 2, 3]
    assert secs[0].title == "" and secs[0].text.startswith("Intro")
    assert secs[3].title == "A1" and "#### deep" in secs[3].text  # level 4 stays inside


def test_hash_inside_code_block_is_not_header():
    md = "## Real\n\n```python\n# a comment\n## not a header\n```\n\n~~~\n# also not\n~~~\n\n## Second\nx\n"
    secs = split_sections(md)
    assert [s.title for s in secs] == ["Real", "Second"]
    assert "# a comment" in secs[0].text


def test_hashtag_and_seven_hashes_not_headers():
    secs = split_sections("## H\n#hashtag\n####### seven\n")
    assert len(secs) == 1


def test_small_sections_are_merged():
    md = "".join(f"## S{i}\n\n{words(100)}\n\n" for i in range(10))
    chunks = chunk_markdown(md, max_words=2000, min_words=1000)
    assert len(chunks) == 1
    assert chunks[0].headings == [f"S{i}" for i in range(10)]


def test_chunks_respect_max_and_min():
    md = "".join(f"## S{i}\n\n{words(400)}\n\n" for i in range(10))  # ~4000 words
    chunks = chunk_markdown(md, max_words=2000, min_words=1000)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.words <= 2000
    assert all(c.words >= 1000 for c in chunks[:-1])


def test_nothing_is_lost_or_reordered():
    md = "pre\n\n" + "".join(f"## S{i}\n\n{words(300, f'w{i}')}\n\n### S{i}.1\n\n{words(250)}\n\n" for i in range(8))
    chunks = chunk_markdown(md, max_words=1500, min_words=800)
    joined = "".join(c.text for c in chunks)
    assert joined.split() == md.split()


def test_oversized_section_split_at_subheaders_then_paragraphs():
    big = "## Big\n\n" + "".join(f"#### Sub{i}\n\n{words(300)}\n\n{words(300)}\n\n" for i in range(6))
    chunks = chunk_markdown(big, max_words=1000, min_words=500)
    assert len(chunks) > 1
    assert all(c.words <= 1000 for c in chunks)
    assert chunks[0].text.startswith("## Big")
    assert chunks[1].continuation_of == "Big"
    assert "(continued)" in chunks[1].label
    assert "".join(c.text for c in chunks).split() == big.split()


def test_paragraph_never_cut_and_code_block_kept_whole():
    code = "```\n" + "\n".join(f"line {i}" for i in range(300)) + "\n\n# x\n```\n"
    md = "## Code\n\n" + words(500) + "\n\n" + code + "\n" + words(500) + "\n"
    chunks = chunk_markdown(md, max_words=600, min_words=100)
    assert sum(c.text.count("```") for c in chunks) == 2
    assert any(c.text.count("```") == 2 for c in chunks)  # both fences in same chunk


def test_crlf_and_empty():
    assert chunk_markdown("") == []
    c = chunk_markdown("## A\r\nhello\r\n")
    assert c[0].text == "## A\nhello\n"


def test_count_words():
    assert count_words("a b\n c") == 3
