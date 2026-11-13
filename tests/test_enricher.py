import json
import os
from types import SimpleNamespace

import pytest
from google.genai import errors

import enricher as E


def rate_limit_error(delay=None, per_day=False):
    details = []
    if delay is not None:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": f"{delay}s"})
    if per_day:
        details.append({"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]})
    return errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED",
                                               "details": details}})


# ------------------------------------------------------------------ backoff
def test_backoff_retries_then_succeeds():
    calls, sleeps = [], []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise rate_limit_error()
        return "ok"

    assert E.call_with_backoff(fn, sleep=sleeps.append, base_delay=2) == "ok"
    assert len(calls) == 3 and len(sleeps) == 2
    assert 2 <= sleeps[0] < 3 and 4 <= sleeps[1] < 5  # exponential + jitter


def test_backoff_honors_retry_delay():
    sleeps, n = [], []

    def fn():
        n.append(1)
        if len(n) == 1:
            raise rate_limit_error(delay=30)
        return 1

    E.call_with_backoff(fn, sleep=sleeps.append)
    assert sleeps[0] >= 31


def test_backoff_503_is_retried():
    n = []

    def fn():
        n.append(1)
        if len(n) == 1:
            raise errors.ServerError(503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}})
        return 1

    assert E.call_with_backoff(fn, sleep=lambda s: None) == 1


def test_backoff_gives_up_after_max_retries():
    def fn():
        raise rate_limit_error()

    sleeps = []
    with pytest.raises(errors.ClientError):
        E.call_with_backoff(fn, max_retries=3, sleep=sleeps.append)
    assert len(sleeps) == 3


def test_non_retryable_raises_immediately():
    sleeps = []

    def fn():
        raise errors.ClientError(400, {"error": {"code": 400, "message": "bad key", "status": "INVALID_ARGUMENT"}})

    with pytest.raises(errors.ClientError):
        E.call_with_backoff(fn, sleep=sleeps.append)
    assert sleeps == []


def test_daily_quota_stops_immediately():
    with pytest.raises(E.DailyQuotaExceeded):
        E.call_with_backoff(lambda: (_ for _ in ()).throw(rate_limit_error(delay=3600, per_day=True)),
                            sleep=lambda s: None)


# ------------------------------------------------------------------ end to end with a fake client
class FakeModels:
    def __init__(self, fail_first=0, fail_at_call=None):
        self.calls = []
        self.fail_first = fail_first
        self.fail_at_call = fail_at_call

    def generate_content(self, model, contents, config):
        self.calls.append(SimpleNamespace(model=model, contents=contents, config=config))
        n = len(self.calls)
        if n <= self.fail_first:
            raise rate_limit_error(delay=1)
        if self.fail_at_call == n:
            raise RuntimeError("boom")
        if config.system_instruction is None:
            return SimpleNamespace(text="GLOBAL SUMMARY: thermodynamics notes")
        chunk = contents.split("information:\n\n", 1)[1]
        return SimpleNamespace(text=f"```markdown\n{chunk}\n\n> Enriched definition.\n```")


def notes():
    return "Intro.\n\n" + "".join(f"## Topic {i}\n\n" + " ".join(["word"] * 600) + "\n\n" for i in range(6))


def make(models, **kw):
    s = E.Settings(model="test-model", delay=5, max_words=1300, min_words=1000, **kw)
    return E.Enricher(SimpleNamespace(models=models), s, sleep=lambda x: None, log=lambda m: None)


def test_pipeline_end_to_end(tmp_path):
    models = FakeModels(fail_first=1)
    out = tmp_path / "output.md"
    seen = []
    make(models).run(notes(), out, progress=lambda d, t, c, p: seen.append(d))
    text = out.read_text(encoding="utf-8")
    chunks = make(models).chunks(notes())
    assert len(models.calls) == 1 + 1 + len(chunks)  # 1 retry + summary + chunks
    # global context is passed as system_instruction to every chunk call
    for call in models.calls[2:]:
        assert "GLOBAL SUMMARY: thermodynamics notes" in call.config.system_instruction
        assert call.contents.startswith("Using the global context provided, enrich the following specific subsection")
        assert call.model == "test-model"
    for i in range(6):
        assert f"## Topic {i}" in text
    assert text.count("> Enriched definition.") == len(chunks)
    assert "```markdown" not in text  # wrapping fence stripped
    assert seen[-1] == len(chunks)
    assert text.index("## Topic 0") < text.index("## Topic 5")


def test_throttle_sleeps_between_requests(tmp_path):
    sleeps = []
    s = E.Settings(delay=5, max_words=1300, min_words=1000)
    enr = E.Enricher(SimpleNamespace(models=FakeModels()), s, sleep=sleeps.append, log=lambda m: None)
    enr.run(notes(), tmp_path / "o.md")
    assert sleeps and all(4 < x <= 5 for x in sleeps)
    assert len(sleeps) == len(enr.chunks(notes()))  # one wait before every call except the first


def test_resume_after_failure(tmp_path):
    out = tmp_path / "output.md"
    models = FakeModels(fail_at_call=3)  # summary ok, chunk1 ok, chunk2 crashes
    with pytest.raises(RuntimeError):
        make(models, max_retries=0).run(notes(), out)
    partial = out.read_text(encoding="utf-8")
    assert "## Topic 0" in partial and "## Topic 5" not in partial
    state = json.loads((tmp_path / "output.md.progress.json").read_text())
    assert state["done"] == 1

    models2 = FakeModels()
    make(models2).run(notes(), out, resume=True)
    # no new summary call; remaining chunks only
    assert all(c.config.system_instruction for c in models2.calls)
    final = out.read_text(encoding="utf-8")
    for i in range(6):
        assert final.count(f"## Topic {i}") == 1


def test_include_summary_and_language(tmp_path):
    models = FakeModels()
    out = tmp_path / "o.md"
    make(models, include_summary=True, language="Spanish").run(notes(), out)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Global context (generated)")
    assert "Spanish" in models.calls[1].config.system_instruction


def test_cli_dry_run(tmp_path, capsys):
    import enrich
    f = tmp_path / "n.md"
    f.write_text(notes(), encoding="utf-8")
    assert enrich.main([str(f), "--dry-run"]) == 0
    assert "chunk(s)" in capsys.readouterr().out


@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="no GEMINI_API_KEY for live test")
def test_live_tiny(tmp_path):
    out = tmp_path / "live.md"
    md = "# Biology\n\n## Cell\n\nThe cell is the basic unit of life. Mitochondria make ATP.\n"
    E.Enricher(E.make_client(), E.Settings(delay=1)).run(md, out)
    assert "Cell" in out.read_text(encoding="utf-8")
