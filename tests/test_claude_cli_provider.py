"""Smoke tests for the claude_cli provider.

The provider shells out to the ``claude`` CLI so we can't easily run a true
end-to-end test in CI (no Claude Code login on a typical CI runner, no
guarantee the CLI is installed). These tests cover everything we *can*
check without invoking the binary:

  - Model-name parsing for ``claude-cli/<model>``
  - Resolver returns provider="claude_cli" and the unprefixed model name
  - get_client_llm returns None for this provider
  - _build_argv composes the expected flags from kwargs and env defaults
  - Mocked subprocess returns are converted into a QueryResult correctly
  - Error paths (missing binary, broken JSON, error-typed result event)

The real end-to-end (actually invoking ``claude -p`` against a logged-in
user) requires interactive Claude Code OAuth and is deferred to manual /
integration testing.
"""
import json

import pytest

from shinka.llm.client import get_client_llm
from shinka.llm.providers import claude_cli as ccp
from shinka.llm.providers.claude_cli import (
    ClaudeCliError,
    _build_argv,
    _parse_json_response,
    _resolve_effort,
    _resolve_fallback_model,
    _result_to_query,
    _strip_prefix,
    query_claude_cli,
)
from shinka.llm.providers.model_resolver import resolve_model_backend


# ---------------------------------------------------------------------------
# Model name parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("claude-cli/sonnet",              "sonnet"),
    ("claude-cli/opus",                "opus"),
    ("claude-cli/haiku",               "haiku"),
    ("claude-cli/claude-sonnet-4-6",   "claude-sonnet-4-6"),
])
def test_strip_prefix(name, expected):
    assert _strip_prefix(name) == expected


def test_strip_prefix_passthrough_on_unprefixed():
    assert _strip_prefix("sonnet") == "sonnet"


# ---------------------------------------------------------------------------
# Resolver / client construction
# ---------------------------------------------------------------------------

def test_resolver_recognizes_claude_cli():
    resolved = resolve_model_backend("claude-cli/sonnet")
    assert resolved.provider == "claude_cli"
    assert resolved.api_model_name == "sonnet"


def test_resolver_rejects_empty_model():
    with pytest.raises(ValueError, match="missing after 'claude-cli/'"):
        resolve_model_backend("claude-cli/")


def test_get_client_llm_returns_none_for_claude_cli():
    client, name, provider = get_client_llm("claude-cli/haiku")
    assert client is None
    assert provider == "claude_cli"
    assert name == "haiku"


# ---------------------------------------------------------------------------
# Effort / fallback resolution (config precedence)
# ---------------------------------------------------------------------------

def test_effort_from_kwargs_overrides_env():
    assert _resolve_effort({"effort": "high"}) == "high"


def test_effort_reasoning_effort_alias():
    assert _resolve_effort({"reasoning_effort": "max"}) == "max"


def test_effort_kwarg_takes_priority_over_alias():
    assert _resolve_effort({"effort": "low", "reasoning_effort": "high"}) == "low"


def test_effort_falls_back_to_module_default(monkeypatch):
    monkeypatch.setattr(ccp, "CLAUDE_EFFORT", "medium")
    assert _resolve_effort({}) == "medium"


def test_fallback_model_from_kwargs(monkeypatch):
    monkeypatch.setattr(ccp, "CLAUDE_FALLBACK_MODEL", "")
    assert _resolve_fallback_model({"fallback_model": "haiku"}) == "haiku"


def test_fallback_model_from_env(monkeypatch):
    monkeypatch.setattr(ccp, "CLAUDE_FALLBACK_MODEL", "haiku")
    assert _resolve_fallback_model({}) == "haiku"


# ---------------------------------------------------------------------------
# argv composition
# ---------------------------------------------------------------------------

def test_build_argv_minimal(monkeypatch):
    monkeypatch.setattr(ccp, "CLAUDE_EFFORT", "disabled")
    monkeypatch.setattr(ccp, "CLAUDE_FALLBACK_MODEL", "")
    monkeypatch.setattr(ccp, "CLAUDE_MAX_BUDGET_USD", "")
    argv = _build_argv("sonnet", "be terse", {})
    # Mandatory flags appear in expected positions.
    assert argv[1] == "-p"
    assert "--tools" in argv and "" in argv
    assert "--output-format" in argv and "json" in argv
    assert "--no-session-persistence" in argv
    assert argv[argv.index("--model") + 1] == "sonnet"
    assert argv[argv.index("--system-prompt") + 1] == "be terse"
    # Effort disabled -> no --effort flag.
    assert "--effort" not in argv
    assert "--fallback-model" not in argv
    assert "--max-budget-usd" not in argv


def test_build_argv_with_effort(monkeypatch):
    monkeypatch.setattr(ccp, "CLAUDE_FALLBACK_MODEL", "")
    monkeypatch.setattr(ccp, "CLAUDE_MAX_BUDGET_USD", "")
    argv = _build_argv("sonnet", "x", {"effort": "high"})
    assert argv[argv.index("--effort") + 1] == "high"


def test_build_argv_with_fallback_model_and_budget(monkeypatch):
    monkeypatch.setattr(ccp, "CLAUDE_EFFORT", "low")
    monkeypatch.setattr(ccp, "CLAUDE_MAX_BUDGET_USD", "0.50")
    argv = _build_argv("opus", "x", {"fallback_model": "sonnet"})
    assert argv[argv.index("--fallback-model") + 1] == "sonnet"
    assert argv[argv.index("--max-budget-usd") + 1] == "0.50"


# ---------------------------------------------------------------------------
# JSON response parsing
# ---------------------------------------------------------------------------

def _result_event(content="hi", **extras):
    base = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": content,
        "total_cost_usd": 0.012,
        "usage": {
            "input_tokens": 3,
            "output_tokens": 2,
            "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 50,
        },
    }
    base.update(extras)
    return base


def test_parse_json_response_success():
    events = [{"type": "system"}, _result_event(content="hi there")]
    r = _parse_json_response(json.dumps(events))
    assert r["result"] == "hi there"


def test_parse_json_response_no_result_event():
    events = [{"type": "system"}]  # no result event
    with pytest.raises(ClaudeCliError, match="No result event"):
        _parse_json_response(json.dumps(events))


def test_parse_json_response_error_event():
    events = [_result_event(content="x", is_error=True,
                            api_error_status="rate_limited")]
    with pytest.raises(ClaudeCliError, match="rate_limited"):
        _parse_json_response(json.dumps(events))


def test_parse_json_response_bad_json():
    with pytest.raises(ClaudeCliError, match="Failed to parse"):
        _parse_json_response("not json")


def test_parse_json_response_not_a_list():
    with pytest.raises(ClaudeCliError, match="Unexpected JSON shape"):
        _parse_json_response('{"not": "a list"}')


# ---------------------------------------------------------------------------
# Result → QueryResult conversion
# ---------------------------------------------------------------------------

def test_result_to_query_shape():
    qr = _result_to_query(
        result_event=_result_event(content="the answer"),
        msg="ask",
        system_msg="be helpful",
        model_name="claude-cli/sonnet",
        new_msg_history=[{"role": "user",
                          "content": [{"type": "text", "text": "ask"}]}],
        kwargs={},
        model_posteriors=None,
    )
    assert qr.content == "the answer"
    assert qr.model_name == "claude-cli/sonnet"
    assert qr.new_msg_history[-1]["role"] == "assistant"
    assert qr.input_tokens == 153  # 3 + 100 + 50
    assert qr.output_tokens == 2
    assert qr.cost == 0.012


# ---------------------------------------------------------------------------
# End-to-end with mocked subprocess
# ---------------------------------------------------------------------------

class _FakeCompletedProcess:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_query_claude_cli_success(monkeypatch):
    monkeypatch.setattr(
        "shinka.llm.providers.claude_cli.shutil.which",
        lambda _bin: "/usr/local/bin/claude",
    )
    canned = [{"type": "system"}, _result_event(content="Hello from Claude!")]
    monkeypatch.setattr(
        "shinka.llm.providers.claude_cli.subprocess.run",
        lambda *_a, **_kw: _FakeCompletedProcess(stdout=json.dumps(canned)),
    )

    result = query_claude_cli(
        client=None,
        model="claude-cli/sonnet",
        msg="say hi",
        system_msg="be terse",
        msg_history=[],
        output_model=None,
    )
    assert result.content == "Hello from Claude!"
    assert result.model_name == "claude-cli/sonnet"
    assert result.new_msg_history[-1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# Error paths (tested directly to avoid the @backoff retry storm)
# ---------------------------------------------------------------------------

def test_ensure_cli_missing(monkeypatch):
    monkeypatch.setattr(
        "shinka.llm.providers.claude_cli.shutil.which", lambda _bin: None)
    with pytest.raises(ClaudeCliError, match="not found on PATH"):
        ccp._ensure_cli()


def test_structured_output_not_supported():
    """output_model != None raises NotImplementedError before any subprocess."""
    # Unwrap both @backoff layers (TimeoutExpired outer, ClaudeCliError inner)
    # so we hit the real function body without the retry machinery.
    fn = query_claude_cli
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    with pytest.raises(NotImplementedError, match="Structured output"):
        fn(
            client=None,
            model="claude-cli/sonnet",
            msg="x",
            system_msg="",
            msg_history=[],
            output_model=object(),  # any non-None
        )


# ---------------------------------------------------------------------------
# Argv: positional user_msg gets a `--` separator so leading `-` is safe.
# ---------------------------------------------------------------------------

def test_query_claude_cli_inserts_argv_separator(monkeypatch):
    """A user_msg starting with `-` must be passed positionally, not parsed
    as a CLI flag. The provider inserts ``--`` to terminate option parsing."""
    captured = {}

    def fake_run(argv, *_a, **_kw):
        captured["argv"] = argv
        return _FakeCompletedProcess(
            stdout=json.dumps([_result_event(content="ok")]),
        )

    monkeypatch.setattr(
        "shinka.llm.providers.claude_cli.shutil.which",
        lambda _bin: "/usr/local/bin/claude",
    )
    monkeypatch.setattr(
        "shinka.llm.providers.claude_cli.subprocess.run", fake_run,
    )

    query_claude_cli(
        client=None,
        model="claude-cli/sonnet",
        msg="-this looks like a flag",
        system_msg="be terse",
        msg_history=[],
        output_model=None,
    )
    argv = captured["argv"]
    assert "--" in argv
    sep_idx = argv.index("--")
    # The user message is the only positional after `--`.
    assert argv[sep_idx + 1] == "-this looks like a flag"
    assert sep_idx == len(argv) - 2


def test_query_claude_cli_uses_cross_platform_tempdir(monkeypatch):
    """cwd must be tempfile.gettempdir() so the provider works on Windows
    (where '/tmp' doesn't exist)."""
    import tempfile as _tf
    captured = {}

    def fake_run(_argv, *_a, **kw):
        captured["cwd"] = kw.get("cwd")
        return _FakeCompletedProcess(
            stdout=json.dumps([_result_event(content="ok")]),
        )

    monkeypatch.setattr(
        "shinka.llm.providers.claude_cli.shutil.which",
        lambda _bin: "/usr/local/bin/claude",
    )
    monkeypatch.setattr(
        "shinka.llm.providers.claude_cli.subprocess.run", fake_run,
    )
    query_claude_cli(
        client=None,
        model="claude-cli/sonnet",
        msg="hi",
        system_msg="",
        msg_history=[],
        output_model=None,
    )
    assert captured["cwd"] == _tf.gettempdir()
    assert captured["cwd"] != "/tmp" or _tf.gettempdir() == "/tmp"
