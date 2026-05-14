"""Claude Code CLI provider.

Routes ShinkaEvolve LLM calls through ``claude -p`` so they authenticate via
the user's Claude Code OAuth login (Pro/Max subscription) instead of an
``ANTHROPIC_API_KEY``. Usage counts against the same subscription quota as
interactive Claude Code, with no additional billing.

Cross-platform: works on native Linux, macOS, and Windows wherever the
``claude`` CLI binary is on PATH (or located via ``CLAUDE_CLI_BIN``).
Scratch ``cwd`` is set to ``tempfile.gettempdir()`` to keep Claude Code
from auto-discovering a project-local ``CLAUDE.md``.

Model name format::

    claude-cli/<alias or full name>

Examples::

    claude-cli/sonnet
    claude-cli/opus
    claude-cli/haiku
    claude-cli/claude-sonnet-4-6   (full names accepted too)

Configuration. Environment variables set defaults; kwargs override per call:

  CLAUDE_CLI_BIN              path to the ``claude`` binary (default: PATH)
  CLAUDE_CLI_TIMEOUT_SEC      per-call subprocess timeout (default 600)
  CLAUDE_CLI_EFFORT           default --effort: ``low``/``medium``/``high``/
                              ``xhigh``/``max``/``disabled`` (default low).
                              ``low`` disables extended thinking and makes
                              single calls 5-10x faster.
  CLAUDE_CLI_FALLBACK_MODEL   pass to ``--fallback-model`` if set; used by the
                              CLI when the primary model is overloaded.
  CLAUDE_CLI_MAX_BUDGET_USD   pass to ``--max-budget-usd`` if set; hard cap
                              on the call's reported cost.

Per-call ``**kwargs`` recognized:

  effort                      same values as CLAUDE_CLI_EFFORT
  reasoning_effort            alias for ``effort`` to match the kwarg naming
                              used by other providers
  fallback_model              same as CLAUDE_CLI_FALLBACK_MODEL

Notes:

  - ``msg_history`` is collapsed into a single user message because the CLI's
    ``-p`` mode is single-turn. ShinkaEvolve's mutation prompts are single-
    turn so this is fine in practice. Non-text content blocks (images, tool
    calls) inside ``msg_history`` are dropped silently — the CLI has no
    representation for them on the input side.
  - ``claude-cli/<model>`` is not in ``pricing.csv``, so ``is_reasoning_model``
    returns False and ShinkaEvolve's ``kwargs.py`` sampler does NOT plumb
    ``reasoning_efforts`` through to this provider. The CLAUDE_CLI_EFFORT
    env var is the operative knob; the per-call ``effort`` / ``reasoning_effort``
    kwarg is honoured only when callers pass it directly (not via
    ``LLMClient.batch_kwargs_query``'s sampling layer).
  - Costs reported by the CLI (``total_cost_usd``) reflect what the call
    would cost via the Anthropic API; actual billing for OAuth users is
    against the subscription quota, not per-call. We surface the CLI-reported
    number for visibility.
  - Streaming is not used; we wait for the full JSON result.
"""
import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile

import backoff

from shinka.llm.constants import BACKOFF_MAX_TIME, BACKOFF_MAX_TRIES, BACKOFF_MAX_VALUE
from .result import QueryResult

logger = logging.getLogger(__name__)


MAX_TRIES = BACKOFF_MAX_TRIES
MAX_VALUE = BACKOFF_MAX_VALUE
MAX_TIME = BACKOFF_MAX_TIME

# A run that times out is almost always going to time out again; cap retries
# so one hung CLI doesn't monopolize the caller for hours.
TIMEOUT_MAX_TRIES = 2
TIMEOUT_MAX_VALUE = 10

CLAUDE_BIN = os.environ.get("CLAUDE_CLI_BIN", "claude")
CLAUDE_TIMEOUT = int(os.environ.get("CLAUDE_CLI_TIMEOUT_SEC", "600"))
CLAUDE_EFFORT = os.environ.get("CLAUDE_CLI_EFFORT", "low")
CLAUDE_FALLBACK_MODEL = os.environ.get("CLAUDE_CLI_FALLBACK_MODEL", "")
CLAUDE_MAX_BUDGET_USD = os.environ.get("CLAUDE_CLI_MAX_BUDGET_USD", "")


class ClaudeCliError(RuntimeError):
    pass


def backoff_handler(details):
    exc = details.get("exception")
    if exc:
        logger.info(
            f"Claude CLI - Retry {details['tries']} due to error: {exc}. "
            f"Waiting {details['wait']:0.1f}s..."
        )


def _strip_prefix(model_name):
    """claude-cli/sonnet -> sonnet"""
    if model_name.startswith("claude-cli/"):
        return model_name.split("/", 1)[1]
    return model_name


def _flatten_history(msg, msg_history):
    """Collapse prior turns into a single user message for the CLI."""
    if not msg_history:
        return msg
    parts = []
    for turn in msg_history:
        role = turn.get("role", "user").upper()
        c = turn.get("content")
        if isinstance(c, list):
            text = "\n".join(b.get("text", "") for b in c if b.get("type") == "text")
        else:
            text = str(c) if c is not None else ""
        parts.append(f"--- prior {role} ---\n{text}")
    parts.append(f"--- current USER ---\n{msg}")
    return "\n\n".join(parts)


def _resolve_effort(kwargs):
    """Per-call effort, with ``reasoning_effort`` accepted as an alias.

    Falls back to ``CLAUDE_CLI_EFFORT``. Unknown values pass through to the
    CLI which will reject them with a clear error.
    """
    val = kwargs.get("effort") or kwargs.get("reasoning_effort") or CLAUDE_EFFORT
    if not val:
        return ""
    return str(val).strip().lower()


def _resolve_fallback_model(kwargs):
    return str(kwargs.get("fallback_model") or CLAUDE_FALLBACK_MODEL or "").strip()


def _build_argv(model, system_msg, kwargs):
    argv = [
        CLAUDE_BIN,
        "-p",
        "--tools", "",
        "--output-format", "json",
        "--no-session-persistence",
        "--model", model,
        "--system-prompt", system_msg,
    ]
    effort = _resolve_effort(kwargs)
    if effort and effort != "disabled":
        argv.extend(["--effort", effort])
    fallback = _resolve_fallback_model(kwargs)
    if fallback:
        argv.extend(["--fallback-model", fallback])
    if CLAUDE_MAX_BUDGET_USD:
        argv.extend(["--max-budget-usd", CLAUDE_MAX_BUDGET_USD])
    return argv


def _parse_json_response(stdout):
    """Parse the CLI's --output-format json blob and return the result event."""
    try:
        events = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ClaudeCliError(
            f"Failed to parse claude CLI JSON: {exc}. stdout head: {stdout[:400]}"
        )
    if not isinstance(events, list):
        raise ClaudeCliError(f"Unexpected JSON shape (not a list): {type(events)}")
    result_event = next((e for e in events if e.get("type") == "result"), None)
    if result_event is None:
        raise ClaudeCliError(
            f"No result event in CLI output. Events: {[e.get('type') for e in events]}"
        )
    if result_event.get("is_error"):
        err = result_event.get("api_error_status") or result_event.get("subtype") or "unknown"
        raise ClaudeCliError(f"Claude CLI returned error: {err}")
    return result_event


def _result_to_query(result_event, msg, system_msg, model_name, new_msg_history,
                     kwargs, model_posteriors):
    content = result_event.get("result", "")
    usage = result_event.get("usage", {}) or {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    cache_create = int(usage.get("cache_creation_input_tokens", 0) or 0)
    cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
    cost = float(result_event.get("total_cost_usd", 0.0) or 0.0)

    new_msg_history = new_msg_history + [{
        "role": "assistant",
        "content": [{"type": "text", "text": content}],
    }]

    return QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model_name,
        kwargs=kwargs,
        input_tokens=input_tokens + cache_create + cache_read,
        output_tokens=output_tokens,
        thinking_tokens=0,
        cost=cost,
        input_cost=0.0,
        output_cost=0.0,
        thought="",
        model_posteriors=model_posteriors,
    )


def _ensure_cli():
    if shutil.which(CLAUDE_BIN) is None:
        raise ClaudeCliError(
            f"`{CLAUDE_BIN}` not found on PATH. Install Claude Code or set "
            "CLAUDE_CLI_BIN to its absolute path."
        )


def _build_history_record(msg, msg_history):
    return msg_history + [{
        "role": "user",
        "content": [{"type": "text", "text": msg}],
    }]


@backoff.on_exception(
    backoff.expo,
    subprocess.TimeoutExpired,
    max_tries=TIMEOUT_MAX_TRIES,
    max_value=TIMEOUT_MAX_VALUE,
    on_backoff=backoff_handler,
)
@backoff.on_exception(
    backoff.expo,
    ClaudeCliError,
    max_tries=MAX_TRIES,
    max_value=MAX_VALUE,
    max_time=MAX_TIME,
    on_backoff=backoff_handler,
)
def query_claude_cli(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Query a Claude model via the Claude Code CLI (OAuth-backed)."""
    if output_model is not None:
        raise NotImplementedError("Structured output not supported for claude_cli.")
    _ensure_cli()

    cli_model = _strip_prefix(model)
    user_msg = _flatten_history(msg, msg_history)
    # `--` terminates option parsing so a user_msg starting with `-` is
    # treated as a positional argument rather than a CLI flag.
    argv = _build_argv(cli_model, system_msg, kwargs) + ["--", user_msg]
    new_msg_history = _build_history_record(msg, msg_history)

    proc = subprocess.run(
        argv, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        timeout=CLAUDE_TIMEOUT, cwd=tempfile.gettempdir(),
    )
    if proc.returncode != 0:
        raise ClaudeCliError(
            f"claude CLI returned {proc.returncode}. "
            f"stderr: {proc.stderr.strip()[-400:]}"
        )

    result_event = _parse_json_response(proc.stdout)
    return _result_to_query(
        result_event, msg, system_msg, model, new_msg_history,
        kwargs, model_posteriors,
    )


@backoff.on_exception(
    backoff.expo,
    asyncio.TimeoutError,
    max_tries=TIMEOUT_MAX_TRIES,
    max_value=TIMEOUT_MAX_VALUE,
    on_backoff=backoff_handler,
)
@backoff.on_exception(
    backoff.expo,
    ClaudeCliError,
    max_tries=MAX_TRIES,
    max_value=MAX_VALUE,
    max_time=MAX_TIME,
    on_backoff=backoff_handler,
)
async def query_claude_cli_async(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    """Async variant — same protocol, asyncio subprocess."""
    if output_model is not None:
        raise NotImplementedError("Structured output not supported for claude_cli.")
    _ensure_cli()

    cli_model = _strip_prefix(model)
    user_msg = _flatten_history(msg, msg_history)
    argv = _build_argv(cli_model, system_msg, kwargs) + ["--", user_msg]
    new_msg_history = _build_history_record(msg, msg_history)

    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=tempfile.gettempdir(),
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(), timeout=CLAUDE_TIMEOUT)
    except (asyncio.TimeoutError, asyncio.CancelledError, KeyboardInterrupt):
        proc.kill()
        try:
            await proc.wait()
        except BaseException:
            pass
        raise

    if proc.returncode != 0:
        raise ClaudeCliError(
            f"claude CLI returned {proc.returncode}. "
            f"stderr: {stderr_b.decode('utf-8', errors='replace').strip()[-400:]}"
        )

    result_event = _parse_json_response(stdout_b.decode("utf-8", errors="replace"))
    return _result_to_query(
        result_event, msg, system_msg, model, new_msg_history,
        kwargs, model_posteriors,
    )
