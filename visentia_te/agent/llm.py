"""Thin Claude API wrapper + the generate -> check -> retry once -> template fallback loop.

The client is always injected (tests pass a fake). `default_client()` returns None when no
ANTHROPIC_API_KEY is set, which selects the deterministic, zero-network template path.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

from .grounding import GroundingReport

DEFAULT_MODEL = os.environ.get("VISENTIA_LLM_MODEL", "claude-sonnet-5-5")
# Server-side refusal fallback: if a safety classifier declines, the API re-runs the request on
# a fallback model inside the same call ("default" = API picks by refusal category).
_FALLBACK_BETA = "server-side-fallback-2026-07-01"

try:  # anthropic is an optional extra ([app]); the template path must work without it
    from anthropic import APIError as _APIError
except ImportError:  # pragma: no cover
    _APIError = Exception  # type: ignore[misc,assignment]


class LLMError(RuntimeError):
    pass


def default_client() -> Any | None:
    """Real Anthropic client if ANTHROPIC_API_KEY is set, else None (template path)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    import anthropic

    return anthropic.Anthropic(timeout=60.0, max_retries=2)


def complete(client: Any, *, system: str, user: str, model: str = DEFAULT_MODEL, max_tokens: int = 8000) -> str:
    """One Messages API call; returns the concatenated text blocks."""
    resp = client.beta.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": "low"},  # short, tightly constrained writing task
        betas=[_FALLBACK_BETA],
        fallbacks="default",
    )
    if resp.stop_reason in ("refusal", "max_tokens"):
        raise LLMError(f"stop_reason={resp.stop_reason}")
    text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text").strip()
    if not text:
        raise LLMError("empty response")
    return text


@dataclass
class DraftResult:
    text: str
    source: str  # "template" | "llm" | "llm_retry" | "template_fallback"
    grounding: GroundingReport  # report for the final `text`
    llm_failed: bool = False  # an LLM draft was rejected (offending spans kept in `attempts`)
    attempts: list[dict] = field(default_factory=list)
    model: str | None = None
    extra: dict = field(default_factory=dict)  # e.g. readability

    def to_dict(self) -> dict:
        return {"text": self.text, "source": self.source, "grounding": self.grounding.to_dict(),
                "llm_failed": self.llm_failed, "attempts": self.attempts, "model": self.model, **self.extra}


def generate_checked(client: Any | None, *, system: str, user: str, check: Callable[[str], GroundingReport],
                     template: Callable[[], str], model: str = DEFAULT_MODEL) -> DraftResult:
    """LLM draft -> check; on failure one retry with the check's feedback; then template fallback."""
    if client is None:
        text = template()
        return DraftResult(text, "template", check(text))
    attempts: list[dict] = []
    prompt = user
    for source in ("llm", "llm_retry"):
        try:
            text = complete(client, system=system, user=prompt, model=model)
        except (_APIError, LLMError) as e:
            attempts.append({"source": source, "error": f"{type(e).__name__}: {e}"})
            break  # API/refusal failures are not fixed by feedback; go straight to template
        rep = check(text)
        attempts.append({"source": source, "text": text, "grounding": rep.to_dict()})
        if rep.ok:
            return DraftResult(text, source, rep, llm_failed=source != "llm", attempts=attempts, model=model)
        prompt = (f"{user}\n\nYour previous draft was rejected by an automated check. {rep.feedback()}\n"
                  f"Previous draft:\n<draft>\n{text}\n</draft>\nRewrite it, using only values in FACTS.")
    text = template()
    return DraftResult(text, "template_fallback", check(text), llm_failed=True, attempts=attempts, model=model)
