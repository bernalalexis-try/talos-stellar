"""Tests for redaction of retry and circuit-breaker diagnostics.

Coverage
--------
Exception path: bearer tokens, Stellar secret seeds, secrets past the
  truncation boundary.
Payment headers: X-PAYMENT in plain text and in JSON, x-payment-signature.
Nested payloads: secrets inside nested objects and lists of objects.
Breaker paths: open-circuit skip, half-open probe failure, exhausted chain.
Safe metadata: non-secret diagnostics pass through unchanged.
Decisions unchanged: provider order, attempt counts, breaker recording.
"""

from __future__ import annotations

import logging

import pytest

from talos_agent.circuit_breaker import (
    CircuitBreakerOpen,
    CircuitState,
    cb_registry,
)
from talos_agent.http import redact_text
from talos_agent.routing.fallback import FallbackChain, _summarise_exception

FALLBACK_LOGGER = "talos_agent.routing.fallback"

# ── Sensitive fixtures ────────────────────────────────────────────────────────

BEARER_TOKEN = "sk-live-4f9aQ2mZx7VbNc1LpR8tYw3EhJ6UdG0sKiOa"
STELLAR_SECRET = "SB7WQ2LNTYFXQVBQAB3CFKBEDMJUCB6X4EHBK5CIHTL4FKUNSZJ6PVHT"
PAYMENT_HEADER = "eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.cGF5bG9hZA.c2ln"


def _messages(caplog) -> str:
    """Join the fallback log lines into one searchable string."""
    return "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name == FALLBACK_LOGGER
    )


def _attempt_text(result) -> str:
    """Join the error strings returned to the caller."""
    return "\n".join(msg for _, msg in result.attempts)


@pytest.fixture(autouse=True)
def _reset_breakers():
    """Reset the module-level breaker registry between tests."""
    cb_registry.reset_all()
    yield
    cb_registry.reset_all()


def _failing(exc: Exception):
    """Build an operation that always raises *exc*."""

    async def operation(provider_name: str, *args, **kwargs):
        raise exc

    return operation


# ═══════════════════════════════════════════════════════════════════════════════
# Exception path
# ═══════════════════════════════════════════════════════════════════════════════


class TestExceptionPathRedaction:
    @pytest.mark.asyncio
    async def test_bearer_token_absent_from_log_and_attempts(self, caplog):
        chain = FallbackChain(["groq"])
        exc = RuntimeError(f"401 Unauthorized — Authorization: Bearer {BEARER_TOKEN}")

        with caplog.at_level(logging.WARNING, logger=FALLBACK_LOGGER):
            result = await chain.execute(_failing(exc))

        logs = _messages(caplog)
        assert BEARER_TOKEN not in logs
        assert BEARER_TOKEN not in _attempt_text(result)
        assert "[REDACTED]" in logs
        # Safe metadata is still there.
        assert "groq" in logs
        assert "RuntimeError" in logs
        assert "attempt=1" in logs
        assert "elapsed=" in logs

    @pytest.mark.asyncio
    async def test_stellar_secret_key_absent(self, caplog):
        chain = FallbackChain(["stellar"])
        exc = ValueError(f"signing failed for key {STELLAR_SECRET}")

        with caplog.at_level(logging.WARNING, logger=FALLBACK_LOGGER):
            result = await chain.execute(_failing(exc))

        assert STELLAR_SECRET not in _messages(caplog)
        assert STELLAR_SECRET not in _attempt_text(result)

    @pytest.mark.asyncio
    async def test_secret_past_truncation_boundary_is_still_redacted(self, caplog):
        """Redaction must run before truncation, not after."""
        exc = RuntimeError("x" * 400 + f" api_key='{BEARER_TOKEN}'")

        summary = _summarise_exception(exc)

        assert BEARER_TOKEN not in summary
        assert summary.startswith("RuntimeError: ")

    def test_summarise_still_truncates(self):
        """Existing truncation contract is unchanged."""
        summary = _summarise_exception(RuntimeError("x" * 500))
        assert len(summary) <= 220
        assert summary.endswith("...")


# ═══════════════════════════════════════════════════════════════════════════════
# Payment headers
# ═══════════════════════════════════════════════════════════════════════════════


class TestPaymentHeaderRedaction:
    @pytest.mark.asyncio
    async def test_plaintext_x402_header_absent(self, caplog):
        chain = FallbackChain(["x402"])
        exc = RuntimeError(f"402 rejected — X-PAYMENT: {PAYMENT_HEADER}")

        with caplog.at_level(logging.WARNING, logger=FALLBACK_LOGGER):
            result = await chain.execute(_failing(exc))

        logs = _messages(caplog)
        assert PAYMENT_HEADER not in logs
        assert PAYMENT_HEADER not in _attempt_text(result)
        # The header name stays; only its value goes.
        assert "X-PAYMENT" in logs.upper()

    @pytest.mark.asyncio
    async def test_json_payment_headers_absent(self, caplog):
        chain = FallbackChain(["x402"])
        body = (
            '{"error": "payment_required", "headers": '
            f'{{"X-PAYMENT": "{PAYMENT_HEADER}", '
            f'"x-payment-signature": "{STELLAR_SECRET}", '
            f'"Authorization": "Bearer {BEARER_TOKEN}"}}, '
            '"status": 402}'
        )

        with caplog.at_level(logging.WARNING, logger=FALLBACK_LOGGER):
            result = await chain.execute(_failing(RuntimeError(body)))

        combined = _messages(caplog) + _attempt_text(result)
        for secret in (PAYMENT_HEADER, STELLAR_SECRET, BEARER_TOKEN):
            assert secret not in combined
        # Other fields of the same body are untouched.
        assert "payment_required" in combined

    def test_payment_header_variants(self):
        for key in ("X-PAYMENT", "x-payment-signature", "X_Payment_Proof"):
            redacted = redact_text(f'{{"{key}": "{PAYMENT_HEADER}"}}')
            assert PAYMENT_HEADER not in redacted, key


# ═══════════════════════════════════════════════════════════════════════════════
# Nested payloads
# ═══════════════════════════════════════════════════════════════════════════════


class TestNestedPayloadRedaction:
    @pytest.mark.asyncio
    async def test_deeply_nested_secret_absent(self, caplog):
        chain = FallbackChain(["groq"])
        body = (
            '{"request": {"config": {"credentials": '
            f'{{"api_key": "{BEARER_TOKEN}"}}}}, '
            '"provider": "groq"}}'
        )

        with caplog.at_level(logging.WARNING, logger=FALLBACK_LOGGER):
            result = await chain.execute(_failing(RuntimeError(body)))

        combined = _messages(caplog) + _attempt_text(result)
        assert BEARER_TOKEN not in combined

    def test_secret_inside_list_of_objects(self):
        payload = (
            f'{{"attempts": [{{"provider": "groq", "auth": {{"token": "{BEARER_TOKEN}"}}}}, '
            '{"provider": "openai", "total_tokens": 1250}]}'
        )
        redacted = redact_text(payload)

        assert BEARER_TOKEN not in redacted
        # Token counts are usage metadata, not secrets.
        assert "1250" in redacted
        assert "groq" in redacted


# ═══════════════════════════════════════════════════════════════════════════════
# Open-circuit and half-open paths
# ═══════════════════════════════════════════════════════════════════════════════


class TestBreakerPathRedaction:
    @pytest.mark.asyncio
    async def test_open_circuit_skip_keeps_timing_metadata(self, caplog):
        breaker = cb_registry.get("groq")
        breaker.state = CircuitState.OPEN

        chain = FallbackChain(["groq"])

        async def never_called(provider_name: str, *args, **kwargs):
            raise AssertionError("operation must not run while OPEN")

        with caplog.at_level(logging.WARNING, logger=FALLBACK_LOGGER):
            result = await chain.execute(never_called)

        logs = _messages(caplog)
        assert result.success is False
        assert "state=open" in logs
        assert "retry in" in logs
        assert "groq" in logs

    @pytest.mark.asyncio
    async def test_half_open_probe_failure_is_labelled_and_redacted(self, caplog):
        breaker = cb_registry.get("groq")
        breaker.state = CircuitState.HALF_OPEN

        chain = FallbackChain(["groq"])
        exc = CircuitBreakerOpen("groq", 12.5, fallback_hint=f"api_key={BEARER_TOKEN}")

        with caplog.at_level(logging.WARNING, logger=FALLBACK_LOGGER):
            result = await chain.execute(_failing(exc))

        logs = _messages(caplog)
        assert BEARER_TOKEN not in logs
        assert BEARER_TOKEN not in _attempt_text(result)
        assert "state=half_open" in logs
        assert "12.5s" in logs
        assert "groq" in logs

    @pytest.mark.asyncio
    async def test_exhausted_chain_summary_carries_no_secrets(self, caplog):
        chain = FallbackChain(["groq", "openai"])
        exc = RuntimeError(f"Authorization: Bearer {BEARER_TOKEN}")

        with caplog.at_level(logging.WARNING, logger=FALLBACK_LOGGER):
            result = await chain.execute(_failing(exc))

        logs = _messages(caplog)
        assert BEARER_TOKEN not in logs
        assert "exhausted" in logs.lower()
        assert "groq" in logs and "openai" in logs
        assert result.total_attempts == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Safe metadata must survive
# ═══════════════════════════════════════════════════════════════════════════════


class TestSafeMetadataPreserved:
    """Diagnostics this codebase emits that contain no secrets.

    Over-redaction is as much a bug as under-redaction, so these must come
    through unchanged.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "HTTP 503 from https://api.groq.com/openai/v1/chat/completions",
            "Circuit breaker OPEN (retry in 12.5s)",
            "provider=groq attempt=2 elapsed=431.2ms state=half_open",
            "rate limited, retry_after=12.5s, attempt=2/3",
            "idempotency_key=job-4471 already processed",
            "primary_key=id, foreign_key=talos_id",
            "partition key=user_id not found",
            '{"usage": {"prompt_tokens": 120, "total_tokens": 160}}',
            '{"payment_status": "settled", "amount": "0.25", "asset": "USDC"}',
            '{"authenticated": true, "region": "us-east-1"}',
            '{"error": {"code": "rate_limit_exceeded", "retry_after": 30}}',
        ],
    )
    def test_non_secret_diagnostics_pass_through_unchanged(self, text):
        assert redact_text(text) == text


class TestKnownSecretShapes:
    """Secret shapes that must never survive redaction."""

    @pytest.mark.parametrize(
        "text,secret",
        [
            ("Authorization: Bearer sk-live-ABC123xyz", "sk-live-ABC123xyz"),
            (f"seed {STELLAR_SECRET}", STELLAR_SECRET),
            ("api_key=sk-live-UNQUOTED1 rejected", "sk-live-UNQUOTED1"),
            ("GET https://x.com/v1?key=sk-live-INURL1", "sk-live-INURL1"),
            ("private_key=SECRETPRIV", "SECRETPRIV"),
            ("signing_key=SECRETSIGN", "SECRETSIGN"),
            ('{"payment_proof": "SECRETPROOF"}', "SECRETPROOF"),
            ('{"items": [{"auth_token": "SECRETLIST"}]}', "SECRETLIST"),
        ],
    )
    def test_secret_is_redacted(self, text, secret):
        assert secret not in redact_text(text)


# ═══════════════════════════════════════════════════════════════════════════════
# Behaviour is unchanged
# ═══════════════════════════════════════════════════════════════════════════════


class TestDecisionsUnchanged:
    @pytest.mark.asyncio
    async def test_success_passthrough(self):
        chain = FallbackChain(["groq", "openai"])

        async def operation(provider_name: str, *args, **kwargs):
            return f"ok:{provider_name}"

        result = await chain.execute(operation)

        assert result.success is True
        assert result.provider_name == "groq"
        assert result.result == "ok:groq"
        assert result.total_attempts == 1

    @pytest.mark.asyncio
    async def test_attempt_count_and_order_unchanged_with_secrets_present(self):
        """Provider order and attempt count are unaffected by redaction."""
        chain = FallbackChain(["groq", "openai", "anthropic"])
        seen: list[str] = []

        async def operation(provider_name: str, *args, **kwargs):
            seen.append(provider_name)
            if provider_name != "anthropic":
                raise RuntimeError(f"api_key={BEARER_TOKEN} rejected")
            return "ok"

        result = await chain.execute(operation)

        assert seen == ["groq", "openai", "anthropic"]
        assert result.success is True
        assert result.provider_name == "anthropic"
        assert result.total_attempts == 3
        assert len(result.attempts) == 2

    @pytest.mark.asyncio
    async def test_failures_still_recorded_on_breaker(self):
        chain = FallbackChain(["groq"])

        await chain.execute(_failing(RuntimeError(f"token {BEARER_TOKEN}")))

        metrics = cb_registry.get("groq").metrics()
        assert metrics.total_failures == 1

    @pytest.mark.asyncio
    async def test_successes_still_recorded_on_breaker(self):
        chain = FallbackChain(["groq"])

        async def operation(provider_name: str, *args, **kwargs):
            return "ok"

        await chain.execute(operation)

        metrics = cb_registry.get("groq").metrics()
        assert metrics.total_successes == 1
