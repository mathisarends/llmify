import logging

import pytest

from llmify.utils import timed


def test_timed_logs_slow_sync_function(caplog, monkeypatch) -> None:
    timestamps = iter((10.0, 10.3))
    monkeypatch.setattr("llmify.utils.time.perf_counter", lambda: next(timestamps))

    @timed("sync operation")
    def operation(value: int) -> int:
        return value * 2

    with caplog.at_level(logging.DEBUG):
        result = operation(21)

    assert result == 42
    assert "⏳ sync operation() took 0.30s" in caplog.messages


@pytest.mark.asyncio
async def test_timed_logs_slow_async_function(caplog, monkeypatch) -> None:
    timestamps = iter((10.0, 10.3))
    monkeypatch.setattr("llmify.utils.time.perf_counter", lambda: next(timestamps))

    @timed()
    async def operation(value: int) -> int:
        return value * 2

    with caplog.at_level(logging.DEBUG):
        result = await operation(21)

    assert result == 42
    assert "⏳ operation() took 0.30s" in caplog.messages


def test_timed_does_not_log_fast_function(caplog, monkeypatch) -> None:
    timestamps = iter((10.0, 10.1))
    monkeypatch.setattr("llmify.utils.time.perf_counter", lambda: next(timestamps))

    @timed()
    def operation() -> None:
        pass

    with caplog.at_level(logging.DEBUG):
        operation()

    assert not caplog.messages
