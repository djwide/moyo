"""Unit tests for the long-running Pub/Sub worker (no network)."""

from __future__ import annotations

import json

import pytest

import pool_worker


class _Message:
    def __init__(self, data: bytes):
        self.data = data
        self.acked = False
        self.nacked = False

    def ack(self) -> None:
        self.acked = True

    def nack(self) -> None:
        self.nacked = True


def test_order_id_from_payload_accepts_camel_and_snake_case():
    assert pool_worker.order_id_from_payload(b'{"orderId": "ord_1"}') == "ord_1"
    assert pool_worker.order_id_from_payload(b'{"order_id": "ord_2"}') == "ord_2"


def test_order_id_from_payload_rejects_empty_body():
    with pytest.raises(json.JSONDecodeError):
        pool_worker.order_id_from_payload(b"")
    with pytest.raises(ValueError):
        pool_worker.order_id_from_payload(b"{}")


def _claim_run(_order_id: str) -> pool_worker.OrderClaim:
    return pool_worker.OrderClaim("run")


def test_handle_message_acks_after_run_order():
    message = _Message(json.dumps({"orderId": "ord_9"}).encode())
    seen: list[str] = []
    pool_worker.handle_message(
        message,
        run=lambda order_id: seen.append(order_id) or 0,
        claim=_claim_run,
    )
    assert seen == ["ord_9"]
    assert message.acked
    assert not message.nacked


def test_handle_message_acks_recorded_failure():
    message = _Message(json.dumps({"orderId": "ord_fail"}).encode())
    pool_worker.handle_message(message, run=lambda _order_id: 1, claim=_claim_run)
    assert message.acked
    assert not message.nacked


def test_handle_message_nacks_when_run_order_raises():
    message = _Message(json.dumps({"orderId": "ord_boom"}).encode())

    def _boom(_order_id: str) -> int:
        raise RuntimeError("worker crashed")

    pool_worker.handle_message(message, run=_boom, claim=_claim_run)
    assert message.nacked
    assert not message.acked


def test_handle_message_acks_delivered_without_running():
    message = _Message(json.dumps({"orderId": "ord_done"}).encode())

    def _claim(_order_id: str) -> pool_worker.OrderClaim:
        return pool_worker.OrderClaim("ack", "delivered")

    def _run(_order_id: str) -> int:
        raise AssertionError("delivered order should not run")

    pool_worker.handle_message(message, run=_run, claim=_claim)
    assert message.acked
    assert not message.nacked


def test_handle_message_nacks_while_another_worker_owns_it():
    message = _Message(json.dumps({"orderId": "ord_busy"}).encode())

    def _claim(_order_id: str) -> pool_worker.OrderClaim:
        return pool_worker.OrderClaim("nack", "processing", 1)

    def _run(_order_id: str) -> int:
        raise AssertionError("in-progress order should not run")

    pool_worker.handle_message(message, run=_run, claim=_claim)
    assert message.nacked
    assert not message.acked


def test_handle_message_nacks_invalid_payload():
    message = _Message(b"not-json")
    pool_worker.handle_message(message, run=lambda _order_id: 0, claim=_claim_run)
    assert message.nacked
    assert not message.acked


def test_main_requires_subscription(monkeypatch):
    monkeypatch.delenv("SUBSCRIPTION_ID", raising=False)
    monkeypatch.setenv("PROJECT_ID", "senteguard-website")
    monkeypatch.setattr(pool_worker, "ensure_env_loaded", lambda: None)
    assert pool_worker.main() == 2


def test_main_accepts_google_cloud_project(monkeypatch):
    monkeypatch.delenv("PROJECT_ID", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "senteguard-website")
    monkeypatch.delenv("SUBSCRIPTION_ID", raising=False)
    monkeypatch.setattr(pool_worker, "ensure_env_loaded", lambda: None)
    assert pool_worker.project_id() == "senteguard-website"
    assert pool_worker.main() == 2
