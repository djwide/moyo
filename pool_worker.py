"""Long-running Pub/Sub entrypoint for the Moyo image.

Cloud Run Jobs stay one-shot::

    python cloud_worker.py

That reads ``ORDER_ID``, calls :func:`cloud_worker.run_order`, and exits.
A Worker Pool uses the same image with a different command::

    python pool_worker.py

The process stays up, pulls orders, and calls ``run_order(order_id)`` for
each message. The report code itself does not change.

Environment:
    PROJECT_ID or GOOGLE_CLOUD_PROJECT
    SUBSCRIPTION_ID

Message body is JSON, ``{"orderId": "ord_..."}`` (``order_id`` is also
accepted). Before work starts, the order is claimed in a transaction
(``queued`` → ``processing``). An order that is already ``delivered`` — or
otherwise finished — is acknowledged and not run again. A finished
``run_order`` call is also acknowledged, including a recorded failure.
An exception before that — bad payload, or a crash that escaped
``run_order`` — is nacked so Pub/Sub can redeliver it. Give the
subscription a dead-letter topic so a poison message does not retry forever.

One message is pulled at a time. ``run_order`` keeps per-order files and
process-wide clients, so two orders must not run in the same process.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cloud_worker import OrderClaim, claim_queued_order, ensure_env_loaded, run_order

logger = logging.getLogger("moyo.pool_worker")

# Report jobs are allowed to run for two hours. Keep the lease longer so
# Pub/Sub does not redeliver an order that is still in progress.
DEFAULT_MAX_LEASE_SECONDS = 3 * 60 * 60


def project_id() -> str:
    return (
        os.environ.get("PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT") or ""
    ).strip()


def subscription_id() -> str:
    return (os.environ.get("SUBSCRIPTION_ID") or "").strip()


def order_id_from_payload(data: bytes) -> str:
    """Return the order id from a Pub/Sub message body."""
    payload = json.loads(data.decode("utf-8"))
    if isinstance(payload, str):
        order_id = payload.strip()
    elif isinstance(payload, dict):
        order_id = str(payload.get("orderId") or payload.get("order_id") or "").strip()
    else:
        order_id = ""
    if not order_id:
        raise ValueError("Pub/Sub message is missing orderId")
    return order_id


def _run_claimed_order(order_id: str) -> int:
    return run_order(order_id, already_claimed=True)


def handle_message(
    message: Any,
    *,
    run: Callable[[str], int] = _run_claimed_order,
    claim: Callable[[str], OrderClaim] = claim_queued_order,
) -> None:
    """Claim one order, run it, then ack or nack the message."""
    try:
        order_id = order_id_from_payload(message.data)
        logger.info("received order %s", order_id)
        decision = claim(order_id)
        if decision.action == "ack":
            logger.info(
                "order %s is %s; acknowledging",
                order_id,
                decision.report_status or "finished",
            )
            message.ack()
            return
        if decision.action == "nack":
            logger.info(
                "order %s is %s; leaving it with the worker that claimed it",
                order_id,
                decision.report_status or "processing",
            )
            message.nack()
            return
        code = run(order_id)
        if code:
            logger.error("order %s finished with status %s", order_id, code)
        message.ack()
        logger.info("completed order %s", order_id)
    except Exception:
        logger.exception("worker failed")
        message.nack()


def _flow_control(pubsub_v1: Any) -> Any:
    max_messages = int(os.environ.get("MOYO_POOL_MAX_MESSAGES") or "1")
    max_lease = float(
        os.environ.get("MOYO_POOL_MAX_LEASE_SECONDS") or DEFAULT_MAX_LEASE_SECONDS
    )
    if max_messages < 1:
        max_messages = 1
    if max_messages > 1:
        logger.warning(
            "MOYO_POOL_MAX_MESSAGES=%s; overlapping run_order calls share one process",
            max_messages,
        )
    return pubsub_v1.types.FlowControl(
        max_messages=max_messages,
        max_lease_duration=max_lease,
    )


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("MOYO_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ensure_env_loaded()
    project = project_id()
    subscription = subscription_id()
    if not project or not subscription:
        logger.error("PROJECT_ID and SUBSCRIPTION_ID are required")
        return 2

    from google.cloud import pubsub_v1

    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(project, subscription)
    logger.info("Moyo worker ready: %s", subscription_path)
    streaming_pull = subscriber.subscribe(
        subscription_path,
        callback=handle_message,
        flow_control=_flow_control(pubsub_v1),
    )

    def _stop(signum: int, _frame: Any) -> None:
        logger.info("received signal %s; shutting down", signum)
        streaming_pull.cancel()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        streaming_pull.result()
    except (KeyboardInterrupt, concurrent.futures.CancelledError):
        logger.info("subscriber stopped")
        return 0
    except Exception:
        logger.exception("subscriber stopped with an error")
        return 1
    finally:
        streaming_pull.cancel()
        subscriber.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
