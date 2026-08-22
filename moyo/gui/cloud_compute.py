"""Submit a GUI explore/report run to the Cloud Run worker.

The desktop app does not run the LLM fan-out itself. It writes a Firestore
order (same shape the storefront uses) and executes ``moyo-report-worker``.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from moyo.order_storage import order_storage_folder

ProgressFn = Callable[[str], None]


DEFAULT_PROJECT = "senteguard-website"
DEFAULT_REGION = "us-central1"
DEFAULT_JOB = "moyo-report-worker"
DEFAULT_COLLECTION = "reports"


@dataclass
class CloudComputeConfig:
    project: str = DEFAULT_PROJECT
    region: str = DEFAULT_REGION
    job: str = DEFAULT_JOB
    collection: str = DEFAULT_COLLECTION
    wait: bool = True

    @classmethod
    def from_env(cls) -> "CloudComputeConfig":
        return cls(
            project=(
                os.environ.get("MOYO_CLOUD_PROJECT")
                or os.environ.get("GOOGLE_CLOUD_PROJECT")
                or os.environ.get("GCLOUD_PROJECT")
                or DEFAULT_PROJECT
            ).strip()
            or DEFAULT_PROJECT,
            region=(os.environ.get("MOYO_CLOUD_REGION") or DEFAULT_REGION).strip()
            or DEFAULT_REGION,
            job=(os.environ.get("MOYO_CLOUD_JOB") or DEFAULT_JOB).strip() or DEFAULT_JOB,
            collection=(
                os.environ.get("FIRESTORE_ORDERS_COLLECTION")
                or os.environ.get("FIRESTORE_COLLECTION")
                or DEFAULT_COLLECTION
            ).strip()
            or DEFAULT_COLLECTION,
        )


@dataclass
class CloudSubmitResult:
    order_id: str
    execution_name: str | None
    firestore_path: str
    gcs_prefix: str


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_gui_order_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"ord_gui_{stamp}_{uuid.uuid4().hex[:8]}"


def build_order_payload(
    *,
    prompts: list[str],
    product: str = "snapshot",
    fuzz_mode: str = "basic",
    strategies: list[str] | None = None,
    languages: list[str] | None = None,
    include_remediation: bool = False,
    seeds: int = 3,
) -> dict[str, Any]:
    cleaned = [str(p).strip() for p in prompts if str(p).strip()]
    if not cleaned:
        raise ValueError("At least one prompt is required for a cloud run.")
    return {
        "orderId": None,  # filled by submit
        "prompts": cleaned,
        "customerPrompts": cleaned,
        "product": product,
        "paymentStatus": "paid",
        "reportStatus": "queued",
        "qcRequired": True,
        "qcStatus": "pending",
        "fuzzMode": fuzz_mode or "basic",
        "strategies": list(strategies or []),
        "languages": list(languages or []),
        "seeds": int(seeds),
        "includeRemediation": bool(include_remediation),
        "source": "gui",
        "createdAt": utc_now(),
    }


def firestore_value(value: Any) -> dict[str, Any]:
    """Encode a Python value as a Firestore REST API value."""
    if value is None:
        return {"nullValue": None}
    if isinstance(value, bool):
        return {"booleanValue": value}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"integerValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, list):
        return {"arrayValue": {"values": [firestore_value(v) for v in value]}}
    if isinstance(value, dict):
        return {
            "mapValue": {
                "fields": {str(k): firestore_value(v) for k, v in value.items()}
            }
        }
    return {"stringValue": str(value)}


def firestore_document(payload: dict[str, Any]) -> dict[str, Any]:
    return {"fields": {k: firestore_value(v) for k, v in payload.items()}}


def _gcloud_output(args: list[str]) -> str:
    proc = subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"{' '.join(args)}\n{err}")
    return (proc.stdout or "").strip()


def gcloud_access_token() -> str:
    token = _gcloud_output(["gcloud", "auth", "print-access-token"])
    if not token:
        raise RuntimeError(
            "gcloud auth print-access-token returned empty. "
            "Run `gcloud auth login` (and `gcloud auth application-default login` if needed)."
        )
    return token.splitlines()[0].strip()


def write_firestore_order(
    cfg: CloudComputeConfig,
    order_id: str,
    payload: dict[str, Any],
    *,
    token: str | None = None,
) -> str:
    token = token or gcloud_access_token()
    url = (
        f"https://firestore.googleapis.com/v1/projects/{cfg.project}"
        f"/databases/(default)/documents/{cfg.collection}?documentId={order_id}"
    )
    body = json.dumps(firestore_document(payload)).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(
            f"Failed to create Firestore {cfg.collection}/{order_id} "
            f"(HTTP {exc.code}): {detail}"
        ) from exc
    return f"{cfg.collection}/{order_id}"


def execute_cloud_job(
    cfg: CloudComputeConfig,
    order_id: str,
    *,
    progress: ProgressFn | None = None,
) -> str:
    """Start the Cloud Run job. Returns the execution resource name."""
    cmd = [
        "gcloud",
        "run",
        "jobs",
        "execute",
        cfg.job,
        "--project",
        cfg.project,
        "--region",
        cfg.region,
        "--update-env-vars",
        f"ORDER_ID={order_id}",
        "--async",
        "--format",
        "value(metadata.name)",
    ]
    if progress:
        progress(" ".join(cmd))
    name = _gcloud_output(cmd)
    if not name:
        raise RuntimeError("gcloud run jobs execute returned no execution name")
    return name.strip().splitlines()[-1].strip()


def wait_for_execution(
    cfg: CloudComputeConfig,
    execution_name: str,
    *,
    progress: ProgressFn | None = None,
    poll_seconds: int = 15,
    log_limit: int = 30,
) -> None:
    """Poll execution status and print recent logs until it finishes."""
    short = execution_name.rsplit("/", 1)[-1]
    seen_logs: set[str] = set()
    while True:
        status = _gcloud_output(
            [
                "gcloud",
                "run",
                "jobs",
                "executions",
                "describe",
                short,
                "--project",
                cfg.project,
                "--region",
                cfg.region,
                "--format",
                "json",
            ]
        )
        data = json.loads(status) if status else {}
        conds = ((data.get("status") or {}).get("conditions") or [])
        done = False
        ok = False
        message = ""
        for cond in conds:
            if cond.get("type") == "Completed":
                done = cond.get("status") in {"True", "False"}
                ok = cond.get("status") == "True"
                message = cond.get("message") or ""
        if progress:
            completion = (data.get("status") or {}).get("completionTime") or "running"
            progress(f"execution {short}: {completion}" + (f" — {message}" if message else ""))
            _emit_new_logs(cfg, short, seen_logs, progress, limit=log_limit)
        if done:
            if not ok:
                raise RuntimeError(
                    f"Cloud job execution {short} failed. {message}".strip()
                )
            return
        time.sleep(max(5, int(poll_seconds)))


def _emit_new_logs(
    cfg: CloudComputeConfig,
    execution_short: str,
    seen: set[str],
    progress: ProgressFn,
    *,
    limit: int,
) -> None:
    try:
        raw = _gcloud_output(
            [
                "gcloud",
                "logging",
                "read",
                (
                    f'resource.type="cloud_run_job" AND '
                    f'resource.labels.job_name="{cfg.job}" AND '
                    f'labels."run.googleapis.com/execution_name"="{execution_short}"'
                ),
                "--project",
                cfg.project,
                "--limit",
                str(limit),
                "--freshness",
                "2h",
                "--format",
                "value(timestamp,textPayload)",
            ]
        )
    except Exception as exc:
        progress(f"(log read skipped: {exc})")
        return
    for line in raw.splitlines():
        line = line.strip()
        if not line or line in seen:
            continue
        seen.add(line)
        progress(line)


def submit_cloud_compute(
    *,
    prompts: list[str],
    product: str = "snapshot",
    fuzz_mode: str = "basic",
    strategies: list[str] | None = None,
    languages: list[str] | None = None,
    include_remediation: bool = False,
    seeds: int = 3,
    cfg: CloudComputeConfig | None = None,
    progress: ProgressFn | None = None,
) -> CloudSubmitResult:
    """Write a Firestore order and execute the Cloud Run worker."""
    cfg = cfg or CloudComputeConfig.from_env()
    order_id = new_gui_order_id()
    payload = build_order_payload(
        prompts=prompts,
        product=product,
        fuzz_mode=fuzz_mode,
        strategies=strategies,
        languages=languages,
        include_remediation=include_remediation,
        seeds=seeds,
    )
    payload["orderId"] = order_id
    folder = order_storage_folder(order_id, prompts)
    payload["storageFolder"] = folder

    def _p(msg: str) -> None:
        if progress:
            progress(msg)

    _p(f"Creating Firestore order {cfg.collection}/{order_id} …")
    path = write_firestore_order(cfg, order_id, payload)
    _p(f"Wrote {path}")
    _p(f"Executing Cloud Run job {cfg.job} in {cfg.region} …")
    execution = execute_cloud_job(cfg, order_id, progress=_p)
    _p(f"Started execution {execution}")
    if cfg.wait:
        wait_for_execution(cfg, execution, progress=_p)
        _p("Cloud execution finished.")
    gcs_prefix = f"gs://senteguard-website-moyo-reports/reports/{folder}/"
    _p(f"Artifacts (when complete): {gcs_prefix}")
    return CloudSubmitResult(
        order_id=order_id,
        execution_name=execution,
        firestore_path=path,
        gcs_prefix=gcs_prefix,
    )
