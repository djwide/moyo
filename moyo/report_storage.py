"""GCS layout and upload for local and Cloud Run report jobs.

Cloud jobs write ``gs://<bucket>/reports/<storageFolder>/`` where
``storageFolder`` is the prompt's primary topic plus a short order suffix
(:func:`moyo.order_storage.order_storage_folder`). Local ``build_report``
runs use the same object names and the same bucket by default.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("moyo.report_storage")

# Dedicated worker/QC bucket. Not the Firebase Auth app bucket.
DEFAULT_MOYO_REPORTS_BUCKET = "senteguard-website-moyo-reports"
DEFAULT_GCP_PROJECT = "senteguard-website"
_SKIP_UPLOAD_VALUES = frozenset({"1", "true", "yes", "on"})
_GCS_PREFIX_PTR = "gcs.prefix"


@dataclass(frozen=True)
class PublishResult:
    bucket: str
    storage_folder: str
    prefix: str
    gcs_prefix: str
    urls: dict[str, str]


def normalize_storage_bucket_name(value: str) -> str:
    text = value.strip()
    if text.lower().startswith("gs://"):
        text = text[5:]
    return text.strip().strip("/")


def reports_bucket_name() -> str:
    """Dedicated reports bucket, never the Firebase Auth app bucket."""
    explicit = (
        os.environ.get("MOYO_REPORTS_STORAGE_BUCKET")
        or os.environ.get("STORAGE_BUCKET")
        or ""
    ).strip()
    if explicit:
        return normalize_storage_bucket_name(explicit)
    return DEFAULT_MOYO_REPORTS_BUCKET


def gcp_project_id() -> str:
    return (
        os.environ.get("MOYO_CLOUD_PROJECT")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCLOUD_PROJECT")
        or os.environ.get("GCP_PROJECT")
        or DEFAULT_GCP_PROJECT
    ).strip() or DEFAULT_GCP_PROJECT


def local_order_id(run_id: str) -> str:
    """Stable order id so re-running a local report overwrites the same prefix."""
    slug = (run_id or "").strip().strip("/") or "report"
    return f"ord_local_{slug}"


def project_slug_from_path(path: Path | None) -> str | None:
    """``projects/<slug>/...`` → slug, else None."""
    if path is None:
        return None
    try:
        from moyo.project import projects_root

        rel = Path(path).resolve().relative_to(projects_root())
    except (ValueError, OSError):
        return None
    parts = rel.parts
    return parts[0] if parts else None


def infer_local_run_id(exploration: Path | None, fallback: str | None = None) -> str:
    """Prefer the project slug over a ``public_sources`` parent folder name."""
    if fallback and fallback.strip() and fallback.strip() not in {
        "public_sources",
        "explorations",
    }:
        return fallback.strip()
    slug = project_slug_from_path(exploration)
    if slug:
        return slug
    if exploration is not None:
        parent = Path(exploration).parent
        if parent.name not in {"public_sources", "explorations"}:
            return parent.name
        if parent.parent.name:
            return parent.parent.name
        return parent.name
    if fallback and fallback.strip():
        return fallback.strip()
    raise ValueError("Provide --run-id (or --exploration to infer it).")


def skip_upload_requested() -> bool:
    return os.environ.get("MOYO_REPORTS_SKIP_UPLOAD", "").strip().lower() in (
        _SKIP_UPLOAD_VALUES
    )


def should_upload_local_report(
    *,
    upload: bool = True,
    test_mode: bool = False,
    rendered: bool = True,
    graphics_only: bool = False,
) -> bool:
    """True when a desktop ``build_report`` should push artifacts to GCS.

    Cloud Run skips this path: the worker uploads once after all prompts.
    ``--test`` / ``MOYO_TEST_MODE`` never talks to GCS.
    """
    if not upload or test_mode or graphics_only or not rendered:
        return False
    if skip_upload_requested():
        return False
    try:
        from moyo.llm.testing import is_test_mode

        if is_test_mode():
            return False
    except Exception:
        pass
    try:
        from moyo.llm.utility import running_in_cloud

        if running_in_cloud():
            return False
    except Exception:
        pass
    return True


def content_type_for(path: Path) -> str | None:
    ext = path.suffix.lower()
    return {
        ".pdf": "application/pdf",
        ".json": "application/json; charset=utf-8",
        ".jsonl": "application/jsonl; charset=utf-8",
        ".md": "text/markdown; charset=utf-8",
        ".html": "text/html; charset=utf-8",
        ".svg": "image/svg+xml",
        ".txt": "text/plain; charset=utf-8",
    }.get(ext)


def upload_files(bucket: Any, pairs: list[tuple[str, Path]]) -> dict[str, str]:
    urls: dict[str, str] = {}
    for object_path, path in pairs:
        if not path.exists():
            continue
        blob = bucket.blob(object_path)
        content_type = content_type_for(path)
        if content_type:
            blob.upload_from_filename(str(path), content_type=content_type)
        else:
            blob.upload_from_filename(str(path))
        urls[object_path] = f"gs://{bucket.name}/{object_path}"
        logger.info("uploaded %s", urls[object_path])
    return urls


def reports_bucket(bucket: Any | None = None):
    """Return a ``google.cloud.storage`` bucket (ADC, then gcloud user token)."""
    if bucket is not None:
        return bucket
    try:
        from google.cloud import storage as gcs
    except ImportError as exc:
        raise RuntimeError(
            "google-cloud-storage is required to upload reports. "
            'Install with: pip install -e ".[reports,cloud]"'
        ) from exc
    project = gcp_project_id()
    name = reports_bucket_name()
    try:
        client = gcs.Client(project=project)
    except Exception:
        logger.debug("GCS ADC unavailable; trying gcloud access token", exc_info=True)
        from google.oauth2.credentials import Credentials
        from moyo.gui.cloud_compute import gcloud_access_token

        client = gcs.Client(
            project=project,
            credentials=Credentials(gcloud_access_token()),
        )
    return client.bucket(name)


def _prompts_from_exploration_file(path: Path) -> list[str]:
    """Cover prompt from the exploration.md title line (same rule as parse.py)."""
    first = path.read_text(encoding="utf-8").splitlines()[:8]
    for line in first:
        if line.startswith("# Topic exploration:"):
            topic = line.split(":", 1)[1].strip()
            return [topic] if topic else []
        if line.startswith("# "):
            topic = line[2:].strip()
            return [topic] if topic else []
    return []


def gcs_prefix_for(folder: str, *, bucket_name: str | None = None) -> str:
    name = bucket_name or reports_bucket_name()
    return f"gs://{name}/reports/{folder.strip('/')}/"


def publish_local_report(
    *,
    run_dir: Path,
    run_id: str,
    product: str = "snapshot",
    exploration: Path | None = None,
    prompts: list[str] | None = None,
    bucket: Any | None = None,
) -> PublishResult:
    """Upload ``reports/build/<run-id>`` using the Cloud Run object layout."""
    from moyo.llm.client import ensure_env_loaded

    ensure_env_loaded()
    import cloud_worker as cw

    prompt_list = [str(p).strip() for p in (prompts or []) if str(p).strip()]
    if not prompt_list and exploration is not None and Path(exploration).is_file():
        prompt_list = _prompts_from_exploration_file(Path(exploration))
    if not prompt_list:
        prompt_list = [run_id.replace("_", " ")]

    spec = cw.OrderSpec(
        order_id=local_order_id(run_id),
        prompts=prompt_list,
        product=product,
        generation_mode="full",
        qc_required=False,
        source="local",
    )
    evidence = cw.build_evidence(
        run_dir, prompt=prompt_list[0] if prompt_list else None
    )
    (run_dir / "evidence.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    slug = cw.prompt_slug(1, prompt_list[0])
    artifacts = cw.collect_artifacts(run_dir, run_dir, product)
    if exploration is not None and Path(exploration).is_file():
        artifacts.setdefault("exploration.md", Path(exploration))
    run = cw.PromptRun(
        index=1,
        prompt=prompt_list[0],
        slug=slug,
        run_id=run_id,
        artifacts=artifacts,
    )
    cw.write_prompt_report_json(run_dir, spec, run, evidence=evidence)
    canonical = cw.write_canonical_report_json(spec, [run], run_dir / "report.json")
    prefix = spec.reports_prefix()
    dest = dict(cw.storage_destinations(spec.storage_folder, [run]))
    dest[f"{prefix}/report.json"] = canonical

    target = reports_bucket(bucket)
    urls = upload_files(target, list(dest.items()))
    manifest = cw.artifact_manifest(
        spec.order_id, [run], folder=spec.storage_folder
    )
    manifest_blob = target.blob(f"{prefix}/manifest.json")
    manifest_blob.upload_from_string(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        content_type="application/json",
    )
    urls[f"{prefix}/manifest.json"] = f"gs://{target.name}/{prefix}/manifest.json"

    result = PublishResult(
        bucket=target.name,
        storage_folder=spec.storage_folder,
        prefix=prefix,
        gcs_prefix=gcs_prefix_for(spec.storage_folder, bucket_name=target.name),
        urls=urls,
    )
    (run_dir / _GCS_PREFIX_PTR).write_text(result.gcs_prefix + "\n", encoding="utf-8")
    return result
