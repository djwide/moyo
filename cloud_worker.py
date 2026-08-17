"""Cloud Run / GCE worker: Firestore order → explore → report → Storage.

Triggered with ``ORDER_ID`` set. Reads ``reports/{ORDER_ID}`` (storefront
collection; override with ``FIRESTORE_ORDERS_COLLECTION``), runs the same
``moyo-gather explore`` + ``reports/build_report.py`` path used locally, then
uploads artifacts and marks the order ``qc_pending``.

Storefront order fields used here::

    prompts              list[str] | JSON string   required, non-empty
    product              snapshot | basis | both   e.g. "basis"
    paymentStatus        informational
    reportStatus         queued / awaiting_prompts → generating → qc_pending | failed
    qcStatus             left as pending (checkout default)
    generationStartedAt  ISO-8601 UTC, set when work begins
    generationFinishedAt ISO-8601 UTC, set on success or failure

One report per prompt. A single-prompt order also writes the five contract
files at ``reports/{order_id}/`` (the path QC already uses). Multi-prompt
orders use ``reports/{order_id}/{nn}_{slug}/`` plus ``manifest.json``.

Ollama is optional for now (deterministic seeds + unmerged clusters).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("moyo.cloud_worker")

PRODUCT_ALIASES = {
    "snapshot": "snapshot",
    "exposure": "snapshot",
    "exposure_snapshot": "snapshot",
    "exposure-snapshot": "snapshot",
    "one_page": "snapshot",
    "one-page": "snapshot",
    "onepage": "snapshot",
    "basis": "basis",
    "basis_report": "basis",
    "basis-report": "basis",
    "full": "basis",
    "both": "both",
    "all": "both",
}

CONTRACT_ARTIFACTS = (
    "report.md",
    "report.html",
    "report.pdf",
    "raw_responses.json",
    "evidence.json",
)


@dataclass
class OrderSpec:
    order_id: str
    prompts: list[str]
    product: str = "snapshot"
    fuzz_mode: str = "basic"
    seeds: int = 3
    languages: list[str] = field(default_factory=list)
    strategies: list[str] = field(default_factory=list)
    include_remediation: bool = False
    headline: str | None = None
    workers: int | None = None
    payment_status: str | None = None
    customer_email: str | None = None


@dataclass
class PromptRun:
    index: int
    prompt: str
    slug: str
    run_id: str
    artifacts: dict[str, Path] = field(default_factory=dict)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _first(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def normalize_product(raw: Any) -> str:
    """Map storefront product strings to build_report --report values."""
    if raw is None or raw == "":
        return "snapshot"
    key = str(raw).strip().lower().replace(" ", "_")
    if key not in PRODUCT_ALIASES:
        raise ValueError(
            f"Unknown product {raw!r}. Use snapshot, basis, or both "
            f"(aliases: {', '.join(sorted(PRODUCT_ALIASES))})."
        )
    return PRODUCT_ALIASES[key]


def coerce_prompt_list(raw: Any) -> Any:
    """Accept a list, a single string, or a JSON-encoded list (``\"[]\"``)."""
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if text in {"", "null", "None"}:
            return []
        if text[:1] in {"[", "{"}:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        return text
    return raw


def normalize_prompts(raw: Any) -> list[str]:
    from moyo.publicside.gatherpublicsources.explorer import normalize_prompts as _norm

    coerced = coerce_prompt_list(raw)
    if coerced is None:
        coerced = []
    prompts = _norm(coerced)
    if not prompts:
        raise ValueError(
            "Order prompts are empty (reportStatus is still awaiting_prompts)"
        )
    return prompts


def prompt_slug(index: int, prompt: str) -> str:
    from moyo.publicside.gatherpublicsources.explorer import _slugify

    return f"{index:02d}_{_slugify(prompt)}"


def parse_order(order_id: str, data: dict[str, Any] | None) -> OrderSpec:
    if not data:
        raise ValueError(f"Order {order_id!r} is missing or empty")
    seeds_raw = _first(data, "seeds", "numSeeds", "num_seeds", default=3)
    try:
        seeds = max(1, int(seeds_raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid seeds value: {seeds_raw!r}") from exc

    languages = _first(data, "languages", "extraLanguages", default=[]) or []
    if isinstance(languages, str):
        languages = [part.strip() for part in languages.split(",") if part.strip()]

    strategies = _first(data, "strategies", default=[]) or []
    if isinstance(strategies, str):
        strategies = [part.strip() for part in strategies.split(",") if part.strip()]

    workers_raw = _first(data, "workers", default=None)
    workers = None
    if workers_raw is not None and workers_raw != "":
        workers = max(1, int(workers_raw))

    fuzz_mode = str(_first(data, "fuzzMode", "fuzz_mode", default="basic") or "basic")
    headline = _first(data, "headline", default=None)
    if headline is not None:
        headline = str(headline).strip() or None

    email = _first(data, "customerEmail", "customer_email", default=None)
    if email is not None:
        email = str(email).strip() or None

    return OrderSpec(
        order_id=order_id,
        prompts=normalize_prompts(
            _first(data, "prompts", "customerPrompts", "customer_prompts", "prompt")
        ),
        product=normalize_product(_first(data, "product", default="snapshot")),
        fuzz_mode=fuzz_mode,
        seeds=seeds,
        languages=[str(x) for x in languages],
        strategies=[str(x) for x in strategies],
        include_remediation=bool(
            _first(data, "includeRemediation", "include_remediation", default=False)
        ),
        headline=headline,
        workers=workers,
        payment_status=_first(data, "paymentStatus", "payment_status", default=None),
        customer_email=email,
    )


def work_dir_for(order_id: str) -> Path:
    root = Path(os.environ.get("MOYO_CLOUD_WORK_DIR") or "/tmp/moyo")
    path = root / order_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def serialize_raw_responses(explore_results: Iterable[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in explore_results:
        prompt = getattr(result, "prompt", "")
        for item in getattr(result, "results", []) or []:
            row = asdict(item) if hasattr(item, "__dataclass_fields__") else dict(item)
            row["prompt"] = prompt
            rows.append(row)
    return rows


def build_evidence(run_dir: Path, *, prompt: str | None = None) -> dict[str, Any]:
    """Compact evidence pack for QC: claims + scored findings."""
    claims_path = run_dir / "claims.jsonl"
    report_data_path = run_dir / "report_data.json"
    claims: list[dict[str, Any]] = []
    if claims_path.exists():
        for line in claims_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                claims.append(json.loads(line))
    report_data: dict[str, Any] = {}
    if report_data_path.exists():
        report_data = json.loads(report_data_path.read_text(encoding="utf-8"))
    return {
        "prompt": prompt or report_data.get("topic"),
        "topic": report_data.get("topic"),
        "headline": report_data.get("headline"),
        "counts": report_data.get("counts") or {},
        "findings": report_data.get("findings") or [],
        "claims": claims,
        "explore_meta": report_data.get("explore_meta") or {},
    }


def _count_usable_raw_responses(raw_path: Path) -> tuple[int, int, list[str]]:
    """Return (ok, total, sample_errors) from raw_responses.json."""
    if not raw_path.is_file():
        return 0, 0, ["raw_responses.json missing"]
    try:
        rows = json.loads(raw_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return 0, 0, [f"invalid raw_responses.json: {exc}"]
    if not isinstance(rows, list):
        return 0, 0, ["raw_responses.json is not a list"]
    ok = 0
    errors: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        err = (row.get("error") or "").strip()
        text = (row.get("text") or "").strip()
        if err or not text:
            label = row.get("source_label") or row.get("label") or "unknown"
            reason = err or "(no content returned)"
            if len(errors) < 8:
                errors.append(f"{label}: {reason[:160]}")
        else:
            ok += 1
    return ok, len(rows), errors


def _required_llm_env_presence() -> dict[str, bool]:
    """Which provider env vars the cloud job needs (True = set, not values)."""
    keys = (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "XAI_API_KEY",
        "GEMINI_API_KEY",
        "DASHSCOPE_API_KEY",
        "MOONSHOT_API_KEY",
        "PERPLEXITY_API_KEY",
        "OPENROUTER_API_KEY",
    )
    return {k: bool(os.environ.get(k, "").strip()) for k in keys}


def assert_explore_produced_content(prompt_dir: Path, prompt: str) -> None:
    """Fail loudly when explore wrote a shell report with no usable answers."""
    ok, total, errors = _count_usable_raw_responses(prompt_dir / "raw_responses.json")
    if ok > 0:
        return
    exploration = prompt_dir / "exploration.md"
    failed_lines = 0
    if exploration.is_file():
        text = exploration.read_text(encoding="utf-8")
        failed_lines = text.count("Retrieval failed")
    sample = "; ".join(errors[:5]) if errors else "no error detail"
    raise RuntimeError(
        f"Explore produced 0 usable LLM answers for {prompt!r} "
        f"({ok}/{total} raw responses; exploration 'Retrieval failed' "
        f"markers={failed_lines}). Typical cause: missing provider API keys "
        f"on the Cloud Run job. Sample: {sample}"
    )


def assert_report_has_claims(run_dir: Path, prompt: str) -> None:
    """Fail when build_report finishes with an empty claims inventory."""
    claims_path = run_dir / "claims.jsonl"
    n = 0
    if claims_path.is_file():
        n = sum(1 for line in claims_path.read_text(encoding="utf-8").splitlines() if line.strip())
    if n > 0:
        return
    chunks_path = run_dir / "chunks.jsonl"
    n_chunks = 0
    if chunks_path.is_file():
        n_chunks = sum(
            1 for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    raise RuntimeError(
        f"build_report produced 0 claims for {prompt!r} "
        f"(chunks.jsonl rows={n_chunks}). If chunks are also 0, explore "
        "answers were empty/failed and were filtered before extract. "
        "If chunks > 0, extract/API (MOONSHOT_API_KEY) likely failed or "
        "every chunk was gated as refusal/tiny."
    )


def collect_artifacts(work: Path, run_dir: Path, product: str) -> dict[str, Path]:
    """Resolve the five contract names plus useful extras."""
    found: dict[str, Path] = {}
    output = run_dir / "output"

    md = run_dir / "report.md"
    if md.exists():
        found["report.md"] = md

    html = output / "report.html"
    if html.exists():
        found["report.html"] = html
    elif product == "basis" and (output / "basis-report.html").exists():
        found["report.html"] = output / "basis-report.html"

    pdf = output / "report.pdf"
    if pdf.exists():
        found["report.pdf"] = pdf
    elif product == "basis" and (output / "basis-report.pdf").exists():
        found["report.pdf"] = output / "basis-report.pdf"

    raw = work / "raw_responses.json"
    if raw.exists():
        found["raw_responses.json"] = raw
    evidence = work / "evidence.json"
    if evidence.exists():
        found["evidence.json"] = evidence

    extras = {
        "exploration.md": work / "exploration.md",
        "llm-retrieval-check.md": work / "llm-retrieval-check.md",
        "llm-retrieval-check.json": work / "llm-retrieval-check.json",
        "one-page.pdf": output / "one-page.pdf",
        "one-page.html": output / "one-page.html",
        "basis-report.pdf": output / "basis-report.pdf",
        "basis-report.html": output / "basis-report.html",
        "report_data.json": run_dir / "report_data.json",
        "claims.jsonl": run_dir / "claims.jsonl",
        "report.yaml": run_dir / "report.yaml",
    }
    for name, path in extras.items():
        if path.exists() and name not in found:
            found[name] = path
    return found


def _stage_retrieval_check(prompt_dir: Path, result: Any) -> None:
    """Write LLM Retrieval Check docs next to the per-report artifacts."""
    from moyo.publicside.gatherpublicsources.explorer import write_llm_retrieval_check

    write_llm_retrieval_check(result, prompt_dir)

def retrieval_check_storage_paths(order_id: str, work: Path) -> list[tuple[str, Path]]:
    """GCS object paths for any llm-retrieval-check files under the work dir."""
    dest: list[tuple[str, Path]] = []
    slugs = [
        p.name
        for p in work.iterdir()
        if p.is_dir() and (p / "llm-retrieval-check.md").exists()
    ]
    single = len(slugs) == 1
    for path in work.rglob("llm-retrieval-check.*"):
        if path.suffix not in {".md", ".json"}:
            continue
        slug = path.parent.name
        dest.append((f"reports/{order_id}/{slug}/{path.name}", path))
        if single:
            dest.append((f"reports/{order_id}/{path.name}", path))
    return dest


def _write_report_config(work: Path, spec: OrderSpec, run_id: str) -> Path:
    import yaml

    src = REPO_ROOT / "reports" / "config.yaml"
    cfg = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    cfg.setdefault("output", {})
    cfg["output"]["dir"] = str(work / "report_runs")
    cfg["output"]["run_id"] = run_id
    if spec.headline:
        cfg.setdefault("render", {})
        cfg["render"]["headline"] = spec.headline
    dest = work / "report_config.yaml"
    dest.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return dest


def _run_one_prompt(
    spec: OrderSpec,
    *,
    prompt: str,
    index: int,
    work: Path,
    explore_kwargs: dict[str, Any],
    test_mode: bool,
    progress: Callable[[str], None],
) -> PromptRun:
    from moyo.publicside.gatherpublicsources.explorer import explore_and_save
    from reports.build_report import main as build_report_main

    slug = prompt_slug(index, prompt)
    run_id = f"{spec.order_id}__{slug}"
    prompt_dir = work / slug
    prompt_dir.mkdir(parents=True, exist_ok=True)

    progress(f"[{index}/{len(spec.prompts)}] explore: {prompt}")
    result = explore_and_save(
        prompt,
        output_directory=str(prompt_dir / "explorations"),
        **explore_kwargs,
    )
    if not result.output_path:
        raise RuntimeError(f"Explore did not write exploration.md for {prompt!r}")
    exploration_path = prompt_dir / "exploration.md"
    exploration_path.write_text(
        Path(result.output_path).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _stage_retrieval_check(prompt_dir, result)
    (prompt_dir / "raw_responses.json").write_text(
        json.dumps(serialize_raw_responses([result]), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    assert_explore_produced_content(prompt_dir, prompt)

    cfg_path = _write_report_config(prompt_dir, spec, run_id)
    argv = [
        "--exploration",
        str(exploration_path),
        "--run-id",
        run_id,
        "--config",
        str(cfg_path),
        "--report",
        spec.product,
    ]
    if spec.include_remediation:
        argv.append("--include-remediation")
    if test_mode:
        argv.append("--test")

    progress(f"[{index}/{len(spec.prompts)}] build_report {spec.product}")
    rc = build_report_main(argv)
    if rc != 0:
        raise RuntimeError(f"build_report exited with {rc} for {prompt!r}")

    run_dir = prompt_dir / "report_runs" / run_id
    if not test_mode:
        assert_report_has_claims(run_dir, prompt)
    (prompt_dir / "evidence.json").write_text(
        json.dumps(build_evidence(run_dir, prompt=prompt), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    artifacts = collect_artifacts(prompt_dir, run_dir, spec.product)
    missing = [name for name in CONTRACT_ARTIFACTS if name not in artifacts]
    if missing:
        raise RuntimeError(
            f"Missing required artifacts for {prompt!r}: {', '.join(missing)}"
        )
    return PromptRun(
        index=index,
        prompt=prompt,
        slug=slug,
        run_id=run_id,
        artifacts=artifacts,
    )


def run_moyo(
    spec: OrderSpec,
    *,
    work: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[PromptRun]:
    """Explore and build one report product per prompt."""
    work = work or work_dir_for(spec.order_id)
    work.mkdir(parents=True, exist_ok=True)
    test_mode = os.environ.get("MOYO_CLOUD_TEST", "").strip() in {"1", "true", "yes"}
    if test_mode:
        try:
            from moyo.llm.testing import enable_test_mode

            enable_test_mode()
        except Exception as exc:
            logger.warning("Could not enable LLM test mode: %s", exc)

    def _progress(msg: str) -> None:
        logger.info(msg)
        if progress:
            progress(msg)

    key_presence = _required_llm_env_presence()
    missing_keys = [k for k, present in key_presence.items() if not present]
    _progress(
        "LLM API key env presence: "
        + ", ".join(f"{k}={'yes' if v else 'NO'}" for k, v in key_presence.items())
    )
    if missing_keys:
        logger.warning(
            "Missing LLM API key env vars (explore/extract will fail for those "
            "providers): %s",
            ", ".join(missing_keys),
        )

    explore_kwargs: dict[str, Any] = {
        "fuzz_mode": spec.fuzz_mode,
        "num_seeds": spec.seeds,
        "progress": _progress,
    }
    if spec.languages:
        explore_kwargs["extra_languages"] = spec.languages
    if spec.strategies:
        explore_kwargs["strategies"] = spec.strategies
    if spec.workers is not None:
        explore_kwargs["workers"] = spec.workers

    _progress(
        f"explore {len(spec.prompts)} prompt(s) separately "
        f"fuzz_mode={spec.fuzz_mode} seeds={spec.seeds} product={spec.product}"
    )
    runs: list[PromptRun] = []
    for i, prompt in enumerate(spec.prompts, start=1):
        runs.append(
            _run_one_prompt(
                spec,
                prompt=prompt,
                index=i,
                work=work,
                explore_kwargs=explore_kwargs,
                test_mode=test_mode,
                progress=_progress,
            )
        )
    _progress(f"finished {len(runs)} report(s)")
    return runs


def artifact_manifest(order_id: str, runs: list[PromptRun]) -> dict[str, Any]:
    return {
        "orderId": order_id,
        "reports": [
            {
                "index": run.index,
                "prompt": run.prompt,
                "slug": run.slug,
                "prefix": (
                    f"reports/{order_id}/"
                    if len(runs) == 1
                    else f"reports/{order_id}/{run.slug}/"
                ),
                "files": sorted(run.artifacts),
            }
            for run in runs
        ],
    }


def storage_destinations(
    order_id: str, runs: list[PromptRun]
) -> list[tuple[str, Path]]:
    """(object path, local file) pairs. Single-prompt also writes the flat QC prefix."""
    dest: list[tuple[str, Path]] = []
    single = len(runs) == 1
    for run in runs:
        prefix = f"reports/{order_id}/{run.slug}"
        for name, path in run.artifacts.items():
            dest.append((f"{prefix}/{name}", path))
            if single:
                dest.append((f"reports/{order_id}/{name}", path))
    return dest


def _skip_firebase() -> bool:
    return os.environ.get("MOYO_CLOUD_SKIP_FIREBASE", "").strip() in {
        "1",
        "true",
        "yes",
    }


def _firebase_project_id() -> str | None:
    return (
        os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCLOUD_PROJECT")
        or os.environ.get("GCP_PROJECT")
        or None
    )


def _storage_bucket_name() -> str | None:
    explicit = (
        os.environ.get("STORAGE_BUCKET")
        or os.environ.get("FIREBASE_STORAGE_BUCKET")
        or ""
    ).strip()
    if explicit:
        return explicit
    project = _firebase_project_id()
    if not project:
        return None
    # Default Firebase bucket names (new, then classic).
    return f"{project}.firebasestorage.app"


def _init_firebase_app():
    """Initialize the default Firebase app (Firestore does not need a bucket)."""
    import firebase_admin

    if firebase_admin._apps:
        return
    opts: dict[str, str] = {}
    project = _firebase_project_id()
    if project:
        opts["projectId"] = project
    bucket_name = _storage_bucket_name()
    if bucket_name:
        opts["storageBucket"] = bucket_name
    firebase_admin.initialize_app(options=opts or None)


def _init_firebase():
    from firebase_admin import firestore, storage

    _init_firebase_app()
    db = firestore.client()
    name = _storage_bucket_name()
    bucket = storage.bucket(name) if name else None
    return db, bucket, firestore


def _orders_collection_candidates() -> list[str]:
    primary = (
        os.environ.get("FIRESTORE_ORDERS_COLLECTION")
        or os.environ.get("FIRESTORE_COLLECTION")
        or "reports"
    ).strip() or "reports"
    out = [primary]
    for alt in ("reports", "orders"):
        if alt not in out:
            out.append(alt)
    return out


def _load_order_data(order_id: str) -> tuple[dict[str, Any], Any | None]:
    raw = os.environ.get("ORDER_JSON")
    if raw:
        return json.loads(raw), None
    if _skip_firebase():
        raise ValueError("ORDER_JSON is required when MOYO_CLOUD_SKIP_FIREBASE=1")
    from firebase_admin import firestore

    _init_firebase_app()
    db = firestore.client()
    tried: list[str] = []
    for collection in _orders_collection_candidates():
        ref = db.collection(collection).document(order_id)
        snap = ref.get()
        tried.append(collection)
        if snap.exists:
            logger.info(
                "loaded order %s from Firestore %s/%s",
                order_id,
                collection,
                order_id,
            )
            return snap.to_dict() or {}, ref
    raise ValueError(
        f"Firestore document {order_id!r} not found in collections: {', '.join(tried)}"
    )


def _upload_runs(bucket, order_id: str, runs: list[PromptRun]) -> dict[str, Any]:
    urls = _upload_files(bucket, storage_destinations(order_id, runs))
    manifest = artifact_manifest(order_id, runs)
    manifest_blob = bucket.blob(f"reports/{order_id}/manifest.json")
    manifest_blob.upload_from_string(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        content_type="application/json",
    )
    urls[f"reports/{order_id}/manifest.json"] = (
        f"gs://{bucket.name}/reports/{order_id}/manifest.json"
    )
    return {"manifest": manifest, "urls": urls}


def _upload_files(bucket, pairs: list[tuple[str, Path]]) -> dict[str, str]:
    urls: dict[str, str] = {}
    for object_path, path in pairs:
        if not path.exists():
            continue
        blob = bucket.blob(object_path)
        blob.upload_from_filename(str(path))
        urls[object_path] = f"gs://{bucket.name}/{object_path}"
        logger.info("uploaded %s", urls[object_path])
    return urls


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("MOYO_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    order_id = os.environ.get("ORDER_ID") or ""
    if not order_id and os.environ.get("ORDER_JSON"):
        order_id = "local"
    if not order_id:
        logger.error("ORDER_ID is required")
        return 2

    order_ref = None
    started = utc_now()
    try:
        data, order_ref = _load_order_data(order_id)
        spec = parse_order(order_id, data)
        if spec.payment_status and str(spec.payment_status).lower() != "paid":
            logger.warning(
                "order %s paymentStatus=%s (continuing)",
                spec.order_id,
                spec.payment_status,
            )
        work = work_dir_for(spec.order_id)

        def _mark(fields: dict[str, Any]) -> None:
            if order_ref is None:
                return
            order_ref.update(fields)

        _mark(
            {
                "reportStatus": "generating",
                "generationStartedAt": started,
                "generationFinishedAt": None,
            }
        )

        runs = run_moyo(spec, work=work)
        uploaded: dict[str, Any] = {}
        if not _skip_firebase():
            _db, bucket, _fs = _init_firebase()
            if bucket is None:
                raise RuntimeError(
                    "Storage bucket name not set. Set STORAGE_BUCKET or "
                    "FIREBASE_STORAGE_BUCKET on the Cloud Run job "
                    "(e.g. senteguard-website.firebasestorage.app)."
                )
            uploaded = _upload_runs(bucket, spec.order_id, runs)
        finished = utc_now()
        _mark(
            {
                "reportStatus": "qc_pending",
                "qcStatus": "pending",
                "generationStartedAt": started,
                "generationFinishedAt": finished,
                "artifactPaths": uploaded.get("urls") or {},
                "reportManifest": uploaded.get("manifest")
                or artifact_manifest(spec.order_id, runs),
            }
        )
        logger.info("order %s ready for QC (%d report(s))", spec.order_id, len(runs))
        return 0
    except Exception as exc:
        logger.exception("order %s failed", order_id)
        if not _skip_firebase():
            try:
                work = work_dir_for(order_id)
                _db, bucket, _fs = _init_firebase()
                if bucket is not None:
                    urls = _upload_files(
                        bucket, retrieval_check_storage_paths(order_id, work)
                    )
                    if urls:
                        logger.info(
                            "uploaded %d retrieval-check object(s) after failure",
                            len(urls),
                        )
            except Exception:
                logger.exception("failed to upload retrieval check after error")
        if order_ref is not None:
            try:
                order_ref.update(
                    {
                        "reportStatus": "failed",
                        "generationStartedAt": started,
                        "generationFinishedAt": utc_now(),
                        "error": f"{type(exc).__name__}: {exc}"[:2000],
                    }
                )
            except Exception:
                logger.exception("failed to write error status")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
