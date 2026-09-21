"""Cloud Run / GCE worker: Firestore order → explore → report → Storage.

Triggered with ``ORDER_ID`` set. Reads ``reports/{ORDER_ID}`` (storefront
collection; override with ``FIRESTORE_ORDERS_COLLECTION``), runs the same
``moyo-gather explore`` + ``reports/build_report.py`` path used locally, then
uploads artifacts to the dedicated reports bucket and writes storefront
output paths.

Storefront order fields used here::

    prompts              list[str] | JSON string   required, non-empty
    product              snapshot | basis | both   e.g. "basis"
    productId            moyo_snapshot | moyo_basis (optional)
    paymentStatus        informational
    reportStatus         queued → generating → awaiting_qc | delivered | failed
                         | held (auto-validation failed after retry)
    reportStage          querying_models | analyzing_results |
                         generating_report | validating  (live customer progress)
    qcRequired           false skips human QC (agent orders → delivered).
                         GUI and Checkout default true when the field is missing.
                         Auto-validation (coverage, required sections, retry/hold)
                         runs only when qcRequired is false and generationMode=full.
    qcStatus             pending | not_required
    validationRetryCount 0 on first attempt; 1 after a validation requeue
    generationMode       full | exposure_preview | pdf_from_markdown |
                         rebuild_graphics | from_stage
                         (or a pipeline stage name: parse…render)
    fromStage            parse | extract | cluster | score | synthesize |
                         graphics | render  (rebuilds; same as local --from-stage)
    keepGraphics         reuse assets/*.svg (local --keep-graphics)
    keepContent          reuse report.yaml / report.md (local --keep-content)
    generationStartedAt  ISO-8601 UTC, set when work begins
    generationFinishedAt ISO-8601 UTC, set on success or failure
    output.pdfPath       reports/{storageFolder}/report.pdf
    output.jsonPath      reports/{storageFolder}/report.json
    output.markdownPath  reports/{storageFolder}/report.md
    output.htmlPath      reports/{storageFolder}/report.html
    storageFolder        UTC stamp + topic + short order suffix (GCS prefix)

``awaiting_qc`` is the canonical human-QC state. ``qc_pending`` is accepted
only as a legacy alias when reading status.

One report per prompt. GCS folders are ``reports/{storageFolder}/`` where
``storageFolder`` is ``YYYYMMDDTHHMMSSZ_<topic>_<order-suffix>``
(not the Firestore ``ord_xxx`` id). A single-prompt order writes artifacts
at that prefix (the path QC reads via ``output.pdfPath``). Multi-prompt
orders use ``reports/{storageFolder}/{nn}_{slug}/`` plus a canonical root
``report.json`` and ``manifest.json``.

Ollama is not used in Cloud Run. Rewording, translation, clustering,
summaries, extract, synthesize, and Englishize use Vertex Gemini Flash
(``google/gemini-2.5-flash`` via the job service account). Retrieval
fan-out uses each provider's own key, including Gemini Frontier /
Frontier-1 on AI Studio (``GEMINI_API_KEY``).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moyo.llm.client import (
    ensure_env_loaded,
)
from moyo.order_storage import (
    order_storage_folder,
    report_title_for_firestore,
    sort_stamp_from_text,
)
from moyo.report_storage import (
    DEFAULT_MOYO_REPORTS_BUCKET,
    reports_bucket_name as _storage_bucket_name,
    upload_files as _upload_files,
)
from report_validation import ValidationResult, validate_prompt_runs

logger = logging.getLogger("moyo.cloud_worker")

PRODUCT_ALIASES = {
    "snapshot": "snapshot",
    "snapshot_raw": "snapshot",
    "moyo_snapshot_raw": "snapshot",
    "exposure_report_raw": "snapshot",
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
    "report.json",
    "normalized_responses.json",
    "provider_responses.jsonl",
    "evidence.json",
)

# Exposure Data still delivers claims + structured findings, and now also
# the snapshot one-pager so the hosted link and email include a PDF.
RAW_CONTRACT_ARTIFACTS = (
    "claims.jsonl",
    "report_data.json",
    "one-page.pdf",
    "normalized_responses.json",
    "provider_responses.jsonl",
    "evidence.json",
    "report.json",
)

REBUILD_ARTIFACTS = ("report.md", "report.html", "report.pdf", "report.json")

PIPELINE_STAGES = (
    "parse",
    "extract",
    "cluster",
    "score",
    "synthesize",
    "graphics",
    "render",
)

REPORT_STAGES = (
    "querying_models",
    "analyzing_results",
    "generating_report",
    "validating",
)
MAX_VALIDATION_RETRIES = 1
REBUILD_INPUT_FILES = (
    "report.md",
    "report.yaml",
    "report_data.json",
    "exploration.md",
    "claims.jsonl",
    "chunks.jsonl",
)

REBUILD_MODES = frozenset({"pdf_from_markdown", "rebuild_graphics", "from_stage"})


class ModelRerunIncomplete(RuntimeError):
    """Selected-model retrieval did not all succeed; report rebuild was skipped."""

    def __init__(
        self,
        message: str,
        *,
        incomplete_models: list[str] | None = None,
        failures: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.incomplete_models = [str(x).strip() for x in (incomplete_models or []) if str(x).strip()]
        self.failures = [str(x) for x in (failures or []) if str(x).strip()]


def incomplete_model_ids_from_failures(
    failures: list[str],
    llms: list[Any],
) -> list[str]:
    """Map failure lines back to retrieval model ids (provider:model)."""
    from moyo.llm.registry import retrieval_model_id

    by_label = {
        str(getattr(llm, "label", "") or "").strip(): retrieval_model_id(llm.spec)
        for llm in llms
        if getattr(llm, "spec", None) is not None
    }
    out: list[str] = []
    seen: set[str] = set()
    for line in failures:
        text = str(line or "")
        for label, model_id in by_label.items():
            if not label or label not in text:
                continue
            if model_id in seen:
                continue
            seen.add(model_id)
            out.append(model_id)
    return out


GENERATION_MODE_ALIASES = {
    "full": "full",
    "explore": "full",
    "pdf_from_markdown": "pdf_from_markdown",
    "pdf": "pdf_from_markdown",
    "rebuild_graphics": "rebuild_graphics",
    "graphics_only": "rebuild_graphics",
    "from_stage": "from_stage",
    "fromstage": "from_stage",
    "exposure_preview": "exposure_preview",
    "preview": "exposure_preview",
    "rerun_models": "rerun_models",
    "rerun_model": "rerun_models",
}

CANONICAL_AWAITING_QC = "awaiting_qc"
LEGACY_AWAITING_QC = "qc_pending"
AWAITING_QC_STATUSES = frozenset({CANONICAL_AWAITING_QC, LEGACY_AWAITING_QC})
HUMAN_QC_SOURCES = frozenset({"stripe_checkout", "admin", "gui"})
PRODUCT_IDS = {
    "snapshot": "moyo_snapshot",
    "snapshot_raw": "moyo_snapshot_raw",
    "moyo_snapshot_raw": "moyo_snapshot_raw",
    "basis": "moyo_basis",
    "both": "moyo_basis",
    "moyo_snapshot": "moyo_snapshot",
    "moyo_basis": "moyo_basis",
    "moyo_deep": "moyo_deep",
    "deep": "moyo_deep",
}
OUTPUT_PATH_FILES = {
    "pdfPath": ("report.pdf", "basis-report.pdf", "one-page.pdf"),
    "jsonPath": ("report.json",),
    "markdownPath": ("report.md",),
    "htmlPath": ("report.html", "basis-report.html"),
    "summaryPath": ("one-page.pdf",),
}


@dataclass
class OrderSpec:
    order_id: str
    prompts: list[str]
    product: str = "snapshot"
    fuzz_mode: str = "basic"
    seeds: int = 3
    languages: list[str] = field(default_factory=list)
    retrieval_models: list[str] = field(default_factory=list)
    retrieval_web_search_models: list[str] = field(default_factory=list)
    scan_language_selection: bool = False
    strategies: list[str] = field(default_factory=list)
    include_remediation: bool = False
    headline: str | None = None
    workers: int | None = None
    payment_status: str | None = None
    customer_email: str | None = None
    generation_mode: str = "full"
    from_stage: str | None = None
    keep_graphics: bool | None = None
    keep_content: bool | None = None
    rerun_models: list[str] = field(default_factory=list)
    qc_required: bool = True
    product_id: str = "moyo_snapshot"
    source: str | None = None
    storage_folder: str = ""

    def __post_init__(self) -> None:
        folder = (self.storage_folder or "").strip().strip("/")
        if not sort_stamp_from_text(folder):
            folder = order_storage_folder(self.order_id, self.prompts)
        self.storage_folder = folder

    def reports_prefix(self) -> str:
        return f"reports/{self.storage_folder}"


def snapshot_scan_deadlines(product: str) -> dict[str, int]:
    """No product-specific retrieval deadlines.

    Exposure Data, Snapshot, and Basis share the same per-model timeouts and
    ``max_tokens`` from ``retrieval_llms.json`` / :func:`apply_retrieval_timeout`.
    Admin web search still gets 300s inside :func:`get_retrieval_llms`.
    """
    del product
    return {}


def scan_fuzz_options(spec: OrderSpec) -> dict[str, Any]:
    """Multilingual / translations only when extra languages are selected."""
    languages = [str(x).strip() for x in (spec.languages or []) if str(x).strip()]
    if languages:
        return {
            "fuzz_mode": "multilingual",
            "extra_languages": languages,
            "language_selection_explicit": True,
        }
    # Storefront English-only (including leftover scanLanguageSelection=true
    # without extras) stays basic. GUI multilingual without extras still uses
    # the default language set.
    if spec.scan_language_selection or spec.fuzz_mode != "multilingual":
        return {"fuzz_mode": "basic"}
    return {"fuzz_mode": "multilingual"}


@dataclass(frozen=True)
class RebuildPlan:
    """How to invoke ``reports/build_report.py`` for a Storage rebuild."""

    from_stage: str
    keep_graphics: bool
    keep_content: bool


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


def _mode_key(raw: Any) -> str:
    return str(raw or "").strip().lower().replace("-", "_")


def normalize_generation_mode(raw: Any) -> str:
    """Map storefront generationMode to explore vs Storage rebuild."""
    key = _mode_key(raw) or "full"
    if key in PIPELINE_STAGES:
        return "from_stage"
    return GENERATION_MODE_ALIASES.get(key, "full")


def normalize_from_stage(raw: Any, generation_mode_raw: Any = None) -> str | None:
    """Pipeline stage for a rebuild; same names as local ``--from-stage``."""
    for candidate in (raw, generation_mode_raw):
        key = _mode_key(candidate)
        if key in PIPELINE_STAGES:
            return key
    mode = normalize_generation_mode(generation_mode_raw)
    if mode == "pdf_from_markdown":
        return "render"
    if mode == "rebuild_graphics":
        return "graphics"
    return None


def default_keep_graphics(from_stage: str) -> bool:
    return from_stage == "render"


def default_keep_content(from_stage: str) -> bool:
    return from_stage in {"graphics", "render"}


def resolve_rebuild_plan(spec: OrderSpec) -> RebuildPlan | None:
    """None means a full explore; otherwise rebuild from existing artifacts."""
    if spec.generation_mode in {"full", "exposure_preview", "rerun_models"}:
        return None
    from_stage = spec.from_stage
    if from_stage not in PIPELINE_STAGES:
        if spec.generation_mode == "pdf_from_markdown":
            from_stage = "render"
        elif spec.generation_mode == "rebuild_graphics":
            from_stage = "graphics"
        else:
            raise ValueError(
                f"Rebuild requested (generationMode={spec.generation_mode!r}) "
                f"but fromStage={spec.from_stage!r} is not a pipeline stage."
            )
    keep_graphics = (
        spec.keep_graphics
        if spec.keep_graphics is not None
        else default_keep_graphics(from_stage)
    )
    keep_content = (
        spec.keep_content
        if spec.keep_content is not None
        else default_keep_content(from_stage)
    )
    if spec.generation_mode == "pdf_from_markdown":
        keep_graphics = True if spec.keep_graphics is None else keep_graphics
        keep_content = True if spec.keep_content is None else keep_content
    elif spec.generation_mode == "rebuild_graphics":
        keep_graphics = False if spec.keep_graphics is None else keep_graphics
        keep_content = True if spec.keep_content is None else keep_content
    return RebuildPlan(
        from_stage=from_stage,
        keep_graphics=bool(keep_graphics),
        keep_content=bool(keep_content),
    )


def required_rebuild_files(plan: RebuildPlan) -> tuple[str, ...]:
    stage = plan.from_stage
    if stage in {"parse", "extract"}:
        return ("exploration.md",)
    if stage in {"cluster", "score"}:
        return ("claims.jsonl",)
    if stage in {"synthesize", "graphics"}:
        return ("report_data.json",)
    if plan.keep_content:
        return ("report.md", "report.yaml")
    return ("report_data.json",)


def is_raw_product(spec: OrderSpec) -> bool:
    """True for Exposure Data (scan + one-pager, no human QC)."""
    return spec.product_id == "moyo_snapshot_raw"


def stop_after_for(spec: OrderSpec) -> str | None:
    """Full Exposure Data, Snapshot, and Basis runs render through PDF."""
    return None


def required_artifacts(spec: OrderSpec) -> tuple[str, ...]:
    if spec.generation_mode in REBUILD_MODES:
        return REBUILD_ARTIFACTS
    if is_raw_product(spec):
        return RAW_CONTRACT_ARTIFACTS
    return CONTRACT_ARTIFACTS


def full_build_argv(
    spec: OrderSpec,
    *,
    exploration: Path,
    run_id: str,
    cfg_path: Path,
    test_mode: bool = False,
) -> list[str]:
    """CLI args for a full explore→build_report run."""
    argv: list[str] = [
        "--exploration",
        str(exploration),
        "--run-id",
        run_id,
        "--config",
        str(cfg_path),
        "--report",
        spec.product,
    ]
    if spec.include_remediation:
        argv.append("--include-remediation")
    stop_after = stop_after_for(spec)
    if stop_after:
        argv.extend(["--stop-after", stop_after])
    if test_mode:
        argv.append("--test")
    argv.append("--no-upload")
    return argv


def rebuild_build_argv(
    spec: OrderSpec,
    plan: RebuildPlan,
    *,
    run_id: str,
    cfg_path: Path,
    exploration: Path | None = None,
) -> list[str]:
    """CLI args for ``build_report.main``, matching the local GUI."""
    argv: list[str] = []
    if exploration is not None:
        argv.extend(["--exploration", str(exploration)])
    argv.extend(
        [
            "--run-id",
            run_id,
            "--config",
            str(cfg_path),
            "--report",
            spec.product,
            "--from-stage",
            plan.from_stage,
        ]
    )
    if spec.include_remediation:
        argv.append("--include-remediation")
    if plan.keep_graphics:
        argv.append("--keep-graphics")
    if plan.keep_content:
        argv.append("--keep-content")
    argv.append("--no-upload")
    return argv


def normalize_product_id(raw: Any, product: str) -> str:
    key = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key in PRODUCT_IDS:
        return PRODUCT_IDS[key]
    return PRODUCT_IDS.get(product, "moyo_snapshot")


def _coerce_bool(raw: Any) -> bool | None:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return None
    key = str(raw).strip().lower()
    if key in {"true", "1", "yes", "on"}:
        return True
    if key in {"false", "0", "no", "off"}:
        return False
    return None


def order_requires_human_qc(source: Any) -> bool:
    """Match moyomapwebpage orderRequiresHumanQc: Checkout + admin only."""
    return str(source or "stripe_checkout").strip().lower() in HUMAN_QC_SOURCES


def normalize_qc_required(raw: Any, source: Any = None) -> bool:
    """Honor qcRequired; fall back to source when the field is missing."""
    parsed = _coerce_bool(raw)
    if parsed is not None:
        return parsed
    return order_requires_human_qc(source)


def is_awaiting_qc_status(raw: Any) -> bool:
    """True for canonical awaiting_qc and legacy qc_pending."""
    key = str(raw or "").strip().lower().replace("-", "_")
    return key in AWAITING_QC_STATUSES


# Shuffle needs local embeddings / torch; the Cloud Run image does not ship them.
CLOUD_UNSUPPORTED_STRATEGIES = frozenset({"shuffle"})


def drop_cloud_unsupported_strategies(raw: Any) -> list[str]:
    """Drop strategies the worker image cannot run (currently ``shuffle``)."""
    values: list[str]
    if raw is None:
        values = []
    elif isinstance(raw, str):
        values = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        values = [str(item).strip() for item in raw if str(item).strip()]
    dropped = [item for item in values if item.lower() in CLOUD_UNSUPPORTED_STRATEGIES]
    kept = [item for item in values if item.lower() not in CLOUD_UNSUPPORTED_STRATEGIES]
    if dropped:
        logger.warning(
            "Dropping Cloud-unsupported fuzz strategies %s (need local embeddings). Kept %s.",
            dropped,
            kept or "(mode defaults)",
        )
    return kept


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

    retrieval_models = _first(data, "retrievalModels", "retrieval_models", default=[]) or []
    if isinstance(retrieval_models, str):
        retrieval_models = [
            part.strip() for part in retrieval_models.split(",") if part.strip()
        ]

    retrieval_web_search_models = _first(
        data,
        "retrievalWebSearchModels",
        "retrieval_web_search_models",
        default=[],
    ) or []
    if isinstance(retrieval_web_search_models, str):
        retrieval_web_search_models = [
            part.strip()
            for part in retrieval_web_search_models.split(",")
            if part.strip()
        ]
    retrieval_model_set = {str(x) for x in retrieval_models}
    retrieval_web_search_models = [
        str(x)
        for x in retrieval_web_search_models
        if str(x) in retrieval_model_set
    ]

    scan_language_selection = _coerce_bool(
        _first(
            data,
            "scanLanguageSelection",
            "scan_language_selection",
            default=False,
        )
    )

    strategies = _first(data, "strategies", default=[]) or []
    if isinstance(strategies, str):
        strategies = [part.strip() for part in strategies.split(",") if part.strip()]
    strategies = drop_cloud_unsupported_strategies(strategies)

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

    product_raw = _first(data, "product", default="snapshot")
    product = normalize_product(product_raw)
    source = _first(data, "source", default=None)
    if source is not None:
        source = str(source).strip() or None

    rerun_models = _first(data, "rerunModels", "rerun_models", default=[]) or []
    if isinstance(rerun_models, str):
        rerun_models = [
            part.strip() for part in rerun_models.split(",") if part.strip()
        ]
    rerun_models = [str(x).strip() for x in rerun_models if str(x).strip()]

    return OrderSpec(
        order_id=order_id,
        prompts=normalize_prompts(
            _first(data, "prompts", "customerPrompts", "customer_prompts", "prompt")
        ),
        product=product,
        fuzz_mode=fuzz_mode,
        seeds=seeds,
        languages=[str(x) for x in languages],
        retrieval_models=[str(x) for x in retrieval_models],
        retrieval_web_search_models=[str(x) for x in retrieval_web_search_models],
        scan_language_selection=bool(scan_language_selection),
        strategies=[str(x) for x in strategies],
        include_remediation=bool(
            _first(data, "includeRemediation", "include_remediation", default=False)
        ),
        headline=headline,
        workers=workers,
        payment_status=_first(data, "paymentStatus", "payment_status", default=None),
        customer_email=email,
        generation_mode=normalize_generation_mode(
            _first(data, "generationMode", "generation_mode", default="full")
        ),
        from_stage=normalize_from_stage(
            _first(data, "fromStage", "from_stage", default=None),
            _first(data, "generationMode", "generation_mode", default=None),
        ),
        keep_graphics=_coerce_bool(
            _first(data, "keepGraphics", "keep_graphics", default=None)
        ),
        keep_content=_coerce_bool(
            _first(data, "keepContent", "keep_content", default=None)
        ),
        rerun_models=rerun_models,
        qc_required=normalize_qc_required(
            _first(data, "qcRequired", "qcRequire", "qc_required", default=None),
            source,
        ),
        product_id=normalize_product_id(
            _first(data, "productId", "product_id", default=None) or product_raw,
            product,
        ),
        source=source,
        storage_folder=str(
            _first(data, "storageFolder", "storage_folder", default="") or ""
        ).strip(),
    )


def work_dir_for(order_id: str) -> Path:
    root = Path(os.environ.get("MOYO_CLOUD_WORK_DIR") or "/tmp/moyo")
    path = root / order_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def serialize_normalized_responses(explore_results: Iterable[Any]) -> list[dict[str, Any]]:
    """Compiled/localized retrieval rows (no provider_record / auth headers)."""
    rows: list[dict[str, Any]] = []
    for result in explore_results:
        prompt = getattr(result, "prompt", "")
        for item in getattr(result, "results", []) or []:
            row = asdict(item) if hasattr(item, "__dataclass_fields__") else dict(item)
            row.pop("provider_record", None)
            label = getattr(item, "source_label", None)
            if label:
                row["source_label"] = label
            elif not row.get("source_label"):
                row["source_label"] = (
                    row.get("llm_label") or row.get("label") or row.get("model") or ""
                )
            row["prompt"] = prompt
            rows.append(row)
    return rows


def serialize_raw_responses(explore_results: Iterable[Any]) -> list[dict[str, Any]]:
    """Back-compat alias for :func:`serialize_normalized_responses`."""
    return serialize_normalized_responses(explore_results)


def serialize_provider_responses(explore_results: Iterable[Any]) -> list[dict[str, Any]]:
    """One redacted provider payload per model × probe (no sensitive headers)."""
    from moyo.llm.content_filter import redact_secrets

    rows: list[dict[str, Any]] = []
    for result in explore_results:
        prompt = getattr(result, "prompt", "")
        for item in getattr(result, "results", []) or []:
            record = getattr(item, "provider_record", None)
            if isinstance(record, dict) and record:
                row = dict(record)
            else:
                row = {
                    "seed": getattr(item, "seed", None),
                    "seed_index": getattr(item, "seed_index", 0),
                    "llm_index": getattr(item, "llm_index", 0),
                    "strategy": getattr(item, "strategy", None),
                    "language": getattr(item, "language", None),
                    "llm_label": getattr(item, "llm_label", None),
                    "provider": getattr(item, "provider", None),
                    "model": getattr(item, "model", None),
                    "error": getattr(item, "error", None),
                    "content": getattr(item, "original_text", None)
                    or getattr(item, "text", None)
                    or "",
                }
            row["prompt"] = prompt
            rows.append(redact_secrets(row))
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


def _load_json_file(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def compact_finding(row: dict[str, Any]) -> dict[str, Any]:
    """Agent-facing finding: claim, citations, scores — not pipeline internals."""
    claim = row.get("claim") or row.get("text")
    out: dict[str, Any] = {}
    if row.get("claim_id"):
        out["claim_id"] = row["claim_id"]
    if claim:
        out["claim"] = claim
    for key in (
        "status",
        "sensitivity",
        "specificity",
        "novelty",
        "confidence",
        "category",
        "language",
    ):
        if row.get(key) is not None:
            out[key] = row[key]
    models = list(row.get("source_models") or [])
    if not models and row.get("source_model"):
        models = [row["source_model"]]
    if models:
        out["source_models"] = models
    citations = list(row.get("citations") or [])
    if citations:
        out["citations"] = citations
    return out


def collect_citations(findings: list[dict[str, Any]]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for finding in findings:
        for cite in finding.get("citations") or []:
            key = json.dumps(cite, sort_keys=True, default=str) if isinstance(cite, dict) else str(cite)
            if key in seen:
                continue
            seen.add(key)
            out.append(cite)
    return out


def prompt_report_section(
    spec: OrderSpec,
    run: PromptRun,
    *,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifacts = run.artifacts
    evidence = evidence or _load_json_file(artifacts.get("evidence.json"))
    report_data = _load_json_file(artifacts.get("report_data.json"))
    findings_raw = evidence.get("findings") or report_data.get("findings") or []
    findings = [
        compact_finding(row)
        for row in findings_raw
        if isinstance(row, dict)
    ]
    findings = [row for row in findings if row.get("claim") or row.get("claim_id")]
    return {
        "index": run.index,
        "prompt": run.prompt,
        "slug": run.slug,
        "headline": evidence.get("headline") or report_data.get("headline"),
        "topic": evidence.get("topic") or report_data.get("topic"),
        "counts": evidence.get("counts") or report_data.get("counts") or {},
        "findings": findings,
        "citations": collect_citations(findings),
    }


def build_canonical_report(
    spec: OrderSpec,
    runs: list[PromptRun],
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Machine-readable report.json for agents: findings, citations, ids."""
    sections = [prompt_report_section(spec, run) for run in runs]
    findings: list[dict[str, Any]] = []
    citations: list[Any] = []
    seen_cites: set[str] = set()
    for section in sections:
        findings.extend(section.get("findings") or [])
        for cite in section.get("citations") or []:
            key = json.dumps(cite, sort_keys=True, default=str) if isinstance(cite, dict) else str(cite)
            if key in seen_cites:
                continue
            seen_cites.add(key)
            citations.append(cite)
    payload: dict[str, Any] = {
        "orderId": spec.order_id,
        "product": spec.product,
        "productId": spec.product_id,
        "prompts": list(spec.prompts),
        "generationMode": spec.generation_mode,
        "generatedAt": generated_at or utc_now(),
        "counts": {"findings": len(findings), "reports": len(runs)},
        "findings": findings,
        "citations": citations,
    }
    if len(sections) == 1:
        payload["prompt"] = sections[0].get("prompt")
        payload["headline"] = sections[0].get("headline")
        payload["topic"] = sections[0].get("topic")
        if sections[0].get("counts"):
            payload["counts"] = {
                **sections[0]["counts"],
                "findings": len(findings),
                "reports": 1,
            }
    else:
        payload["reports"] = sections
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_prompt_report_json(
    prompt_dir: Path,
    spec: OrderSpec,
    run: PromptRun,
    *,
    evidence: dict[str, Any] | None = None,
) -> Path:
    section = prompt_report_section(spec, run, evidence=evidence)
    payload = {
        "orderId": spec.order_id,
        "product": spec.product,
        "productId": spec.product_id,
        "prompts": list(spec.prompts),
        "generationMode": spec.generation_mode,
        "generatedAt": utc_now(),
        **section,
    }
    path = write_json(prompt_dir / "report.json", payload)
    run.artifacts["report.json"] = path
    return path


def write_canonical_report_json(
    spec: OrderSpec, runs: list[PromptRun], dest: Path
) -> Path:
    return write_json(dest, build_canonical_report(spec, runs))


def output_paths(folder: str, urls: dict[str, str]) -> dict[str, str | None]:
    """Storefront output.pdfPath / jsonPath / markdownPath / htmlPath."""
    prefix = f"reports/{folder.strip('/')}/"

    def pick(filenames: tuple[str, ...]) -> str | None:
        for name in filenames:
            key = f"{prefix}{name}"
            if key in urls:
                return key
        for name in filenames:
            suffix = f"/{name}"
            nested = [
                key
                for key in urls
                if key.startswith(prefix) and key.endswith(suffix)
            ]
            if nested:
                nested.sort(key=lambda item: (item.count("/"), item))
                return nested[0]
        return None

    return {field: pick(names) for field, names in OUTPUT_PATH_FILES.items()}


def success_update_fields(
    spec: OrderSpec,
    *,
    started: str,
    finished: str,
    urls: dict[str, str],
    manifest: dict[str, Any],
    validation: ValidationResult | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "generationStartedAt": started,
        "generationFinishedAt": finished,
        "generationMode": spec.generation_mode,
        "artifactPaths": urls,
        "reportManifest": manifest,
        "output": output_paths(spec.storage_folder, urls),
        "storageFolder": spec.storage_folder,
        "qcRequired": spec.qc_required,
        "reportStage": None,
        "error": None,
        "incompleteRetrievalModels": [],
    }
    if title:
        fields["title"] = title
    if spec.qc_required:
        fields["reportStatus"] = CANONICAL_AWAITING_QC
        fields["qcStatus"] = "pending"
    else:
        fields["reportStatus"] = "delivered"
        fields["qcStatus"] = "not_required"
        fields["deliveredAt"] = finished
    if validation is not None:
        fields["validation"] = validation.to_firestore()
    return fields


def _job_launch_clear_value() -> Any:
    """Drop jobLaunchStatus so startReport can launch a validation retry."""
    try:
        from firebase_admin import firestore as fs

        return fs.DELETE_FIELD
    except Exception:
        return None


def should_auto_validate(spec: OrderSpec) -> bool:
    """Coverage/section checks gate auto-email, not human-QC drafts or rebuilds."""
    return (not spec.qc_required) and spec.generation_mode == "full"


def delivery_action(
    spec: OrderSpec,
    *,
    validation: ValidationResult | None,
    retry_count: int,
) -> str:
    if spec.qc_required:
        return "qc"
    if not should_auto_validate(spec) or validation is None:
        return "deliver"
    if validation.ok:
        return "deliver"
    if retry_count < MAX_VALIDATION_RETRIES:
        return "retry"
    return "hold"


def retry_update_fields(
    *,
    validation: ValidationResult,
    retry_count: int,
    started: str,
    finished: str,
) -> dict[str, Any]:
    return {
        "reportStatus": "queued",
        "reportStage": None,
        "validationRetryCount": retry_count,
        "validation": validation.to_firestore(),
        "generationStartedAt": started,
        "generationFinishedAt": finished,
        "jobLaunchStatus": _job_launch_clear_value(),
        "error": validation.summary()[:2000],
    }


def hold_update_fields(
    *,
    validation: ValidationResult,
    retry_count: int,
    started: str,
    finished: str,
) -> dict[str, Any]:
    return {
        "reportStatus": "held",
        "reportStage": "validating",
        "validationRetryCount": retry_count,
        "validation": validation.to_firestore(),
        "holdNotifyStatus": "pending",
        "generationStartedAt": started,
        "generationFinishedAt": finished,
        "error": validation.summary()[:2000],
    }


def parse_validation_retry_count(data: dict[str, Any] | None) -> int:
    raw = (data or {}).get("validationRetryCount", 0)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def _count_usable_raw_responses(raw_path: Path) -> tuple[int, int, list[str]]:
    """Return (ok, total, sample_errors) from normalized_responses.json."""
    if not raw_path.is_file():
        legacy = raw_path.with_name("raw_responses.json")
        if legacy.is_file():
            raw_path = legacy
        else:
            return 0, 0, ["normalized_responses.json missing"]
    try:
        rows = json.loads(raw_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return 0, 0, [f"invalid {raw_path.name}: {exc}"]
    if not isinstance(rows, list):
        return 0, 0, [f"{raw_path.name} is not a list"]
    ok = 0
    errors: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        err = (row.get("error") or "").strip()
        text = (row.get("text") or "").strip()
        if err or not text:
            label = (
                row.get("source_label")
                or row.get("llm_label")
                or row.get("label")
                or row.get("model")
                or "unknown"
            )
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
        "OPENROUTER_API_KEY",
    )
    return {k: bool(os.environ.get(k, "").strip()) for k in keys}


def note_explore_gaps(prompt_dir: Path, prompt: str) -> list[str]:
    """Log failed/empty retrievals; never abort — the report uses what succeeded."""
    ok, total, errors = _count_usable_raw_responses(prompt_dir / "normalized_responses.json")
    if not errors and ok > 0:
        return []
    sample = "; ".join(errors[:5]) if errors else "no error detail"
    note = (
        f"Explore: {ok}/{total} usable LLM answers for {prompt!r}. "
        f"Failed/empty: {sample}"
    )
    logger.warning(note)
    return [note]


def note_report_gaps(run_dir: Path, prompt: str) -> list[str]:
    """Log an empty claims inventory; still allow the report to be delivered."""
    claims_path = run_dir / "claims.jsonl"
    n = 0
    if claims_path.is_file():
        n = sum(
            1
            for line in claims_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if n > 0:
        return []
    chunks_path = run_dir / "chunks.jsonl"
    n_chunks = 0
    if chunks_path.is_file():
        n_chunks = sum(
            1
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    note = (
        f"build_report produced 0 claims for {prompt!r} "
        f"(chunks.jsonl rows={n_chunks}); report built from remaining artifacts."
    )
    logger.warning(note)
    return [note]


# Back-compat aliases used by older tests / callers.
def assert_explore_produced_content(prompt_dir: Path, prompt: str) -> list[str]:
    return note_explore_gaps(prompt_dir, prompt)


def assert_report_has_claims(run_dir: Path, prompt: str) -> list[str]:
    return note_report_gaps(run_dir, prompt)


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

    raw = work / "normalized_responses.json"
    if not raw.exists():
        raw = run_dir / "normalized_responses.json"
    if not raw.exists():
        raw = work / "raw_responses.json"
    if not raw.exists():
        raw = run_dir / "raw_responses.json"
    if raw.exists():
        found["normalized_responses.json"] = raw
        if raw.name == "raw_responses.json":
            found["raw_responses.json"] = raw
    provider = work / "provider_responses.jsonl"
    if not provider.exists():
        provider = run_dir / "provider_responses.jsonl"
    if provider.exists():
        found["provider_responses.jsonl"] = provider
    evidence = work / "evidence.json"
    if evidence.exists():
        found["evidence.json"] = evidence
    report_json = work / "report.json"
    if report_json.exists():
        found["report.json"] = report_json

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
        "chunks.jsonl": run_dir / "chunks.jsonl",
        "extract_issues.json": run_dir / "extract_issues.json",
        "extract_done.jsonl": run_dir / "extract_done.jsonl",
        "report.yaml": run_dir / "report.yaml",
    }
    for name, path in extras.items():
        if path.exists() and name not in found:
            found[name] = path
    assets = run_dir / "assets"
    if assets.is_dir():
        for path in assets.rglob("*"):
            if path.is_file():
                found[f"assets/{path.relative_to(assets).as_posix()}"] = path
    return found


def _stage_retrieval_check(prompt_dir: Path, result: Any) -> None:
    """Write LLM Retrieval Check docs next to the per-report artifacts."""
    from moyo.publicside.gatherpublicsources.explorer import write_llm_retrieval_check

    write_llm_retrieval_check(result, prompt_dir)

def retrieval_check_storage_paths(folder: str, work: Path) -> list[tuple[str, Path]]:
    """GCS object paths for any llm-retrieval-check files under the work dir."""
    dest: list[tuple[str, Path]] = []
    slugs = [
        p.name
        for p in work.iterdir()
        if p.is_dir() and (p / "llm-retrieval-check.md").exists()
    ]
    single = len(slugs) == 1
    prefix = f"reports/{folder.strip('/')}"
    for path in work.rglob("llm-retrieval-check.*"):
        if path.suffix not in {".md", ".json"}:
            continue
        slug = path.parent.name
        if single:
            dest.append((f"{prefix}/{path.name}", path))
        else:
            dest.append((f"{prefix}/{slug}/{path.name}", path))
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
    from moyo.llm.utility import running_in_cloud, vertex_flash_hosted_config

    if running_in_cloud():
        # Desktop YAML uses Ollama (cluster) and Kimi (extract/synthesize).
        # Cloud Run uses Vertex Flash for all of those stages.
        for key in ("extract", "cluster", "synthesize"):
            cfg[key] = vertex_flash_hosted_config(cfg.get(key) or {})
        cfg["cluster"].setdefault("temperature", 0.1)
        cfg["cluster"].setdefault("max_tokens", 4000)
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
    set_stage: Callable[[str], None] | None = None,
) -> PromptRun:
    from moyo.publicside.gatherpublicsources.explorer import explore_and_save
    from reports.build_report import main as build_report_main

    slug = prompt_slug(index, prompt)
    run_id = f"{spec.order_id}__{slug}"
    prompt_dir = work / slug
    prompt_dir.mkdir(parents=True, exist_ok=True)

    def _stage(name: str) -> None:
        if set_stage:
            set_stage(name)
        progress(f"[{index}/{len(spec.prompts)}] stage {name}")

    _stage("querying_models")
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
    _stage("analyzing_results")
    _stage_retrieval_check(prompt_dir, result)
    (prompt_dir / "normalized_responses.json").write_text(
        json.dumps(serialize_normalized_responses([result]), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    provider_rows = serialize_provider_responses([result])
    (prompt_dir / "provider_responses.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in provider_rows),
        encoding="utf-8",
    )
    pipeline_notes = note_explore_gaps(prompt_dir, prompt)

    cfg_path = _write_report_config(prompt_dir, spec, run_id)
    argv = full_build_argv(
        spec,
        exploration=exploration_path,
        run_id=run_id,
        cfg_path=cfg_path,
        test_mode=test_mode,
    )

    _stage("generating_report")
    stop_after = stop_after_for(spec)
    progress(
        f"[{index}/{len(spec.prompts)}] build_report {spec.product}"
        + (f" stop-after={stop_after}" if stop_after else "")
    )
    rc = build_report_main(argv)
    if rc != 0:
        raise RuntimeError(f"build_report exited with {rc} for {prompt!r}")

    run_dir = prompt_dir / "report_runs" / run_id
    if not test_mode:
        pipeline_notes.extend(note_report_gaps(run_dir, prompt))
    evidence = build_evidence(run_dir, prompt=prompt)
    if pipeline_notes:
        evidence["pipeline_notes"] = pipeline_notes
    (prompt_dir / "evidence.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    artifacts = collect_artifacts(prompt_dir, run_dir, spec.product)
    run = PromptRun(
        index=index,
        prompt=prompt,
        slug=slug,
        run_id=run_id,
        artifacts=artifacts,
    )
    write_prompt_report_json(prompt_dir, spec, run, evidence=evidence)
    missing = [name for name in required_artifacts(spec) if name not in run.artifacts]
    if missing:
        raise RuntimeError(
            f"Missing required artifacts for {prompt!r}: {', '.join(missing)}"
        )
    return run


REBUILD_STAGE_FILES = (
    "report.md",
    "report.yaml",
    "report_data.json",
    "claims.jsonl",
    "chunks.jsonl",
    "exploration.md",
    "extract_done.jsonl",
    "normalized_responses.json",
    "provider_responses.jsonl",
    "raw_responses.json",
    "evidence.json",
)


def download_order_prefix(bucket, folder: str, dest: Path) -> Path:
    """Copy gs://…/reports/{storageFolder}/** into dest."""
    prefix = f"reports/{folder.strip('/')}/"
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for blob in bucket.list_blobs(prefix=prefix):
        rel = blob.name[len(prefix) :]
        if not rel or rel.endswith("/"):
            continue
        path = dest / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(path))
        count += 1
    if count == 0:
        raise RuntimeError(
            f"No artifacts in gs://{bucket.name}/{prefix} to rebuild from."
        )
    return dest


def copy_rebuild_sources(src: Path, run_dir: Path, prompt_dir: Path) -> None:
    """Stage existing QC files into the build_report run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for name in REBUILD_STAGE_FILES:
        item = src / name
        if not item.is_file():
            continue
        shutil.copy2(item, run_dir / name)
        if name in {
            "raw_responses.json",
            "normalized_responses.json",
            "provider_responses.jsonl",
            "evidence.json",
            "exploration.md",
        }:
            shutil.copy2(item, prompt_dir / name)
    assets_src = src / "assets"
    if assets_src.is_dir():
        shutil.copytree(assets_src, run_dir / "assets", dirs_exist_ok=True)
    images_src = src / "images"
    if images_src.is_dir():
        shots = run_dir / "assets" / "screenshots"
        shots.mkdir(parents=True, exist_ok=True)
        for img in images_src.iterdir():
            if img.is_file():
                shutil.copy2(img, shots / img.name)


def rebuild_topic_dirs(gcs_root: Path, spec: OrderSpec) -> list[tuple[int, str, Path]]:
    def has_inputs(folder: Path) -> bool:
        return any((folder / name).is_file() for name in REBUILD_INPUT_FILES)

    if has_inputs(gcs_root):
        prompt = spec.prompts[0] if spec.prompts else "report"
        return [(1, prompt, gcs_root)]

    topics: list[tuple[int, str, Path]] = []
    for i, prompt in enumerate(spec.prompts, start=1):
        folder = gcs_root / prompt_slug(i, prompt)
        if folder.is_dir() and has_inputs(folder):
            topics.append((i, prompt, folder))
    if topics:
        return topics
    for child in sorted(p for p in gcs_root.iterdir() if p.is_dir()):
        if has_inputs(child):
            topics.append((len(topics) + 1, child.name, child))
    if not topics:
        raise RuntimeError(
            "No exploration.md / claims.jsonl / report.md / report_data.json found to rebuild."
        )
    return topics


def _missing_rebuild_files(run_dir: Path, plan: RebuildPlan) -> list[str]:
    missing = [name for name in required_rebuild_files(plan) if not (run_dir / name).is_file()]
    if plan.from_stage == "render" and plan.keep_content:
        if (run_dir / "report.yaml").is_file() or (run_dir / "report.md").is_file():
            return [name for name in missing if name not in {"report.md", "report.yaml"}]
    return missing


def _write_explore_sidecars(prompt_dir: Path, result: Any) -> None:
    """normalized_responses / provider_responses / retrieval-check next to exploration.md."""
    _stage_retrieval_check(prompt_dir, result)
    (prompt_dir / "normalized_responses.json").write_text(
        json.dumps(serialize_normalized_responses([result]), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    provider_rows = serialize_provider_responses([result])
    (prompt_dir / "provider_responses.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in provider_rows),
        encoding="utf-8",
    )


def run_rerun_models(
    spec: OrderSpec,
    *,
    bucket,
    work: Path | None = None,
    progress: Callable[[str], None] | None = None,
    set_stage: Callable[[str], None] | None = None,
) -> tuple[list[PromptRun], list[str], list[str]]:
    """Overwrite selected models in exploration.md, then rebuild from parse if they succeed.

    Returns ``(runs, failure_lines, incomplete_model_ids)``.
    """
    from moyo.llm.registry import get_retrieval_llms
    from moyo.publicside.gatherpublicsources.explorer import rerun_exploration_models
    from reports.build_report import main as build_report_main

    wanted = [str(x).strip() for x in (spec.rerun_models or []) if str(x).strip()]
    if not wanted:
        raise ValueError("rerun_models requires rerunModels (one or more retrieval ids)")

    from moyo.llm.client import RETRIEVAL_TIMEOUT_RERUN

    llms = get_retrieval_llms(
        wanted,
        web_search_model_ids=set(spec.retrieval_web_search_models or []),
        require_match=True,
        timeout=RETRIEVAL_TIMEOUT_RERUN,
    )
    workers = spec.workers

    work = work or work_dir_for(spec.order_id)
    work.mkdir(parents=True, exist_ok=True)

    def _progress(msg: str) -> None:
        logger.info(msg)
        if progress:
            progress(msg)

    def _stage(name: str) -> None:
        if set_stage:
            set_stage(name)
        _progress(f"stage {name}")

    gcs_root = download_order_prefix(bucket, spec.storage_folder, work / "gcs")
    _progress(
        f"rerun models={wanted} then rebuild from parse "
        f"gs://{bucket.name}/reports/{spec.storage_folder}/"
    )
    plan = RebuildPlan(from_stage="parse", keep_graphics=False, keep_content=False)
    runs: list[PromptRun] = []
    failures: list[str] = []
    _stage("querying_models")
    for index, prompt, src in rebuild_topic_dirs(gcs_root, spec):
        slug = prompt_slug(index, prompt)
        run_id = f"{spec.order_id}__{slug}"
        prompt_dir = work / slug
        run_dir = prompt_dir / "report_runs" / run_id
        copy_rebuild_sources(src, run_dir, prompt_dir)
        exploration = run_dir / "exploration.md"
        if not exploration.is_file():
            alt = prompt_dir / "exploration.md"
            exploration = alt if alt.is_file() else exploration
        if not exploration.is_file():
            raise RuntimeError(
                f"Cannot rerun models for {prompt!r}; missing exploration.md."
            )
        _progress(f"[{index}] overwrite retrieval for {', '.join(wanted)}: {prompt}")
        outcome = rerun_exploration_models(
            exploration.read_text(encoding="utf-8"),
            llms,
            progress=_progress,
            workers=workers,
        )
        exploration.write_text(outcome.explore.markdown, encoding="utf-8")
        (prompt_dir / "exploration.md").write_text(
            outcome.explore.markdown, encoding="utf-8"
        )
        _write_explore_sidecars(prompt_dir, outcome.explore)
        if outcome.failures:
            failures.extend(f"{prompt}: {item}" for item in outcome.failures)
            artifacts = collect_artifacts(prompt_dir, run_dir, spec.product)
            run = PromptRun(
                index=index,
                prompt=prompt,
                slug=slug,
                run_id=run_id,
                artifacts=artifacts,
            )
            write_prompt_report_json(prompt_dir, spec, run)
            runs.append(run)
            continue
        _stage("generating_report")
        cfg_path = _write_report_config(prompt_dir, spec, run_id)
        argv = rebuild_build_argv(
            spec,
            plan,
            run_id=run_id,
            cfg_path=cfg_path,
            exploration=exploration,
        )
        _progress(f"[{index}] rebuild from parse after successful model rerun: {prompt}")
        rc = build_report_main(argv)
        if rc != 0:
            raise RuntimeError(f"rebuild from parse exited {rc} for {prompt!r}")
        artifacts = collect_artifacts(prompt_dir, run_dir, spec.product)
        run = PromptRun(
            index=index,
            prompt=prompt,
            slug=slug,
            run_id=run_id,
            artifacts=artifacts,
        )
        write_prompt_report_json(prompt_dir, spec, run)
        missing_out = [name for name in required_artifacts(spec) if name not in run.artifacts]
        if missing_out:
            raise RuntimeError(
                f"Missing after model rerun rebuild for {prompt!r}: {', '.join(missing_out)}"
            )
        runs.append(run)
    if failures:
        _progress(
            "model rerun saved overwrites but did not rebuild: " + "; ".join(failures[:8])
        )
    else:
        _progress(f"finished model rerun + parse rebuild of {len(runs)} report(s)")
    return runs, failures, incomplete_model_ids_from_failures(failures, llms)


def run_rebuild(
    spec: OrderSpec,
    *,
    bucket,
    work: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[PromptRun]:
    """Re-run build_report from a pipeline stage. Does not re-run explore."""
    from reports.build_report import main as build_report_main

    plan = resolve_rebuild_plan(spec)
    if plan is None:
        raise RuntimeError(
            f"generationMode={spec.generation_mode!r} is not a rebuild "
            "(expected from_stage, pdf_from_markdown, or rebuild_graphics)."
        )

    work = work or work_dir_for(spec.order_id)
    work.mkdir(parents=True, exist_ok=True)

    def _progress(msg: str) -> None:
        logger.info(msg)
        if progress:
            progress(msg)

    gcs_root = download_order_prefix(bucket, spec.storage_folder, work / "gcs")
    _progress(
        f"rebuild from-stage={plan.from_stage} keep_graphics={plan.keep_graphics} "
        f"keep_content={plan.keep_content} "
        f"gs://{bucket.name}/reports/{spec.storage_folder}/"
    )
    runs: list[PromptRun] = []
    for index, prompt, src in rebuild_topic_dirs(gcs_root, spec):
        slug = prompt_slug(index, prompt)
        run_id = f"{spec.order_id}__{slug}"
        prompt_dir = work / slug
        run_dir = prompt_dir / "report_runs" / run_id
        copy_rebuild_sources(src, run_dir, prompt_dir)
        missing = _missing_rebuild_files(run_dir, plan)
        if missing:
            raise RuntimeError(
                f"Cannot rebuild from {plan.from_stage} for {prompt!r}; "
                f"missing {', '.join(missing)}."
            )
        exploration = run_dir / "exploration.md"
        if not exploration.is_file():
            alt = prompt_dir / "exploration.md"
            exploration = alt if alt.is_file() else None
        cfg_path = _write_report_config(prompt_dir, spec, run_id)
        argv = rebuild_build_argv(
            spec,
            plan,
            run_id=run_id,
            cfg_path=cfg_path,
            exploration=exploration,
        )
        _progress(f"[{index}] rebuild from {plan.from_stage}: {prompt}")
        rc = build_report_main(argv)
        if rc != 0:
            raise RuntimeError(f"rebuild from {plan.from_stage} exited {rc} for {prompt!r}")
        artifacts = collect_artifacts(prompt_dir, run_dir, spec.product)
        run = PromptRun(
            index=index,
            prompt=prompt,
            slug=slug,
            run_id=run_id,
            artifacts=artifacts,
        )
        write_prompt_report_json(prompt_dir, spec, run)
        missing_out = [name for name in required_artifacts(spec) if name not in run.artifacts]
        if missing_out:
            raise RuntimeError(
                f"Missing after rebuild for {prompt!r}: {', '.join(missing_out)}"
            )
        runs.append(run)
    _progress(f"finished rebuild of {len(runs)} report(s) from {plan.from_stage}")
    return runs


def run_moyo(
    spec: OrderSpec,
    *,
    work: Path | None = None,
    progress: Callable[[str], None] | None = None,
    set_stage: Callable[[str], None] | None = None,
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
        "num_seeds": spec.seeds,
        "progress": _progress,
        **scan_fuzz_options(spec),
    }
    from moyo.llm.registry import get_retrieval_llms
    from moyo.llm.vertex import is_vertex_openai_url

    explore_kwargs["retrieval_llms"] = get_retrieval_llms(
        spec.retrieval_models or None,
        web_search_model_ids=set(spec.retrieval_web_search_models or []),
        **snapshot_scan_deadlines(spec.product),
    )
    scan_llms = explore_kwargs["retrieval_llms"]
    _progress(
        f"this scan will query {len(scan_llms)} retrieval LLM(s)"
        + (
            f" (order selected {len(spec.retrieval_models)} id(s))"
            if spec.retrieval_models
            else " (configured default set)"
        )
    )
    for llm in scan_llms:
        dest = llm.spec.base_url or llm.spec.provider
        via = "vertex" if is_vertex_openai_url(llm.spec.base_url) else llm.spec.provider
        _progress(f"scan retrieval LLM {llm.label}: {llm.spec.model} via {via} ({dest})")
    if spec.strategies:
        explore_kwargs["strategies"] = spec.strategies
    if spec.workers is not None:
        explore_kwargs["workers"] = spec.workers

    _progress(
        f"explore {len(spec.prompts)} prompt(s) separately "
        f"fuzz_mode={explore_kwargs.get('fuzz_mode')} seeds={spec.seeds} "
        f"product={spec.product}"
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
                set_stage=set_stage,
            )
        )
    _progress(f"finished {len(runs)} report(s)")
    return runs


def artifact_manifest(order_id: str, runs: list[PromptRun], *, folder: str) -> dict[str, Any]:
    prefix_root = f"reports/{folder.strip('/')}"
    return {
        "orderId": order_id,
        "storageFolder": folder,
        "reports": [
            {
                "index": run.index,
                "prompt": run.prompt,
                "slug": run.slug,
                "prefix": (
                    f"{prefix_root}/"
                    if len(runs) == 1
                    else f"{prefix_root}/{run.slug}/"
                ),
                "files": sorted(run.artifacts),
            }
            for run in runs
        ],
    }


def storage_destinations(
    folder: str, runs: list[PromptRun]
) -> list[tuple[str, Path]]:
    """(object path, local file) pairs. One copy per file.

    Single-prompt orders land at ``reports/{storageFolder}/``.
    Multi-prompt orders land at ``reports/{storageFolder}/{slug}/``.
    """
    dest: list[tuple[str, Path]] = []
    single = len(runs) == 1
    root = f"reports/{folder.strip('/')}"
    for run in runs:
        prefix = root if single else f"{root}/{run.slug}"
        for name, path in run.artifacts.items():
            dest.append((f"{prefix}/{name}", path))
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


def _upload_runs(
    bucket, spec: OrderSpec, runs: list[PromptRun], *, work: Path
) -> dict[str, Any]:
    canonical = write_canonical_report_json(spec, runs, work / "report.json")
    prefix = spec.reports_prefix()
    dest = dict(storage_destinations(spec.storage_folder, runs))
    dest[f"{prefix}/report.json"] = canonical
    urls = _upload_files(bucket, list(dest.items()))
    manifest = artifact_manifest(
        spec.order_id, runs, folder=spec.storage_folder
    )
    manifest_blob = bucket.blob(f"{prefix}/manifest.json")
    manifest_blob.upload_from_string(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        content_type="application/json",
    )
    urls[f"{prefix}/manifest.json"] = f"gs://{bucket.name}/{prefix}/manifest.json"
    return {"manifest": manifest, "urls": urls}


def run_exposure_preview(
    spec: OrderSpec,
    *,
    mark: Callable[[dict[str, Any]], None],
    started: str,
) -> int:
    """Write preview.json from the cheap topic preprocessor. No model calls."""
    from moyo.exposure_preview import estimate_exposure_preview

    topic = spec.prompts[0] if spec.prompts else ""
    preview = estimate_exposure_preview(topic)
    work = work_dir_for(spec.order_id)
    work.mkdir(parents=True, exist_ok=True)
    preview_path = work / "preview.json"
    preview_path.write_text(json.dumps(preview, indent=2, ensure_ascii=False), encoding="utf-8")

    urls: dict[str, str] = {}
    if not _skip_firebase():
        _db, bucket, _fs = _init_firebase()
        if bucket is None:
            raise RuntimeError("Storage bucket name not set for exposure preview.")
        object_path = f"{spec.reports_prefix()}/preview.json"
        urls = _upload_files(bucket, [(object_path, preview_path)])

    finished = utc_now()
    mark(
        {
            "reportStatus": "delivered",
            "qcRequired": False,
            "qcStatus": "not_required",
            "generationMode": "exposure_preview",
            "generationStartedAt": started,
            "generationFinishedAt": finished,
            "preview": preview,
            "output": {
                "jsonPath": f"{spec.reports_prefix()}/preview.json",
                "pdfPath": None,
                "markdownPath": None,
                "htmlPath": None,
                "summaryPath": None,
            },
            "artifactPaths": urls,
            "error": None,
        }
    )
    logger.info(
        "order %s exposure_preview delivered snapshot=%s basis=%s",
        spec.order_id,
        preview.get("snapshot", {}).get("notableExposures"),
        preview.get("basis", {}).get("inventoryFindings"),
    )
    return 0


HEALTH_CHECKS_COLLECTION = "health_checks"
_SECRET_RE = re.compile(
    r"(?i)(?:sk-[A-Za-z0-9_-]{8,}|Bearer\s+[A-Za-z0-9._\-]+|AIza[A-Za-z0-9_\-]{20,}|ya29\.[A-Za-z0-9._\-]+)"
)


def is_health_check_request() -> bool:
    return os.environ.get("HEALTH_CHECK", "").strip().lower() in {"1", "true", "yes", "on"}


def sanitize_health_id(raw: str | None) -> str:
    text = str(raw or "").strip()
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "" for ch in text)[:80]
    return cleaned or f"hc_{utc_now().replace(':', '').replace('+', '').replace('-', '')}"


def redact_health_text(text: str) -> str:
    cleaned = _SECRET_RE.sub("[redacted]", text or "")
    return cleaned[:500]


def _check(id: str, name: str, *, ok: bool, level: str, detail: str) -> dict[str, Any]:
    return {
        "id": id,
        "name": name,
        "ok": bool(ok),
        "level": level,
        "detail": redact_health_text(detail),
    }


def env_key_checks(presence: dict[str, bool] | None = None) -> list[dict[str, Any]]:
    presence = presence if presence is not None else _required_llm_env_presence()
    checks: list[dict[str, Any]] = []
    for key, present in presence.items():
        if present:
            checks.append(_check(f"env:{key}", key, ok=True, level="ok", detail="Present on the Cloud Run job."))
        else:
            checks.append(
                _check(
                    f"env:{key}",
                    key,
                    ok=False,
                    level="warn",
                    detail=f"Missing on moyo-report-worker-no-vpc. Set {key} as a Cloud Run job secret/env var.",
                )
            )
    return checks


def overall_health_status(checks: list[dict[str, Any]]) -> str:
    levels = {str(item.get("level") or "") for item in checks}
    if "fail" in levels:
        return "fail"
    if "warn" in levels:
        return "warn"
    return "ok"


def _probe_llm_spec(spec: Any, *, extra: bool = False) -> dict[str, Any]:
    from moyo.llm.client import LLMClient, format_llm_error, llm_spec_has_auth

    label = str(getattr(spec, "label", None) or getattr(spec, "model", None) or "LLM")
    if extra:
        label = f"{label} (additional)"
    check_id = f"llm:{label}"
    if not llm_spec_has_auth(spec):
        env_hint = ""
        api_key = getattr(spec, "api_key", None)
        if not api_key:
            env_hint = " Add the matching API key to the Cloud Run job."
        return _check(
            check_id,
            str(label),
            ok=False,
            level="warn",
            detail=f"No credentials for this retrieval model.{env_hint}",
        )
    try:
        client = LLMClient(spec)
        text = client.complete("Reply with the single word OK.", max_tokens=16, retries=2)
        if (text or "").strip():
            return _check(check_id, str(label), ok=True, level="ok", detail="Accepted a 1-token probe.")
        return _check(
            check_id,
            str(label),
            ok=False,
            level="warn",
            detail="Credentials present but the model returned empty text.",
        )
    except Exception as exc:
        return _check(
            check_id,
            str(label),
            ok=False,
            level="warn",
            detail=f"Probe failed: {format_llm_error(exc)}",
        )


def probe_vertex_adc() -> dict[str, Any]:
    try:
        from moyo.llm.vertex import vertex_access_token, vertex_project

        token = vertex_access_token()
        if token:
            return _check(
                "vertex-adc",
                "Vertex AI ADC",
                ok=True,
                level="ok",
                detail=f"Service account token issued for project {vertex_project()}.",
            )
        return _check(
            "vertex-adc",
            "Vertex AI ADC",
            ok=False,
            level="fail",
            detail=(
                "No Vertex access token. Grant the Cloud Run job service account "
                "roles/aiplatform.user and confirm GOOGLE_CLOUD_PROJECT."
            ),
        )
    except Exception as exc:
        return _check("vertex-adc", "Vertex AI ADC", ok=False, level="fail", detail=str(exc))


def probe_firestore() -> dict[str, Any]:
    if _skip_firebase():
        return _check(
            "firestore",
            "Firestore",
            ok=False,
            level="warn",
            detail="Skipped (MOYO_CLOUD_SKIP_FIREBASE=1).",
        )
    try:
        from firebase_admin import firestore as fs

        _init_firebase_app()
        db = fs.client()
        db.collection(HEALTH_CHECKS_COLLECTION).limit(1).get()
        return _check("firestore", "Firestore", ok=True, level="ok", detail="Worker can read Firestore.")
    except Exception as exc:
        return _check(
            "firestore",
            "Firestore",
            ok=False,
            level="fail",
            detail=f"Worker cannot read Firestore: {exc}",
        )


def probe_gcs() -> dict[str, Any]:
    if _skip_firebase():
        return _check("gcs", "Cloud Storage", ok=False, level="warn", detail="Skipped (MOYO_CLOUD_SKIP_FIREBASE=1).")
    try:
        _db, bucket, _fs = _init_firebase()
        name = _storage_bucket_name()
        if bucket is None:
            return _check(
                "gcs",
                "Cloud Storage",
                ok=False,
                level="fail",
                detail="Storage bucket name is not set (MOYO_REPORTS_STORAGE_BUCKET).",
            )
        exists = bool(bucket.exists())
        if not exists:
            return _check(
                "gcs",
                "Cloud Storage",
                ok=False,
                level="fail",
                detail=f"Bucket gs://{name} does not exist or the job SA cannot see it.",
            )
        blob = bucket.blob(f"health-checks/_probe_{os.getpid()}.txt")
        blob.upload_from_string("ok", content_type="text/plain")
        blob.delete()
        return _check("gcs", "Cloud Storage", ok=True, level="ok", detail=f"Read/write ok on gs://{name}.")
    except Exception as exc:
        return _check(
            "gcs",
            "Cloud Storage",
            ok=False,
            level="fail",
            detail=f"Reports bucket probe failed: {exc}",
        )


def probe_utility_llm() -> dict[str, Any]:
    try:
        from moyo.llm.client import format_llm_error
        from moyo.llm.utility import get_utility_llm, running_in_cloud

        client = get_utility_llm()
        text = client.complete("Reply with the single word OK.", max_tokens=16, retries=2)
        if (text or "").strip():
            where = "Vertex Flash" if running_in_cloud() else client.label
            return _check(
                "utility-llm",
                "Utility LLM (extract / synthesize)",
                ok=True,
                level="ok",
                detail=f"{where} accepted a 1-token probe.",
            )
        return _check(
            "utility-llm",
            "Utility LLM (extract / synthesize)",
            ok=False,
            level="fail",
            detail="Utility model returned empty text.",
        )
    except Exception as exc:
        from moyo.llm.client import format_llm_error

        return _check(
            "utility-llm",
            "Utility LLM (extract / synthesize)",
            ok=False,
            level="fail",
            detail=f"Utility probe failed: {format_llm_error(exc)}",
        )


def probe_retrieval_llms() -> list[dict[str, Any]]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from moyo.llm.registry import get_retrieval_specs, retrieval_model_id

    default_ids = {retrieval_model_id(spec) for spec in get_retrieval_specs(include_optional=False)}
    specs = get_retrieval_specs(include_optional=True)
    if not specs:
        return [
            _check(
                "retrieval",
                "Retrieval LLMs",
                ok=False,
                level="fail",
                detail="No retrieval LLMs configured (config/retrieval_llms.json).",
            )
        ]
    checks: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(12, len(specs))) as pool:
        futures = [
            pool.submit(
                _probe_llm_spec,
                spec,
                extra=retrieval_model_id(spec) not in default_ids,
            )
            for spec in specs
        ]
        for future in as_completed(futures):
            checks.append(future.result())
    checks.sort(key=lambda item: str(item.get("name") or ""))
    working = sum(1 for item in checks if item.get("ok"))
    if working == 0:
        checks.append(
            _check(
                "retrieval",
                "Retrieval fan-out",
                ok=False,
                level="fail",
                detail="No retrieval LLM accepted a probe. Scans cannot query models.",
            )
        )
    return checks


def run_container_health_check() -> dict[str, Any]:
    """Low-cost probes of the Cloud Run worker: keys, Vertex, Firestore, GCS, LLMs."""
    started = utc_now()
    checks: list[dict[str, Any]] = []
    checks.extend(env_key_checks())
    checks.append(probe_vertex_adc())
    checks.append(probe_firestore())
    checks.append(probe_gcs())
    checks.append(probe_utility_llm())
    checks.extend(probe_retrieval_llms())
    status = overall_health_status(checks)
    failed = [item["name"] for item in checks if item.get("level") == "fail"]
    warned = [item["name"] for item in checks if item.get("level") == "warn"]
    if status == "ok":
        summary = "Cloud worker can run scans."
    elif status == "warn":
        summary = "Scans can run, with gaps: " + ", ".join(warned[:8])
    else:
        summary = "Scans will fail: " + ", ".join(failed[:8] or ["see details"])
    return {
        "ok": status != "fail",
        "status": status,
        "summary": summary,
        "checks": checks,
        "startedAt": started,
        "finishedAt": utc_now(),
        "job": os.environ.get("CLOUD_RUN_JOB") or os.environ.get("K_SERVICE") or "moyo-report-worker-no-vpc",
    }


def write_health_check(check_id: str, payload: dict[str, Any]) -> None:
    if _skip_firebase():
        logger.info("health check %s: %s", check_id, payload.get("summary"))
        return
    from firebase_admin import firestore as fs

    _init_firebase_app()
    db = fs.client()
    fields = {
        **payload,
        "status": payload.get("status") or "fail",
        "updatedAt": utc_now(),
    }
    db.collection(HEALTH_CHECKS_COLLECTION).document(check_id).set(fields, merge=True)
    db.collection(HEALTH_CHECKS_COLLECTION).document("latest").set(
        {**fields, "checkId": check_id},
        merge=True,
    )


def run_health_check_main() -> int:
    check_id = sanitize_health_id(os.environ.get("HEALTH_CHECK_ID"))
    logger.info("running container health check %s", check_id)
    if not _skip_firebase():
        try:
            write_health_check(
                check_id,
                {
                    "status": "running",
                    "summary": "Probing API keys and Google Cloud services…",
                    "checks": [],
                    "startedAt": utc_now(),
                },
            )
        except Exception:
            logger.exception("could not mark health check running")
    try:
        payload = run_container_health_check()
        write_health_check(check_id, payload)
        logger.info("health check %s %s (%s)", check_id, payload.get("status"), payload.get("summary"))
        return 0
    except Exception as exc:
        logger.exception("health check %s failed", check_id)
        try:
            write_health_check(
                check_id,
                {
                    "ok": False,
                    "status": "fail",
                    "summary": f"Health check crashed: {type(exc).__name__}: {exc}",
                    "checks": [
                        _check("crash", "Health check", ok=False, level="fail", detail=str(exc)),
                    ],
                    "finishedAt": utc_now(),
                },
            )
        except Exception:
            logger.exception("could not write health check failure")
        return 1


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("MOYO_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ensure_env_loaded()
    if is_health_check_request():
        return run_health_check_main()
    order_id = os.environ.get("ORDER_ID") or ""
    if not order_id and os.environ.get("ORDER_JSON"):
        order_id = "local"
    if not order_id:
        logger.error("ORDER_ID is required")
        return 2

    order_ref = None
    spec = None
    started = utc_now()
    try:
        data, order_ref = _load_order_data(order_id)
        spec = parse_order(order_id, data)
        firestore_title = report_title_for_firestore(
            storage_folder=spec.storage_folder,
            prompts=spec.prompts,
            existing_title=str(data.get("title") or ""),
        )
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

        retry_count = parse_validation_retry_count(data)

        def _set_stage(name: str) -> None:
            _mark(
                {
                    "reportStatus": "generating",
                    "reportStage": name,
                }
            )

        _mark(
            {
                "reportStatus": "generating",
                "reportStage": "querying_models",
                "generationStartedAt": started,
                "generationFinishedAt": None,
                "storageFolder": spec.storage_folder,
                "title": firestore_title,
                "error": None,
            }
        )
        if spec.generation_mode == "exposure_preview":
            return run_exposure_preview(spec, mark=_mark, started=started)

        uploaded: dict[str, Any] = {}
        bucket = None
        if not _skip_firebase():
            _db, bucket, _fs = _init_firebase()
            if bucket is None:
                raise RuntimeError(
                    "Storage bucket name not set. Set MOYO_REPORTS_STORAGE_BUCKET "
                    "or STORAGE_BUCKET on the Cloud Run job "
                    f"(default {DEFAULT_MOYO_REPORTS_BUCKET})."
                )
        rerun_failures: list[str] = []
        incomplete_models: list[str] = []
        if spec.generation_mode == "rerun_models":
            if bucket is None:
                raise RuntimeError("Model rerun needs Storage artifacts.")
            runs, rerun_failures, incomplete_models = run_rerun_models(
                spec, bucket=bucket, work=work, set_stage=_set_stage
            )
            uploaded = _upload_runs(bucket, spec, runs, work=work)
            if rerun_failures:
                raise ModelRerunIncomplete(
                    "Selected model retrieval did not all succeed; the report "
                    "was not rebuilt. Overwritten answers were saved. Remaining "
                    "failures: " + "; ".join(rerun_failures[:8]),
                    incomplete_models=incomplete_models or list(spec.rerun_models or []),
                    failures=rerun_failures,
                )
            _mark({"incompleteRetrievalModels": []})
        elif resolve_rebuild_plan(spec) is not None:
            if bucket is None:
                raise RuntimeError("PDF/picture rebuild needs Storage artifacts.")
            _set_stage("generating_report")
            runs = run_rebuild(spec, bucket=bucket, work=work)
            uploaded = _upload_runs(bucket, spec, runs, work=work)
        else:
            runs = run_moyo(spec, work=work, set_stage=_set_stage)
            if bucket is not None:
                uploaded = _upload_runs(bucket, spec, runs, work=work)
            else:
                write_canonical_report_json(spec, runs, work / "report.json")
        finished = utc_now()
        urls = uploaded.get("urls") or {}
        manifest = uploaded.get("manifest") or artifact_manifest(
            spec.order_id, runs, folder=spec.storage_folder
        )
        validation: ValidationResult | None = None
        if should_auto_validate(spec):
            _set_stage("validating")
            validation = validate_prompt_runs(runs)
            logger.info("order %s %s", spec.order_id, validation.summary())

        action = delivery_action(
            spec, validation=validation, retry_count=retry_count
        )
        if action == "retry":
            if validation is None:
                raise RuntimeError("validation retry without a result")
            next_retry = retry_count + 1
            _mark(
                retry_update_fields(
                    validation=validation,
                    retry_count=next_retry,
                    started=started,
                    finished=finished,
                )
            )
            logger.warning(
                "order %s validation failed; requeue attempt %s (%s)",
                spec.order_id,
                next_retry,
                validation.summary() if validation else "no validation",
            )
            return 0
        if action == "hold":
            if validation is None:
                raise RuntimeError("validation hold without a result")
            _mark(
                hold_update_fields(
                    validation=validation,
                    retry_count=retry_count,
                    started=started,
                    finished=finished,
                )
            )
            logger.error(
                "order %s held after validation retry (%s)",
                spec.order_id,
                validation.summary() if validation else "no validation",
            )
            return 0

        _mark(
            success_update_fields(
                spec,
                started=started,
                finished=finished,
                urls=urls,
                manifest=manifest,
                validation=validation,
                title=firestore_title,
            )
        )
        status = CANONICAL_AWAITING_QC if spec.qc_required else "delivered"
        logger.info(
            "order %s %s (%d report(s) generationMode=%s qcRequired=%s)",
            spec.order_id,
            status,
            len(runs),
            spec.generation_mode,
            spec.qc_required,
        )
        return 0
    except Exception as exc:
        logger.exception("order %s failed", order_id)
        if not _skip_firebase():
            try:
                work = work_dir_for(order_id)
                _db, bucket, _fs = _init_firebase()
                if bucket is not None:
                    folder = (
                        spec.storage_folder
                        if spec is not None
                        else order_storage_folder(order_id, None)
                    )
                    urls = _upload_files(
                        bucket, retrieval_check_storage_paths(folder, work)
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
                message = f"{type(exc).__name__}: {exc}"[:2000]
                if isinstance(exc, ModelRerunIncomplete):
                    order_ref.update(
                        {
                            "reportStatus": (
                                CANONICAL_AWAITING_QC if spec is None or spec.qc_required else "delivered"
                            ),
                            "qcStatus": "pending",
                            "generationStartedAt": started,
                            "generationFinishedAt": utc_now(),
                            "error": message,
                            "incompleteRetrievalModels": list(exc.incomplete_models or []),
                        }
                    )
                elif (
                    spec is not None
                    and spec.generation_mode in REBUILD_MODES
                    and is_raw_product(spec)
                ):
                    # Packaging failed; keep the Exposure Data scan delivered.
                    order_ref.update(
                        {
                            "reportStatus": "delivered",
                            "packagedStatus": "failed",
                            "packagedError": message,
                            "generationStartedAt": started,
                            "generationFinishedAt": utc_now(),
                        }
                    )
                else:
                    order_ref.update(
                        {
                            "reportStatus": "failed",
                            "generationStartedAt": started,
                            "generationFinishedAt": utc_now(),
                            "error": message,
                        }
                    )
            except Exception:
                logger.exception("failed to write error status")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
