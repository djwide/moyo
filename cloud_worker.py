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
    qcRequired           false skips human QC (agent orders → delivered).
                         GUI and Checkout default true when the field is missing.
    qcStatus             pending | not_required
    generationMode       full | pdf_from_markdown | rebuild_graphics | from_stage
                         (or a pipeline stage name: parse…render)
    fromStage            parse | extract | cluster | score | synthesize |
                         graphics | render  (rebuilds; same as local --from-stage)
    keepGraphics         reuse assets/*.svg (local --keep-graphics)
    keepContent          reuse report.yaml / report.md (local --keep-content)
    generationStartedAt  ISO-8601 UTC, set when work begins
    generationFinishedAt ISO-8601 UTC, set on success or failure
    output.pdfPath       reports/{orderId}/report.pdf
    output.jsonPath      reports/{orderId}/report.json
    output.markdownPath  reports/{orderId}/report.md
    output.htmlPath      reports/{orderId}/report.html

``awaiting_qc`` is the canonical human-QC state. ``qc_pending`` is accepted
only as a legacy alias when reading status.

One report per prompt. A single-prompt order writes artifacts at
``reports/{order_id}/`` (the path QC already uses). Multi-prompt orders
use ``reports/{order_id}/{nn}_{slug}/`` plus a canonical root ``report.json``
and ``manifest.json``.

Ollama is not used in Cloud Run. Rewording, translation, clustering,
summaries, extract, synthesize, and Englishize use Vertex Gemini Flash
(``google/gemini-2.5-flash`` via the job service account). Retrieval
fan-out still uses each provider's own key (Gemini retrieval stays on
Vertex Pro).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from moyo.llm.client import ensure_env_loaded

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
    "report.json",
    "raw_responses.json",
    "evidence.json",
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
REBUILD_INPUT_FILES = (
    "report.md",
    "report.yaml",
    "report_data.json",
    "exploration.md",
    "claims.jsonl",
    "chunks.jsonl",
)

REBUILD_MODES = frozenset({"pdf_from_markdown", "rebuild_graphics", "from_stage"})
GENERATION_MODE_ALIASES = {
    "full": "full",
    "explore": "full",
    "pdf_from_markdown": "pdf_from_markdown",
    "pdf": "pdf_from_markdown",
    "rebuild_graphics": "rebuild_graphics",
    "graphics_only": "rebuild_graphics",
    "from_stage": "from_stage",
    "fromstage": "from_stage",
}

# Dedicated worker/QC bucket. Not the Firebase Auth app bucket.
DEFAULT_MOYO_REPORTS_BUCKET = "senteguard-website-moyo-reports"
CANONICAL_AWAITING_QC = "awaiting_qc"
LEGACY_AWAITING_QC = "qc_pending"
AWAITING_QC_STATUSES = frozenset({CANONICAL_AWAITING_QC, LEGACY_AWAITING_QC})
HUMAN_QC_SOURCES = frozenset({"stripe_checkout", "admin", "gui"})
PRODUCT_IDS = {
    "snapshot": "moyo_snapshot",
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
    qc_required: bool = True
    product_id: str = "moyo_snapshot"
    source: str | None = None


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
    if spec.generation_mode == "full":
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

    product = normalize_product(_first(data, "product", default="snapshot"))
    source = _first(data, "source", default=None)
    if source is not None:
        source = str(source).strip() or None

    return OrderSpec(
        order_id=order_id,
        prompts=normalize_prompts(
            _first(data, "prompts", "customerPrompts", "customer_prompts", "prompt")
        ),
        product=product,
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
        qc_required=normalize_qc_required(
            _first(data, "qcRequired", "qcRequire", "qc_required", default=None),
            source,
        ),
        product_id=normalize_product_id(
            _first(data, "productId", "product_id", default=None), product
        ),
        source=source,
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


def output_paths(order_id: str, urls: dict[str, str]) -> dict[str, str | None]:
    """Storefront output.pdfPath / jsonPath / markdownPath / htmlPath."""
    prefix = f"reports/{order_id}/"

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
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "generationStartedAt": started,
        "generationFinishedAt": finished,
        "generationMode": spec.generation_mode,
        "artifactPaths": urls,
        "reportManifest": manifest,
        "output": output_paths(spec.order_id, urls),
        "qcRequired": spec.qc_required,
        "error": None,
    }
    if spec.qc_required:
        fields["reportStatus"] = CANONICAL_AWAITING_QC
        fields["qcStatus"] = "pending"
    else:
        fields["reportStatus"] = "delivered"
        fields["qcStatus"] = "not_required"
        fields["deliveredAt"] = finished
    return fields


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


def note_explore_gaps(prompt_dir: Path, prompt: str) -> list[str]:
    """Log failed/empty retrievals; never abort — the report uses what succeeded."""
    ok, total, errors = _count_usable_raw_responses(prompt_dir / "raw_responses.json")
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

    raw = work / "raw_responses.json"
    if raw.exists():
        found["raw_responses.json"] = raw
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
        if single:
            dest.append((f"reports/{order_id}/{path.name}", path))
        else:
            dest.append((f"reports/{order_id}/{slug}/{path.name}", path))
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
    pipeline_notes = note_explore_gaps(prompt_dir, prompt)

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
    missing = [name for name in CONTRACT_ARTIFACTS if name not in run.artifacts]
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
    "raw_responses.json",
    "evidence.json",
)


def download_order_prefix(bucket, order_id: str, dest: Path) -> Path:
    """Copy gs://…/reports/{orderId}/** into dest."""
    prefix = f"reports/{order_id}/"
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
        if name in {"raw_responses.json", "evidence.json", "exploration.md"}:
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

    gcs_root = download_order_prefix(bucket, spec.order_id, work / "gcs")
    _progress(
        f"rebuild from-stage={plan.from_stage} keep_graphics={plan.keep_graphics} "
        f"keep_content={plan.keep_content} gs://{bucket.name}/reports/{spec.order_id}/"
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
        missing_out = [name for name in REBUILD_ARTIFACTS if name not in run.artifacts]
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
    try:
        from moyo.llm.registry import get_retrieval_specs
        from moyo.llm.vertex import is_vertex_openai_url

        for spec_llm in get_retrieval_specs():
            dest = spec_llm.base_url or spec_llm.provider
            via = "vertex" if is_vertex_openai_url(spec_llm.base_url) else spec_llm.provider
            _progress(f"retrieval LLM {spec_llm.label}: {spec_llm.model} via {via} ({dest})")
    except Exception as exc:
        logger.warning("Could not list retrieval LLMs: %s", exc)

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
    """(object path, local file) pairs. One copy per file.

    Single-prompt orders land at ``reports/{order_id}/`` (QC path).
    Multi-prompt orders land at ``reports/{order_id}/{slug}/``.
    """
    dest: list[tuple[str, Path]] = []
    single = len(runs) == 1
    for run in runs:
        prefix = (
            f"reports/{order_id}"
            if single
            else f"reports/{order_id}/{run.slug}"
        )
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


def _normalize_storage_bucket_name(value: str) -> str:
    text = value.strip()
    if text.lower().startswith("gs://"):
        text = text[5:]
    return text.strip().strip("/")


def _storage_bucket_name() -> str | None:
    """Dedicated reports bucket, never the Firebase Auth app bucket."""
    explicit = (
        os.environ.get("MOYO_REPORTS_STORAGE_BUCKET")
        or os.environ.get("STORAGE_BUCKET")
        or ""
    ).strip()
    if explicit:
        return _normalize_storage_bucket_name(explicit)
    return DEFAULT_MOYO_REPORTS_BUCKET


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
    dest = dict(storage_destinations(spec.order_id, runs))
    dest[f"reports/{spec.order_id}/report.json"] = canonical
    urls = _upload_files(bucket, list(dest.items()))
    manifest = artifact_manifest(spec.order_id, runs)
    manifest_blob = bucket.blob(f"reports/{spec.order_id}/manifest.json")
    manifest_blob.upload_from_string(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        content_type="application/json",
    )
    urls[f"reports/{spec.order_id}/manifest.json"] = (
        f"gs://{bucket.name}/reports/{spec.order_id}/manifest.json"
    )
    return {"manifest": manifest, "urls": urls}


def _content_type_for(path: Path) -> str | None:
    ext = path.suffix.lower()
    return {
        ".pdf": "application/pdf",
        ".json": "application/json; charset=utf-8",
        ".md": "text/markdown; charset=utf-8",
        ".html": "text/html; charset=utf-8",
        ".svg": "image/svg+xml",
        ".txt": "text/plain; charset=utf-8",
    }.get(ext)


def _upload_files(bucket, pairs: list[tuple[str, Path]]) -> dict[str, str]:
    urls: dict[str, str] = {}
    for object_path, path in pairs:
        if not path.exists():
            continue
        blob = bucket.blob(object_path)
        content_type = _content_type_for(path)
        if content_type:
            blob.upload_from_filename(str(path), content_type=content_type)
        else:
            blob.upload_from_filename(str(path))
        urls[object_path] = f"gs://{bucket.name}/{object_path}"
        logger.info("uploaded %s", urls[object_path])
    return urls


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("MOYO_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ensure_env_loaded()
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
                "error": None,
            }
        )

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
        if resolve_rebuild_plan(spec) is not None:
            if bucket is None:
                raise RuntimeError("PDF/picture rebuild needs Storage artifacts.")
            runs = run_rebuild(spec, bucket=bucket, work=work)
            uploaded = _upload_runs(bucket, spec, runs, work=work)
        else:
            runs = run_moyo(spec, work=work)
            if bucket is not None:
                uploaded = _upload_runs(bucket, spec, runs, work=work)
            else:
                write_canonical_report_json(spec, runs, work / "report.json")
        finished = utc_now()
        urls = uploaded.get("urls") or {}
        manifest = uploaded.get("manifest") or artifact_manifest(spec.order_id, runs)
        _mark(
            success_update_fields(
                spec,
                started=started,
                finished=finished,
                urls=urls,
                manifest=manifest,
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
