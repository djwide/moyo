"""[3a] Collapse similar claims via local Ollama; merge sources / scores."""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

SCORE_KEYS = (
    "sensitivity",
    "specificity",
    "novelty",
    "interestingness",
)

# Exact quantities: integers with commas, decimals, $, %, ratios, years.
_EXACT_NUMBER_RE = re.compile(
    r"(?:"
    r"[\$€£]\s?\d[\d,]*(?:\.\d+)?"
    r"|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b"
    r"|\b\d+\.\d+\b"
    r"|\b\d+\s?%"
    r"|\b\d{4}\b"
    r"|\b\d+\s*(?:mg|kg|g|ml|oz|lb)\b"
    r"|\b\d+\s*/\s*\d+\b"
    r")"
)

DEFAULT_CLUSTER_PROMPT = Path(__file__).resolve().parents[1] / "prompts" / "cluster_claims.md"


def group_key(finding: dict[str, Any]) -> str:
    """Stable exposure-group key: cluster_id, else claim_id."""
    return str(finding.get("cluster_id") or finding.get("claim_id") or "").strip()


def _group_prefer_rank(finding: dict[str, Any]) -> tuple:
    """Prefer richer collapsed survivors when deduping a group."""
    try:
        merged = int(finding.get("merged_count") or 1)
    except (TypeError, ValueError):
        merged = 1
    try:
        corr = int(finding.get("corroboration") or 1)
    except (TypeError, ValueError):
        corr = 1
    return (
        merged,
        corr,
        _int_score(finding, "sensitivity") * _int_score(finding, "specificity"),
        _int_score(finding, "interestingness"),
        len(str(finding.get("claim") or "")),
    )


def dedupe_findings_by_group(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep **one** finding per exposure group for inventories / PDFs.

    Groups are keyed by ``cluster_id`` (collapsed cluster). Also drops any claim
    whose ``claim_id`` only appears inside another finding's ``merged_from`` list
    (raw members that should not be listed beside their survivor).
    """
    if not findings:
        return []

    survivors: set[str] = set()
    absorbed: set[str] = set()
    for f in findings:
        cid = str(f.get("claim_id") or "")
        if cid:
            survivors.add(cid)
        for mid in f.get("merged_from") or []:
            mid_s = str(mid or "")
            if mid_s and mid_s != cid:
                absorbed.add(mid_s)

    # Members that only live inside merged_from of a survivor must not appear
    # as their own inventory/finding rows.
    hide = absorbed - survivors

    best: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for f in findings:
        cid = str(f.get("claim_id") or "")
        if cid and cid in hide:
            continue
        key = group_key(f) or cid or str(id(f))
        if key not in best:
            order.append(key)
            best[key] = f
            continue
        if _group_prefer_rank(f) > _group_prefer_rank(best[key]):
            best[key] = f
    return [best[k] for k in order]


def has_exact_number(text: str) -> bool:
    """True when the claim text contains a concrete numeric quantity."""
    return bool(_EXACT_NUMBER_RE.search(text or ""))


def specificity_with_numbers(base: int, text: str) -> int:
    """Inherit max specificity, then +1 when exact numbers appear (cap 5)."""
    try:
        n = int(base)
    except (TypeError, ValueError):
        n = 1
    if has_exact_number(text):
        n += 1
    return max(1, min(5, n))


def confidence_from_models(n_models: int) -> int:
    """Confidence is determined by the number of corroborating models (1–5)."""
    try:
        n = int(n_models)
    except (TypeError, ValueError):
        n = 1
    return max(1, min(5, n))


def confidence_boost(*, n_models: int, n_citations: int = 0) -> int:
    """Deprecated alias; prefer :func:`confidence_from_models`."""
    del n_citations
    return max(0, confidence_from_models(n_models) - 1)


def _normalize_citation(cite: Any) -> str:
    if isinstance(cite, dict):
        raw = str(cite.get("url") or cite.get("label") or cite.get("ref") or "").strip()
    else:
        raw = str(cite or "").strip()
    return raw.lower().rstrip("/")


def _citation_display(cite: Any) -> str:
    if isinstance(cite, dict):
        return str(cite.get("url") or cite.get("label") or cite.get("ref") or "").strip()
    return str(cite or "").strip()


def _merge_citations(members: list[dict]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for c in members:
        for cite in c.get("citations") or []:
            key = _normalize_citation(cite)
            if not key or key in seen:
                continue
            seen.add(key)
            display = _citation_display(cite)
            if display:
                out.append(display)
    return out


def _int_score(claim: dict, key: str, default: int = 1) -> int:
    try:
        return int(claim.get(key) if claim.get(key) is not None else default)
    except (TypeError, ValueError):
        return default


def _member_score_snapshot(claim: dict) -> dict[str, Any]:
    return {
        "claim_id": claim.get("claim_id"),
        "source_model": claim.get("source_model"),
        "claim": claim.get("claim"),
        "sensitivity": _int_score(claim, "sensitivity"),
        "specificity": _int_score(claim, "specificity"),
        "novelty": _int_score(claim, "novelty"),
        "interestingness": _int_score(claim, "interestingness"),
        "confidence": _int_score(claim, "confidence", default=1),
        "status": claim.get("status"),
        "citations": list(claim.get("citations") or []),
    }


def _rep_rank(claim: dict) -> tuple:
    text = str(claim.get("claim") or "")
    return (
        1 if has_exact_number(text) else 0,
        _int_score(claim, "interestingness") * _int_score(claim, "specificity"),
        _int_score(claim, "sensitivity"),
        len(text),
    )


def _claim_models(claim: dict) -> list[str]:
    models = claim.get("source_models")
    if isinstance(models, list) and models:
        return [str(m) for m in models if str(m).strip()]
    sm = str(claim.get("source_model") or "").strip()
    return [sm] if sm else []


def _singleton_groups(claim_ids: list[str]) -> list[list[str]]:
    return [[cid] for cid in claim_ids]


def _groups_by_normalized_text(claims: list[dict]) -> list[list[str]]:
    """Dry-run / test helper: exact normalized-text equality (not Jaccard)."""
    buckets: dict[str, list[str]] = defaultdict(list)
    order: list[str] = []
    for c in claims:
        cid = str(c.get("claim_id") or "")
        if not cid:
            continue
        key = " ".join(re.findall(r"[a-z0-9]+", str(c.get("claim") or "").lower()))
        if key not in buckets:
            order.append(key)
        buckets[key].append(cid)
    return [buckets[k] for k in order]


def _parse_groups_payload(text: str, valid_ids: set[str]) -> list[list[str]] | None:
    """Parse ``{"groups": [["C1","C2"], ...]}``; return None if unusable."""
    if not text or not str(text).strip():
        return None
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    groups_raw = data.get("groups") if isinstance(data, dict) else None
    if not isinstance(groups_raw, list):
        return None

    groups: list[list[str]] = []
    seen: set[str] = set()
    for g in groups_raw:
        if not isinstance(g, list):
            continue
        ids = []
        for item in g:
            cid = str(item).strip()
            if cid in valid_ids and cid not in seen:
                seen.add(cid)
                ids.append(cid)
        if ids:
            groups.append(ids)

    # Attach any missing ids as singletons so every claim is covered
    for cid in sorted(valid_ids):
        if cid not in seen:
            groups.append([cid])

    if not groups:
        return None
    return groups


def _load_group_prompt(prompt_path: Path, claims_payload: list[dict]) -> str:
    tmpl = prompt_path.read_text(encoding="utf-8")
    blob = json.dumps(claims_payload, ensure_ascii=False, indent=2)
    return tmpl.replace("{{ claims_json }}", blob)


def _ollama_group_batch(
    batch: list[dict],
    *,
    client: Any,
    prompt_path: Path,
) -> list[list[str]]:
    """Ask Ollama to group one batch of claims; fall back to singletons."""
    valid = {str(c["claim_id"]) for c in batch if c.get("claim_id")}
    if not valid:
        return []
    payload = [
        {
            "claim_id": c.get("claim_id"),
            "claim": c.get("claim"),
            "source_model": c.get("source_model"),
            "query_id": c.get("query_id"),
        }
        for c in batch
    ]
    prompt = _load_group_prompt(prompt_path, payload)
    try:
        text = client.complete(prompt)
    except Exception as exc:
        print(f"  warn: ollama cluster batch failed: {exc}", file=sys.stderr)
        return _singleton_groups(sorted(valid))
    groups = _parse_groups_payload(text or "", valid)
    if not groups:
        print(
            "  warn: ollama cluster returned unparseable groups; "
            "keeping batch claims unmerged",
            file=sys.stderr,
        )
        return _singleton_groups(sorted(valid))
    return groups


def group_claim_ids_with_ollama(
    claims: list[dict],
    *,
    client: Any,
    prompt_path: Path | None = None,
    batch_size: int = 35,
) -> list[list[str]]:
    """Partition claims into same-fact groups using a local Ollama LLM.

    Large claim lists are processed in batches, then batch-group representatives
    are merged in a second Ollama pass so paraphrases across batches can join.
    """
    if not claims:
        return []
    path = prompt_path or DEFAULT_CLUSTER_PROMPT
    if not path.exists():
        raise FileNotFoundError(f"Cluster prompt not found: {path}")

    by_id = {str(c["claim_id"]): c for c in claims if c.get("claim_id")}
    all_ids = list(by_id.keys())
    size = max(5, int(batch_size))

    # Pass 1: group within batches
    batch_groups: list[list[str]] = []
    for i in range(0, len(all_ids), size):
        chunk_ids = all_ids[i : i + size]
        batch = [by_id[cid] for cid in chunk_ids]
        print(
            f"  ollama cluster batch {i // size + 1}/"
            f"{(len(all_ids) + size - 1) // size} ({len(batch)} claims)",
            file=sys.stderr,
        )
        batch_groups.extend(_ollama_group_batch(batch, client=client, prompt_path=path))

    if len(batch_groups) <= 1:
        return batch_groups

    # Pass 2: merge across batches via representatives
    reps: list[dict] = []
    for gi, members in enumerate(batch_groups):
        member_claims = [by_id[cid] for cid in members if cid in by_id]
        if not member_claims:
            continue
        rep = max(member_claims, key=_rep_rank)
        reps.append(
            {
                "claim_id": f"G{gi:04d}",
                "claim": rep.get("claim"),
                "source_model": "group",
                "query_id": rep.get("query_id"),
                "_member_ids": members,
            }
        )

    if len(reps) <= 1:
        return batch_groups

    meta_groups: list[list[str]] = []
    for i in range(0, len(reps), size):
        chunk = reps[i : i + size]
        print(
            f"  ollama cluster merge-pass {i // size + 1}/"
            f"{(len(reps) + size - 1) // size} ({len(chunk)} groups)",
            file=sys.stderr,
        )
        # Strip internal fields from the prompt payload
        prompt_batch = [
            {
                "claim_id": r["claim_id"],
                "claim": r["claim"],
                "source_model": r.get("source_model"),
                "query_id": r.get("query_id"),
            }
            for r in chunk
        ]
        # Temporary map for this chunk call
        valid = {r["claim_id"] for r in chunk}
        prompt = _load_group_prompt(path, prompt_batch)
        try:
            text = client.complete(prompt)
            parsed = _parse_groups_payload(text or "", valid)
        except Exception as exc:
            print(f"  warn: ollama merge-pass failed: {exc}", file=sys.stderr)
            parsed = None
        if not parsed:
            parsed = [[r["claim_id"]] for r in chunk]
        meta_groups.extend(parsed)

    rep_by_gid = {r["claim_id"]: r for r in reps}
    # Union-find over meta group ids that share a meta_groups cell
    parents = {r["claim_id"]: r["claim_id"] for r in reps}

    def find(x: str) -> str:
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parents[rb] = ra

    for g in meta_groups:
        ids = [gid for gid in g if gid in parents]
        for a, b in zip(ids, ids[1:]):
            union(a, b)

    merged: dict[str, list[str]] = defaultdict(list)
    for r in reps:
        root = find(r["claim_id"])
        merged[root].extend(r["_member_ids"])

    # De-dupe member ids while preserving order
    out: list[list[str]] = []
    for root in sorted(merged.keys()):
        seen: set[str] = set()
        ids: list[str] = []
        for cid in merged[root]:
            if cid not in seen and cid in by_id:
                seen.add(cid)
                ids.append(cid)
        if ids:
            out.append(ids)

    # Safety: any claim missing from merges → singleton
    covered = {cid for g in out for cid in g}
    for cid in all_ids:
        if cid not in covered:
            out.append([cid])
    return out


def merge_claim_group(
    members: list[dict],
    *,
    cluster_id: str,
    corroboration_min_sources: int = 2,
) -> dict:
    """Collapse one similarity group into a single claim.

    Individual member scores are kept on ``member_scores``. Union
    ``sensitivity`` is the highest member sensitivity.
    """
    if not members:
        raise ValueError("empty claim group")

    member_scores = [_member_score_snapshot(c) for c in members]
    models = sorted({m for c in members for m in _claim_models(c)})
    n_models = max(1, len(models))
    cites = _merge_citations(members)

    if len(members) == 1:
        c = dict(members[0])
        text = str(c.get("claim") or "")
        c["source_models"] = models
        c["source_model"] = models[0] if models else c.get("source_model") or "unknown"
        c["citations"] = cites
        c["source_count"] = len(cites)
        c["corroboration"] = n_models
        c["confidence"] = confidence_from_models(n_models)
        c["specificity"] = specificity_with_numbers(
            _int_score(c, "specificity"), text
        )
        c["member_scores"] = member_scores
        c["cluster_id"] = cluster_id
        c["merged_from"] = [c.get("claim_id")]
        c["merged_count"] = 1
        prior = str(c.get("status") or "UNVERIFIED")
        if prior != "CONTESTED":
            if n_models >= corroboration_min_sources:
                c["status"] = "CORROBORATED"
            elif prior == "UNVERIFIED" and _int_score(c, "sensitivity") >= 4:
                c["status"] = "MODEL-SPECIFIC"
            elif n_models == 1 and prior not in {"OUTLIER", "MODEL-SPECIFIC"}:
                c["status"] = "MODEL-SPECIFIC"
        return c

    rep = max(members, key=_rep_rank)
    merged = dict(rep)
    text = str(merged.get("claim") or "")

    merged["member_scores"] = member_scores
    merged["sensitivity"] = max(_int_score(c, "sensitivity") for c in members)
    for key in ("specificity", "novelty", "interestingness"):
        merged[key] = max(_int_score(c, key) for c in members)

    merged["specificity"] = specificity_with_numbers(merged["specificity"], text)
    merged["source_models"] = models
    merged["source_model"] = models[0] if models else merged.get("source_model") or "unknown"
    if len(models) > 1:
        merged["source_cite"] = f"{models[0]} +{len(models) - 1}"
    else:
        merged["source_cite"] = models[0] if models else merged.get("source_model")

    merged["citations"] = cites
    merged["source_count"] = len(cites)
    merged["corroboration"] = n_models
    merged["confidence"] = confidence_from_models(n_models)
    merged["cluster_id"] = cluster_id
    merged["merged_from"] = [c.get("claim_id") for c in members if c.get("claim_id")]
    merged["merged_count"] = len(members)

    if any(str(c.get("status") or "").upper() == "CONTESTED" for c in members):
        merged["status"] = "CONTESTED"
    elif n_models >= corroboration_min_sources:
        merged["status"] = "CORROBORATED"
    elif any(str(c.get("status") or "").upper() == "OUTLIER" for c in members):
        merged["status"] = "OUTLIER"
    elif n_models == 1:
        merged["status"] = "MODEL-SPECIFIC"
    else:
        merged["status"] = "UNVERIFIED"

    excerpts = [
        (c.get("raw_excerpt") or "").strip()
        for c in members
        if (c.get("raw_excerpt") or "").strip()
    ]
    if excerpts:
        merged["raw_excerpt"] = max(excerpts, key=len)

    query_ids = sorted({str(c.get("query_id")) for c in members if c.get("query_id")})
    if query_ids:
        merged["query_id"] = query_ids[0]
        if len(query_ids) > 1:
            merged["query_ids"] = query_ids

    chunk_ids = [str(c.get("chunk_id")) for c in members if c.get("chunk_id")]
    if chunk_ids:
        merged["chunk_ids"] = sorted(set(chunk_ids))

    return merged


def _make_ollama_client(llm_config: dict | None) -> Any:
    from moyo.llm.client import LLMClient, LLMSpec
    from moyo.llm.testing import FakeDeterministicLLM, is_test_mode

    cfg = llm_config or {}
    if is_test_mode() or str(cfg.get("provider") or "").lower() in {"test", "echo"}:
        return FakeDeterministicLLM(model_name=str(cfg.get("model") or "echo-test"))

    spec = LLMSpec.from_dict(
        {
            "provider": cfg.get("provider", "ollama"),
            "model": cfg.get("model", "llama3.1:8b"),
            "base_url": cfg.get("base_url", "http://localhost:11434"),
            "api_key": cfg.get("api_key"),
            "temperature": float(cfg.get("temperature", 0.1)),
            "max_tokens": int(cfg.get("max_tokens", 4000)),
            "num_ctx": cfg.get("num_ctx", 16000),
            "timeout": int(cfg.get("timeout", 180)),
        }
    )
    client = LLMClient(spec)
    if not client.is_available():
        raise RuntimeError(
            f"Cluster LLM unavailable ({spec.provider}/{spec.model} @ "
            f"{spec.base_url or 'default'}). For local runs start Ollama; "
            "on Cloud Run set MOONSHOT_API_KEY, or pass --dry-run."
        )
    return client


def cluster_claims(
    claims: list[dict],
    *,
    corroboration_min_sources: int = 2,
    collapse: bool = True,
    llm_config: dict | None = None,
    prompt_path: Path | str | None = None,
    dry_run: bool = False,
    group_fn: Callable[[list[dict]], list[list[str]]] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Group claims that state the same atomic fact (via local Ollama), then
    optionally **collapse** each group into one claim.

    When ``collapse`` is True (default):
      - One surviving claim per group (best representative text)
      - ``citations`` / ``source_models`` are unions of the members
      - ``member_scores`` keeps each member's individual score dims
      - Union ``sensitivity`` = max member sensitivity
      - Other score dims take the max across members; specificity +1 for exact numbers
      - ``confidence`` = number of distinct corroborating models (clamped 1–5)

    ``dry_run`` / missing Ollama: groups by exact normalized claim text (no Jaccard).
    """
    if not claims:
        return [], []

    path = Path(prompt_path) if prompt_path else DEFAULT_CLUSTER_PROMPT
    cfg = dict(llm_config or {})
    batch_size = int(cfg.get("batch_size", 35))

    if group_fn is not None:
        id_groups = group_fn(claims)
    elif dry_run:
        print("  cluster dry-run: grouping by exact normalized text", file=sys.stderr)
        id_groups = _groups_by_normalized_text(claims)
    else:
        try:
            from moyo.llm.testing import is_test_mode

            if is_test_mode():
                print(
                    "  cluster test-mode: grouping by exact normalized text",
                    file=sys.stderr,
                )
                id_groups = _groups_by_normalized_text(claims)
            else:
                client = _make_ollama_client(cfg)
                print(
                    f"  cluster via {cfg.get('provider', 'ollama')}/"
                    f"{cfg.get('model', 'llama3.1:8b')}",
                    file=sys.stderr,
                )
                id_groups = group_claim_ids_with_ollama(
                    claims,
                    client=client,
                    prompt_path=path,
                    batch_size=batch_size,
                )
        except Exception as exc:
            # Last resort: do not use Jaccard; leave unmerged so the run continues.
            print(
                f"  warn: ollama clustering failed ({exc}); "
                "leaving claims unmerged",
                file=sys.stderr,
            )
            id_groups = _singleton_groups(
                [str(c["claim_id"]) for c in claims if c.get("claim_id")]
            )

    by_id = {str(c["claim_id"]): c for c in claims if c.get("claim_id")}
    clusters: list[dict] = []
    collapsed: list[dict] = []
    annotated = list(claims)

    for ci, member_ids in enumerate(id_groups, start=1):
        cluster_id = f"CL{ci:03d}"
        members = [by_id[cid] for cid in member_ids if cid in by_id]
        if not members:
            continue
        models = sorted({m for c in members for m in _claim_models(c)})
        citation_keys = sorted(
            {
                _normalize_citation(cite)
                for c in members
                for cite in (c.get("citations") or [])
                if _normalize_citation(cite)
            }
        )
        rep = max(members, key=_rep_rank)

        if collapse:
            merged = merge_claim_group(
                members,
                cluster_id=cluster_id,
                corroboration_min_sources=corroboration_min_sources,
            )
            collapsed.append(merged)
            claim_ids = [merged["claim_id"]]
            member_id_list = list(merged.get("merged_from") or claim_ids)
            representative_id = merged["claim_id"]
        else:
            n_models = len(models)
            n_citations = len(citation_keys)
            for c in members:
                row = dict(c)
                row["cluster_id"] = cluster_id
                row["corroboration"] = n_models
                row["source_count"] = n_citations
                row["source_models"] = models
                row["member_scores"] = [_member_score_snapshot(row)]
                row["confidence"] = confidence_from_models(n_models)
                row["specificity"] = specificity_with_numbers(
                    _int_score(row, "specificity"),
                    str(row.get("claim") or ""),
                )
                prior = row.get("status", "UNVERIFIED")
                if prior != "CONTESTED":
                    if n_models >= corroboration_min_sources:
                        row["status"] = "CORROBORATED"
                    elif n_models == 1 and prior == "UNVERIFIED" and row.get(
                        "sensitivity", 0
                    ) >= 4:
                        row["status"] = "MODEL-SPECIFIC"
                    elif n_models == 1:
                        row["status"] = "MODEL-SPECIFIC"
                # replace in annotated list
                for i, existing in enumerate(annotated):
                    if existing.get("claim_id") == row.get("claim_id"):
                        annotated[i] = row
                        break
            claim_ids = [c["claim_id"] for c in members]
            member_id_list = claim_ids
            representative_id = rep["claim_id"]

        clusters.append(
            {
                "cluster_id": cluster_id,
                "claim_ids": claim_ids,
                "member_ids": member_id_list,
                "models": models,
                "citations": citation_keys,
                "representative_id": representative_id,
                "size": len(members),
            }
        )

    out = collapsed if collapse else annotated
    return out, clusters
