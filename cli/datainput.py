"""Moyo datainput CLI thin wrapper over shared_utils.ingest pipeline."""

import click
from typing import List, Optional

from shared_utils.storage import get_storage
from shared_utils.ingest.pipeline import ingest_paths, IngestConfig


def _parse_csv_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(',') if v.strip()]


@click.group()
def datainput():
    """Data ingestion commands for Moyo."""
    pass


@datainput.command("add")
@click.argument("path", nargs=-1, required=True)
@click.option("--storage-backend", type=click.Choice(["local", "s3"]), default="local")
@click.option("--local-root", default="./data")
@click.option("--s3-bucket")
@click.option("--s3-prefix", default="")
@click.option("--s3-endpoint")
@click.option("--s3-region")
@click.option("--s3-addressing", default="auto")
@click.option("--policy", multiple=True, help="Policy tags (repeat)")
@click.option("--allowed-mime", default="text/*,application/pdf,application/json")
@click.option("--blocked-ext", default=".exe,.dll,.bin")
@click.option("--max-bytes", default="50MB")
@click.option("--max-pages-pdf", default=500, type=int)
@click.option("--max-expand-bytes-archive", default="200MB")
@click.option("--ocr", type=click.Choice(["off", "on"]), default="off")
@click.option("--chunk", default="strategy=sentences")
@click.option("--overlap", default=1, type=int)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--verbose", is_flag=True, default=False)
def add_cmd(path, storage_backend, local_root, s3_bucket, s3_prefix, s3_endpoint, s3_region, s3_addressing,
            policy, allowed_mime, blocked_ext, max_bytes, max_pages_pdf, max_expand_bytes_archive,
            ocr, chunk, overlap, dry_run, verbose):
    """Ingest files into data/private/ with deterministic, idempotent pipeline."""
    # storage config
    store_cfg = {
        "backend": storage_backend,
        "local_root": local_root,
        "s3_bucket": s3_bucket,
        "s3_prefix": s3_prefix,
        "s3_region": s3_region,
        "s3_endpoint_url": s3_endpoint,
        "s3_addressing": s3_addressing,
    }
    store = get_storage(store_cfg)

    # parse limits
    def _parse_size(s: str) -> int:
        s = s.strip().upper().replace("B", "")
        if s.endswith("KB"):
            return int(float(s[:-2]) * 1024)
        if s.endswith("MB"):
            return int(float(s[:-2]) * 1024 * 1024)
        if s.endswith("GB"):
            return int(float(s[:-2]) * 1024 * 1024 * 1024)
        return int(s)

    max_bytes_int = _parse_size(max_bytes)
    max_expand_int = _parse_size(max_expand_bytes_archive)

    # chunk strategy
    strategy = "sentences"
    fixed_size = 1000
    if "strategy=" in chunk:
        strategy = chunk.split("=", 1)[1]

    cfg = IngestConfig(
        max_bytes_per_file=max_bytes_int,
        max_pages_pdf=max_pages_pdf,
        max_expand_bytes_archive=max_expand_int,
        ocr_enabled=(ocr == "on"),
        chunk_strategy=strategy,
        chunk_overlap=overlap,
        chunk_fixed_size=fixed_size,
        allowed_mime=_parse_csv_list(allowed_mime),
        blocked_ext=_parse_csv_list(blocked_ext),
    )

    if dry_run:
        click.echo(f"[DRY RUN] Would ingest {len(path)} files with {cfg} into backend {storage_backend}")
        return

    recs = ingest_paths(path, policy_tags=list(policy), cfg=cfg, store=store, base_dir="data/private")
    if verbose:
        for r in recs:
            click.echo(f"Wrote manifest for {r.src_path} -> {r.stored_at} ({len(r.chunks)} chunks)")
    click.echo(f"Ingestion complete. {len(recs)} new records written.")


if __name__ == "__main__":
    datainput()


