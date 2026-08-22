"""Build a local sensitive-phrases corpus from documents or a phrase list."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click

from moyo.privateside.phrases.ingest import ingest_document
from moyo.privateside.phrases.schema import LABELS
from moyo.privateside.phrases.store import PhraseStore
from moyo.project import resolve_phrases_dir


def _store(project: Optional[str], corpus_dir: Optional[Path]) -> PhraseStore:
    try:
        root = resolve_phrases_dir(project=project, corpus_dir=corpus_dir, create=True)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    return PhraseStore(root)


_project_option = click.option(
    "--project",
    "-P",
    default=None,
    help="Project slug under projects/ (or MOYO_PROJECT). Phrases are per-project.",
)
_corpus_option = click.option(
    "--corpus-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override phrases directory (default: projects/<project>/phrases)",
)


@click.group()
def cli() -> None:
    """Sensitive phrases: Kimi extract, review, or add phrases by hand.

    Requires MOONSHOT_API_KEY for ingest. Approved phrases land in
    projects/<name>/phrases/ and are the source for Create Private Index.
    """


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@_project_option
@_corpus_option
def ingest(path: Path, project: Optional[str], corpus_dir: Optional[Path]) -> None:
    """Extract sensitive phrases with Kimi and queue them for review."""
    store = _store(project, corpus_dir)
    result = ingest_document(path, store, progress=click.echo)
    click.echo(
        f"Ingested {path.name}: kept={result['candidates']} "
        f"queued={result['queued']} duplicates={result['duplicates']}"
    )
    click.echo(f"Phrases dir: {store.root}")
    if result["queued"] == 0:
        click.echo("Nothing new to review.")
        return
    extra = f" --project {project}" if project else f" --corpus-dir {store.root}"
    click.echo(f"Review with: moyo-phrases review{extra}")
    for rec in result["pending"][:8]:
        click.echo(f"  [{rec.label}] {rec.text[:120]}")
    more = len(result["pending"]) - 8
    if more > 0:
        click.echo(f"  … {more} more")


@cli.command()
@_project_option
@_corpus_option
def review(project: Optional[str], corpus_dir: Optional[Path]) -> None:
    """Approve or reject queued phrases and set their labels."""
    store = _store(project, corpus_dir)
    pending = store.load_pending()
    if not pending:
        click.echo("No pending phrases. Ingest a document or use `add`.")
        return
    click.echo(f"{len(pending)} phrase(s) to review. Labels: {', '.join(LABELS)}")
    for i, rec in enumerate(pending, start=1):
        click.echo("")
        click.echo(f"[{i}/{len(pending)}] score={rec.score} why={rec.reason}")
        click.echo(f"  {rec.text}")
        click.echo(f"  suggested label: {rec.label}")
        action = click.prompt(
            "  [a]pprove  [e]dit-label  [r]eject  [s]kip  [q]uit",
            type=click.Choice(["a", "e", "r", "s", "q"], case_sensitive=False),
            default="a",
            show_choices=False,
        ).lower()
        if action == "q":
            click.echo("Stopped.")
            return
        if action == "s":
            continue
        if action == "r":
            store.decide(rec.id, approve=False)
            click.echo("  rejected")
            continue
        label = rec.label
        if action == "e":
            label = click.prompt("  label", default=rec.label)
        store.decide(rec.id, approve=True, label=label)
        click.echo(f"  approved as {label}")
    click.echo(f"Corpus: {store.corpus_path}")


@cli.command("add")
@click.argument("phrase")
@click.option("--label", default="other", show_default=True)
@_project_option
@_corpus_option
def add_phrase(phrase: str, label: str, project: Optional[str], corpus_dir: Optional[Path]) -> None:
    """Add one approved phrase without ingesting a document."""
    store = _store(project, corpus_dir)
    rec = store.add_manual(phrase, label)
    if rec is None:
        click.echo("Skipped (empty or already in corpus).")
        return
    click.echo(f"Approved {rec.id} [{rec.label}]: {rec.text}")


@cli.command("add-list")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--label", default="other", show_default=True, help="Default label")
@_project_option
@_corpus_option
def add_list(path: Path, label: str, project: Optional[str], corpus_dir: Optional[Path]) -> None:
    """Add phrases from a text file (one per line, or `phrase | label`)."""
    store = _store(project, corpus_dir)
    lines = path.read_text(encoding="utf-8").splitlines()
    added = store.add_manual_lines(lines, default_label=label)
    click.echo(f"Added {len(added)} phrase(s) from {path.name} → {store.corpus_path}")


@cli.command("list")
@_project_option
@_corpus_option
@click.option(
    "--status",
    type=click.Choice(["approved", "pending"]),
    default="approved",
)
@click.option("--json-out", "json_out", is_flag=True)
def list_phrases(project: Optional[str], corpus_dir: Optional[Path], status: str, json_out: bool) -> None:
    """Show approved or pending phrases."""
    store = _store(project, corpus_dir)
    rows = store.load_approved() if status == "approved" else store.load_pending()
    if json_out:
        click.echo(json.dumps([r.to_dict() for r in rows], indent=2, ensure_ascii=False))
        return
    click.echo(f"{len(rows)} {status} phrase(s) in {store.root}")
    for rec in rows:
        click.echo(f"  [{rec.label}] {rec.text}")


if __name__ == "__main__":
    cli()
