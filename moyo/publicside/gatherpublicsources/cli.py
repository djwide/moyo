import json
from pathlib import Path
from typing import Optional

import click

from .crawler import PublicSourcesCrawler
from .schema import CrawlConfig, SourceType


@click.group()
@click.option(
    "--test",
    "test_mode",
    is_flag=True,
    default=False,
    help=(
        "Use fake deterministic LLM clients (no network / API keys). "
        "Also settable via MOYO_TEST_MODE=1."
    ),
)
def cli(test_mode: bool) -> None:
    """CLI for gathering public sources."""
    if test_mode:
        from moyo.llm.testing import enable_test_mode
        enable_test_mode()
        click.echo("LLM test mode ON (fake deterministic clients).", err=True)


@cli.command()
@click.option("--topic", required=True, help="Topic query string")
@click.option("--output", type=click.Path(), default=None, help="Optional output directory")
def crawl(topic: str, output: Optional[str]) -> None:
    """Crawl public sources by topic string."""
    config = CrawlConfig(topic=topic)
    if output:
        config.output_directory = output
    crawler = PublicSourcesCrawler(config)
    res = crawler.crawl(topic)
    click.echo(json.dumps(res.dict(), indent=2, default=str))


@cli.command("crawl-tokens")
@click.option("--tokens", required=True, help="Comma-separated list of tokens")
@click.option("--output", type=click.Path(), default=None, help="Optional output directory")
def crawl_tokens(tokens: str, output: Optional[str]) -> None:
    """Crawl public sources using a list of tokens."""
    token_list = [t.strip() for t in tokens.split(",") if t.strip()]
    config = CrawlConfig(topic=", ".join(token_list) or "tokens_query")
    if output:
        config.output_directory = output
    crawler = PublicSourcesCrawler(config)
    res = crawler.crawl_with_tokens(token_list)
    click.echo(json.dumps(res.dict(), indent=2, default=str))


@cli.command("extract")
@click.option(
    "--project",
    "-P",
    default=None,
    help="Project slug; reads projects/<name>/public_sources/",
)
@click.option(
    "--sources-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Directory with sources.json / exploration.md (default: current project)",
)
@click.option(
    "--direction",
    default=None,
    help="Optional extra direction appended after each source as direction: …",
)
@click.option(
    "--direction-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Read direction from a text file",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Where to write extracted.json (default: <sources-dir>/extracted.json)",
)
def extract_cmd(project, sources_dir, direction, direction_file, output):
    """Extract relevant passages from gather output (Kimi).

    Writes extracted.json. Shows a completion bar of Kimi windows.
    Requires MOONSHOT_API_KEY. Build Public Corpus and naive compare use this file.
    """
    from moyo.project import get_project, load_saved_project
    from .extract import cli_extract_progress, run_public_extract

    extra = (direction or "").strip()
    if direction_file:
        file_text = direction_file.read_text(encoding="utf-8").strip()
        extra = f"{extra}\n{file_text}".strip() if extra else file_text

    root = sources_dir
    if root is None:
        name = project
        try:
            if name:
                proj = get_project(name, create=False)
            else:
                proj = load_saved_project()
        except FileNotFoundError as exc:
            raise click.UsageError(str(exc)) from exc
        if proj is None:
            raise click.UsageError(
                "Pass --sources-dir or --project (or set MOYO_PROJECT / select a GUI project)."
            )
        root = proj.public_sources_dir

    click.echo(f"Extracting from {root}", err=True)
    result = run_public_extract(
        root,
        direction=extra or None,
        output=output,
        progress=cli_extract_progress,
    )
    click.echo(f"Wrote {result['count']} passages to {result['path']}")


@cli.command()
@click.option(
    "--prompt",
    "-p",
    "prompts",
    multiple=True,
    help="Naive, plain-language request to explore (repeatable for multiple prompts)",
)
@click.option(
    "--prompts-file",
    "-f",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Text file with one prompt per line (blank lines ignored)",
)
@click.option("--output", type=click.Path(), default=None,
              help="Path to write the markdown report (single-prompt runs only)")
@click.option("--output-dir", type=click.Path(), default="data/public_sources", show_default=True,
              help="Directory for the report when --output is not given")
@click.option(
    "--seeds",
    type=int,
    default=3,
    show_default=True,
    help="Seed count for basic (n=3 => each strategy once); "
         "for multilingual, seeds per language group",
)
@click.option(
    "--fuzz-mode",
    type=click.Choice(["basic", "multilingual"], case_sensitive=False),
    default="basic",
    show_default=True,
    help="Language fan-out: basic = English seeds; multilingual = English + "
         "Spanish / French / Mandarin Chinese (extend with --language). "
         "Default strategy sets: basic = paraphrase/translate/summarize; "
         "multilingual = paraphrase/abstract/summarize. Override with -S "
         "(typo available a la carte).",
)
@click.option(
    "--strategy",
    "-S",
    "strategies",
    multiple=True,
    type=click.Choice(
        ["paraphrase", "translate", "summarize", "typo", "abstract"],
        case_sensitive=False,
    ),
    help="A la carte fuzz strategy (repeatable). Overrides the mode's default "
         "strategy rotation; --fuzz-mode still controls language fan-out.",
)
@click.option(
    "--language",
    "-l",
    "languages",
    multiple=True,
    help="Additional target language(s) for multilingual mode (repeatable). "
         "Added on top of the defaults Spanish, French, Mandarin Chinese.",
)
@click.option("--workers", type=int, default=None,
              help="Max concurrent retrieval and translation calls "
                   "(default: one per configured LLM for retrieval; same cap "
                   "for foreign-response translation). Use 1 for sequential.")
@click.option(
    "--no-summary",
    is_flag=True,
    hidden=True,
    help="Deprecated no-op: explore never writes summary.md",
)
@click.option(
    "--impact-definition",
    default=None,
    help="Extra high-impact criteria (used by moyo-gather summarize if you run it later)",
)
@click.option(
    "--impact-definition-file",
    type=click.Path(exists=True, dir_okay=False),
    multiple=True,
    help="File(s) whose text is appended as extra high-impact criteria (repeatable)",
)
@click.option("--provider", default=None,
              help="Override the default LLM provider (openai/anthropic/ollama/custom/echo)")
@click.option("--model", default=None, help="Override the default LLM model")
@click.option("--api-key", default=None, help="API key for the override default LLM")
@click.option("--base-url", default=None, help="Base URL for ollama/custom override default LLM")
def explore(
    prompts,
    prompts_file,
    output,
    output_dir,
    seeds,
    fuzz_mode,
    strategies,
    languages,
    workers,
    no_summary,
    impact_definition,
    impact_definition_file,
    provider,
    model,
    api_key,
    base_url,
):
    """Explore one or more naive prompts across configured retrieval LLMs.

    Provide prompts with repeatable ``--prompt`` / ``-p`` and/or
    ``--prompts-file`` / ``-f`` (one prompt per line). Each prompt gets its own
    ``<output-dir>/<slug>/exploration.md``. Explore does not write
    ``summary.md``. ``--output`` is only valid for a single prompt.

    Rewords each prompt into several retrieval queries via the local Ollama
    LLMFuzzer (``llama3.1:8b``, black-box — no target concept), sends each to
    every configured retrieval LLM (closed API, open API, local) in parallel,
    and writes one markdown document of everything returned, marked by source.
    Configure the retrieval LLMs via ``config/retrieval_llms.json`` or
    ``MOYO_RETRIEVAL_LLMS``.

    ``--fuzz-mode basic`` (default) uses English seeds; ``multilingual`` fans
    out English plus Spanish / French / Mandarin Chinese (extend with
    ``--language``). Default strategy rotations match the mode; override a la
    carte with repeatable ``--strategy`` / ``-S``. Every seed is sent to every
    retrieval LLM. Foreign-language responses are translated back to English;
    section headers keep the source language annotation.
    """
    from moyo.llm.client import LLMClient, LLMSpec
    from .explorer import explore_and_save, explore_and_save_many, normalize_prompts

    del no_summary  # deprecated; explore never writes summary.md

    all_prompts = list(prompts)
    if prompts_file:
        with open(prompts_file, encoding="utf-8") as fh:
            all_prompts.extend(line.strip() for line in fh if line.strip())
    all_prompts = normalize_prompts(all_prompts)
    if not all_prompts:
        raise click.UsageError("Provide at least one prompt via --prompt/-p or --prompts-file/-f.")
    if output and len(all_prompts) > 1:
        raise click.UsageError("--output can only be used with a single prompt.")

    default_llm = None
    if provider:
        default_llm = LLMClient(
            LLMSpec(provider=provider, model=model or "", api_key=api_key, base_url=base_url)
        )

    common = dict(
        output_directory=output_dir,
        default_llm=default_llm,
        num_seeds=seeds,
        fuzz_mode=fuzz_mode,
        strategies=list(strategies) or None,
        extra_languages=list(languages) or None,
        summarize=False,
        workers=workers,
        impact_definition=impact_definition,
        impact_definition_files=list(impact_definition_file) or None,
        progress=lambda msg: click.echo(msg, err=True),
    )

    if len(all_prompts) == 1:
        results = [
            explore_and_save(
                all_prompts[0],
                output_path=output,
                **common,
            )
        ]
    else:
        click.echo(f"Exploring {len(all_prompts)} prompts…", err=True)
        results = explore_and_save_many(all_prompts, **common)

    for result in results:
        ok = sum(1 for r in result.results if r.ok)
        click.echo(
            f"[{result.prompt}] {len(result.seeds)} seeds x {len(result.llm_labels)} LLMs "
            f"({ok}/{len(result.results)} successful) [fuzz_mode={fuzz_mode}].",
            err=True,
        )
        click.echo(result.output_path)


@cli.command("summarize")
@click.option(
    "--exploration",
    "-e",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to exploration.md (or pass a directory via --dir)",
)
@click.option(
    "--dir",
    "explore_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Directory containing exploration.md; writes summary.md beside it",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Where to write summary.md (default: beside exploration.md)",
)
@click.option(
    "--impact-definition",
    default=None,
    help="Extra high-impact criteria appended to the built-in definition",
)
@click.option(
    "--impact-definition-file",
    type=click.Path(exists=True, dir_okay=False),
    multiple=True,
    help="File(s) whose text is appended as extra high-impact criteria (repeatable)",
)
@click.option(
    "--with-deliverable",
    is_flag=True,
    help="After summary.md, also build deliverable.md via Grok (xAI)",
)
@click.option("--provider", default=None,
              help="Override the summary LLM provider (openai/anthropic/ollama/custom/echo)")
@click.option("--model", default=None, help="Override the summary LLM model")
@click.option("--api-key", default=None, help="API key for the override summary LLM")
@click.option("--base-url", default=None, help="Base URL for ollama/custom override summary LLM")
def summarize(
    exploration,
    explore_dir,
    output,
    impact_definition,
    impact_definition_file,
    with_deliverable,
    provider,
    model,
    api_key,
    base_url,
):
    """Synthesise summary.md from an existing exploration.md (no re-explore).

    Parses the compiled findings in the report and runs the same claims-brief
    synthesis used at the end of ``explore``. Useful after pruning a report or
    when explore was run with ``--no-summary``. Pass ``--with-deliverable`` to
    also author ``deliverable.md`` via Grok (xAI).
    """
    from moyo.llm.client import LLMClient, LLMSpec
    from .explorer import summarize_exploration

    if exploration is None and explore_dir is None:
        raise click.UsageError("Provide --exploration PATH or --dir DIRECTORY")
    if exploration is not None and explore_dir is not None:
        raise click.UsageError("Use only one of --exploration or --dir")

    exploration_path = (
        Path(exploration) if exploration is not None else Path(explore_dir) / "exploration.md"
    )
    if not exploration_path.is_file():
        raise click.ClickException(f"exploration.md not found: {exploration_path}")

    default_llm = None
    if provider:
        default_llm = LLMClient(
            LLMSpec(provider=provider, model=model or "", api_key=api_key, base_url=base_url)
        )

    result = summarize_exploration(
        exploration_path,
        output_path=str(output) if output else None,
        default_llm=default_llm,
        impact_definition=impact_definition,
        impact_definition_files=list(impact_definition_file) or None,
        progress=lambda msg: click.echo(msg, err=True),
    )
    click.echo(
        f"Summarised {len(result.seeds)} quer(ies) / "
        f"{sum(1 for r in result.results if r.ok)} usable answer(s).",
        err=True,
    )
    click.echo(result.summary_path)

    if with_deliverable:
        from .deliverable import build_deliverable

        dresult = build_deliverable(
            exploration_path,
            summary_path=result.summary_path,
            progress=lambda msg: click.echo(msg, err=True),
        )
        click.echo(dresult.html_path or dresult.output_path)


@cli.command("deliverable")
@click.option(
    "--exploration",
    "-e",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to exploration.md",
)
@click.option(
    "--summary",
    "-s",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to summary.md (default: beside exploration.md)",
)
@click.option(
    "--dir",
    "explore_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Directory containing exploration.md + summary.md",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Where to write deliverable.md (default: beside exploration.md)",
)
def deliverable(exploration, summary, explore_dir, output):
    """Build deliverable.html from exploration.md + summary.md via Grok (xAI).

    Four sections: Executive Exposure Summary, Evidence Graph, Findings &
    Basis Chains, Mitigation Playbook. Writes a browser-ready HTML file
    (open it in Chrome/Edge/Firefox). Requires ``XAI_API_KEY`` (or
    ``MOYO_DELIVERABLE_API_KEY``).
    """
    from .deliverable import build_deliverable

    if explore_dir is not None:
        if exploration is not None or summary is not None:
            raise click.UsageError("Use --dir alone, or --exploration/--summary")
        exploration_path = Path(explore_dir) / "exploration.md"
        summary_path = Path(explore_dir) / "summary.md"
    else:
        if exploration is None:
            raise click.UsageError("Provide --dir DIRECTORY or --exploration PATH")
        exploration_path = Path(exploration)
        summary_path = Path(summary) if summary else exploration_path.parent / "summary.md"

    if not exploration_path.is_file():
        raise click.ClickException(f"exploration.md not found: {exploration_path}")
    if not summary_path.is_file():
        raise click.ClickException(
            f"summary.md not found: {summary_path} (run moyo-gather summarize first)"
        )

    result = build_deliverable(
        exploration_path,
        summary_path=summary_path,
        output_path=str(output) if output else None,
        progress=lambda msg: click.echo(msg, err=True),
    )
    click.echo(result.html_path or result.output_path)


@cli.command("check-llms")
@click.option("--workers", type=int, default=None,
              help="Max concurrent probes (default: one per configured LLM). "
                   "Use 1 for sequential.")
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit machine-readable JSON instead of the status table",
)
def check_llms(workers, as_json):
    """Probe configured retrieval LLMs (same preflight as explore) and exit.

    Does not reword, retrieve topic answers, translate, or write reports.
    Exit code is 0 when every LLM returns ok, 1 when any fail.
    """
    from moyo.llm.registry import get_retrieval_llms
    from .explorer import check_retrieval_llms, format_llm_status_table

    llms = get_retrieval_llms()
    if not llms:
        click.echo("No retrieval LLMs configured.", err=True)
        raise SystemExit(1)

    statuses = check_retrieval_llms(
        llms,
        progress=None if as_json else (lambda msg: click.echo(msg, err=True)),
        workers=workers,
    )
    n_ok = sum(1 for s in statuses if s.status == "ok")
    n_fail = sum(1 for s in statuses if s.status != "ok")

    if as_json:
        click.echo(
            json.dumps(
                [
                    {"name": s.name, "status": s.status, "reason": s.reason}
                    for s in statuses
                ],
                indent=2,
            )
        )
    else:
        click.echo(format_llm_status_table(statuses))
        click.echo(
            f"LLM preflight: {n_ok}/{len(statuses)} working"
            + (f" ({n_fail} not ok)" if n_fail else ""),
            err=True,
        )

    raise SystemExit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    cli()
