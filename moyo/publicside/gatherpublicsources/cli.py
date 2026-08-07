import json
from pathlib import Path
from typing import Optional

import click

from .crawler import PublicSourcesCrawler
from .schema import CrawlConfig, SourceType


@click.group()
def cli() -> None:
    """CLI for gathering public sources."""
    pass


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


@cli.command()
@click.option("--prompt", required=True, help="Naive, plain-language request to explore")
@click.option("--output", type=click.Path(), default=None, help="Path to write the markdown report")
@click.option("--output-dir", type=click.Path(), default="data/public_sources", show_default=True,
              help="Directory for the report when --output is not given")
@click.option("--seeds", type=int, default=5, show_default=True, help="Number of reworded query seeds")
@click.option(
    "--fuzz-mode",
    type=click.Choice(["basic", "full", "full-multilingual"], case_sensitive=False),
    default="basic",
    show_default=True,
    help="basic = paraphrase seeds only; full = abstract / summarize / typo (English); "
         "full-multilingual = full plus translation into Spanish, French, Mainland Chinese",
)
@click.option(
    "--language",
    "-l",
    "languages",
    multiple=True,
    help="Additional target language(s) for full-multilingual mode (repeatable). "
         "Added on top of the defaults Spanish, French, Mainland Chinese.",
)
@click.option("--workers", type=int, default=None,
              help="Max concurrent retrieval calls (default: one per configured LLM). Use 1 for sequential.")
@click.option("--no-summary", is_flag=True, help="Skip the synthesised combined summary")
@click.option(
    "--impact-definition",
    default=None,
    help="Extra high-impact criteria appended to the built-in definition for summary.md",
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
    prompt,
    output,
    output_dir,
    seeds,
    fuzz_mode,
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
    """Explore a naive prompt across all configured retrieval LLMs -> markdown report.

    Rewords the prompt into several retrieval queries via the local Ollama
    LLMFuzzer (``llama3.1:8b``, black-box — no target concept), sends each to
    every configured retrieval LLM (closed API, open API, local) in parallel,
    and writes one markdown document summarising everything learned, marked by
    source. Configure the retrieval LLMs via ``config/retrieval_llms.json`` or
    ``MOYO_RETRIEVAL_LLMS``; hot-swap the default LLM (used for summary
    synthesis) via ``MOYO_LLM_*``.

    ``--fuzz-mode basic`` (default) paraphrases into seeds; ``full`` rotates
    abstract, summarize, and typo (English only); ``full-multilingual`` adds a
    translated seed per language (defaults Spanish, French, Mainland Chinese;
    extend with ``--language``). Foreign-language retrieval bodies are translated
    back to English with a language annotation before the report is written.

    Claims ranking uses a built-in high-impact definition for classified,
    proprietary, and personal/sensitive information; refine it with
    ``--impact-definition``, ``--impact-definition-file``, or
    ``MOYO_IMPACT_DEFINITION`` / ``MOYO_IMPACT_DEFINITION_FILE``.
    """
    from moyo.llm.client import LLMClient, LLMSpec
    from .explorer import explore_and_save, format_retrieval_table

    default_llm = None
    if provider:
        default_llm = LLMClient(
            LLMSpec(provider=provider, model=model or "", api_key=api_key, base_url=base_url)
        )

    result = explore_and_save(
        prompt,
        output_directory=output_dir,
        output_path=output,
        default_llm=default_llm,
        num_seeds=seeds,
        fuzz_mode=fuzz_mode,
        extra_languages=list(languages) or None,
        summarize=not no_summary,
        workers=workers,
        impact_definition=impact_definition,
        impact_definition_files=list(impact_definition_file) or None,
        progress=lambda msg: click.echo(msg, err=True),
    )
    ok = sum(1 for r in result.results if r.ok)
    click.echo(
        f"Explored {len(result.seeds)} seeds x {len(result.llm_labels)} LLMs "
        f"({ok}/{len(result.results)} successful) [fuzz_mode={fuzz_mode}].",
        err=True,
    )
    click.echo(format_retrieval_table(result), err=True)
    click.echo(result.output_path)
    if result.summary_path:
        click.echo(result.summary_path)


if __name__ == "__main__":
    cli()
