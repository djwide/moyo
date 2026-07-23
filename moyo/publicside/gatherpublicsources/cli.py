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


if __name__ == "__main__":
    cli()
