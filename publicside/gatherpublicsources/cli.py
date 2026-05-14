import json
from pathlib import Path
from typing import Optional

import click

from .crawler import PublicSourcesCrawler
from .schema import SourceType


@click.group()
def cli() -> None:
    """CLI for gathering public sources."""
    pass


@cli.command()
@click.option("--topic", required=True, help="Topic query string")
@click.option("--output", type=click.Path(), default=None, help="Optional output directory")
def crawl(topic: str, output: Optional[str]) -> None:
    """Crawl public sources by topic string."""
    crawler = PublicSourcesCrawler()
    if output:
        crawler.config.output_directory = output
    res = crawler.crawl(topic)
    click.echo(json.dumps(res.dict(), indent=2, default=str))


@cli.command("crawl-tokens")
@click.option("--tokens", required=True, help="Comma-separated list of tokens")
@click.option("--output", type=click.Path(), default=None, help="Optional output directory")
def crawl_tokens(tokens: str, output: Optional[str]) -> None:
    """Crawl public sources using a list of tokens."""
    token_list = [t.strip() for t in tokens.split(",") if t.strip()]
    crawler = PublicSourcesCrawler()
    if output:
        crawler.config.output_directory = output
    res = crawler.crawl_with_tokens(token_list)
    click.echo(json.dumps(res.dict(), indent=2, default=str))


if __name__ == "__main__":
    cli()
