# moyo Public Sources Crawler

## Overview

The public sources crawler aggregates open-source information relevant to a topic or a set of topic tokens. It powers the public-side corpus used for barrier analysis against private corpora.

## Components

- `moyo/publicside/gatherpublicsources/crawler.py` – Orchestrator with:
  - `crawl(topic: str, source_types: Optional[List[SourceType]])` – crawl by topic string
  - `crawl_with_tokens(tokens: List[str], ...)` – crawl by token list (bigrams, trigrams, singles)
  - Filtering/scoring pipeline and result persistence
- `schema.py` – Pydantic models for sources, configs, jobs
- `sources/` – Source adapters (patents, press releases, git commits, conference talks, leaks)
- `parsers/` – HTML/PDF/text parsers
- `enrichers/` – Classifiers, dedupe, metadata enrichment

## Token-driven Crawling

Use centroid/topic tokens derived from the private corpus (`privateside/mapcorpus/centroids.py`) to generate focused queries:

```python
from pathlib import Path
from moyo.privateside.mapcorpus import tokens_for_corpus
from moyo.publicside.gatherpublicsources.crawler import PublicSourcesCrawler

centroids, topic_tokens, labels, texts = tokens_for_corpus(Path('data/private/corpus.txt'), top_k=8)
tokens = [t for cluster in topic_tokens for t in cluster][:25]

crawler = PublicSourcesCrawler()
result = crawler.crawl_with_tokens(tokens)
print(result.message, result.output_path)
```

The orchestrator generates bigram/trigram/single queries from tokens, aggregates and filters results across adapters, and saves outputs under `data/public_sources/<topic>/` when enabled.

## Configuration

Configure crawling via `CrawlConfig` (see `schema.py`):
- `source_types`: which adapters to use
- per-source limits and date ranges
- minimum relevance/confidence thresholds
- output directory and persistence flags

## Outputs

- `sources.json` – normalized list of sources
- `summary.json` – counts, source types, date ranges, averages

## Next Steps

- Expand adapters (RSS/news/research)
- Parallelization and robust rate limiting
- Feedback loop from barrier probe to refine token/topic selection

