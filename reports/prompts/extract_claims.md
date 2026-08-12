# Extract claims from an exploration chunk

You are extracting atomic factual claims from a slice of a multi-LLM exploration
report. Return **only** a JSON array of claim objects (no markdown fences).

## Hard rules

- **English only:** write every `claim` and `raw_excerpt` in clear English, even
  when the chunk or prompt language is not English. Translate faithfully; do not
  leave foreign-language wording in either field. The chunk `language` metadata
  already records the prompt language — do not put foreign text in the claim.
- Preserve evidence: every claim MUST include `raw_excerpt` grounded in the chunk
  (translated to English when needed) and approximate `raw_start_line` /
  `raw_end_line` from the chunk metadata.
- Never omit a finding solely because it is unusual, disputed, sensitive, or
  produced by only one model. Classify it instead.
- Prefer precise claims (numbers, named entities, dates, amounts) over vague
  restatements.
- Do not invent citations or source labels. When a claim is grounded in a named
  source or URL that appears in the chunk, put that string in `citations`.
  The pipeline also attaches the chunk's Sources/References list automatically.
- Return `[]` if the chunk is a refusal, safety hedge, or only meta-advice
  (e.g. "check FEC.gov", "I cannot provide…", "no public records known") with
  no concrete factual finding.
- One atomic claim per concrete fact. Split compound bullets.
- Skip boilerplate: model headers, "Sources:" / URL laundry lists, and generic
  "where to look" guidance that is not itself a fact about the subject.

## Status values (use one)

`CORROBORATED` | `CONTESTED` | `OUTLIER` | `UNVERIFIED` | `MODEL-SPECIFIC`

At extraction time, set a provisional `status` and set `corroboration` to 1 for
this chunk's source. Score `confidence` for this chunk alone (how grounded the
excerpt looks). Clustering later raises `corroboration` (distinct LLMs),
`source_count` (distinct citations), and `confidence` when multiple models
and/or multiple sources agree.

## Scores (integers 1–5)

`sensitivity`, `specificity`, `novelty`, `confidence`, `interestingness`

## Object shape

```json
{
  "claim_id": "C0001",
  "claim": "...",
  "source_model": "...",
  "query_id": "Q01",
  "category": "proprietary_adjacent",
  "sensitivity": 4,
  "specificity": 5,
  "novelty": 5,
  "confidence": 3,
  "corroboration": 1,
  "interestingness": 5,
  "status": "OUTLIER",
  "raw_excerpt": "...",
  "raw_start_line": 10,
  "raw_end_line": 20,
  "citations": ["https://example.org/report", "OpenSecrets.org"]
}
```

## Chunk metadata (filled by the pipeline)

- `query_id`: {{ query_id }}
- `query_text`: {{ query_text }}
- `source_model`: {{ source_model }}
- `line_offset`: {{ line_offset }}
- `language`: {{ language }}

Ground each claim in `query_text` when framing what was asked.

## Chunk text

{{ chunk_text }}
