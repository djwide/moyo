# Probe paths

A **probe path** is a curated list of *secrets a specific kind of target would find
valuable to know* — the raw material for hypothesis-driven blind probing in
`moyo-redteam blackbox`.

Each subdirectory is one **target customer / scenario**. Inside it, every `.txt`
file lists one secret (or damaging topic) per line. Lines starting with `#` are
comments and blank lines are ignored.

```
probe_paths/
├── political_opposition_research/   # 20 topics damaging to a candidate's campaign
│   └── secrets.txt
├── pharmaceutical_rd/               # proprietary pharma R&D information
│   └── secrets.txt
└── tech_company_ma/                 # tech M&A / corporate strategy
    └── secrets.txt
```

## How they feed the blind prober

In black-box mode the red-teamer does **not** know the target's real secrets. A
probe path supplies *hypotheses* — plausible high-value topics — which the
`HypothesisEngine` uses directly as seeds and (when a helper LLM is configured)
expands into concrete, varied probes:

```bash
# Use a bundled probe path by name:
moyo-redteam blackbox \
    --target-provider openai --target-model gpt-4o \
    --probe-path political_opposition_research \
    --domain "state gubernatorial campaign" \
    --rounds 8

# Or point at any directory / .txt file:
moyo-redteam blackbox --probe-path ./probe_paths/pharmaceutical_rd --rounds 6
```

Seeds from a probe path are always injected as high-confidence hypotheses. The
remaining hypotheses (up to `--n-hypotheses`) are generated from the configured
`--hypothesis-source` (`llm`, `public_corpus`, or `manual`), with the probe-path
topics used to focus LLM generation.

## Adding a new target customer

1. Create a new subdirectory: `probe_paths/<customer_name>/`.
2. Add one or more `.txt` files with one secret/topic per line.
3. Reference it with `--probe-path <customer_name>`.

> These lists describe topics to *test for* during authorized red-team
> assessments. They contain no actual confidential data.
