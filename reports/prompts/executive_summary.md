# Executive snapshot copy

Using `report_data` JSON, return **only** a JSON object (no markdown fences, no `**` labels).

```json
{
  "headline": "Short finding, not a category header",
  "summary": "2–4 sentences of what the models actually disclosed. Name claim IDs, sources, or counts from the JSON. No markdown.",
  "top_finding_blurb": "2–3 sentences on the lead finding, anchored to its claim text and source. No markdown.",
  "why_it_matters": "1–2 sentences on practical impact for this run only. Omit if you would have to guess.",
  "confidence_label": "High | Medium | Low",
  "confidence_rationale": "One short sentence citing corroboration, contest, or a score already in the JSON.",
  "public_sources": [
    "Named public source or filing type from the findings",
    "Second public source",
    "Third public source"
  ],
  "inference_chain": [
    "Step 1 grounded in a claim ID or source",
    "Step 2",
    "Step 3"
  ],
  "defensive_action": "One concrete action, or empty string if remediation is not in the data.",
  "exposure_teaser": "One short sentence of the exposure path using names already in the JSON.",
  "exposure_chain": ["optional longer step 1", "step 2", "step 3"],
  "what_else": ["specific leftover finding", "specific leftover finding"],
  "model_commonality": [
    "One point several models agreed on, naming the models and a claim ID",
    "Second shared disclosure"
  ],
  "model_differences": [
    "One material disagreement or model-specific disclosure, naming the model and claim ID",
    "Second difference"
  ]
}
```

Rules:
- Do not invent findings. Prefer precision and disagreements already scored.
- `headline` is a statement of fact. Never "Overview", "Introduction", "Executive summary", or "What AI systems reveal" unless that is the only honest title and it is under 48 characters.
- `public_sources` must be real outlets/filings named in the findings. Exactly 3 when possible.
- `inference_chain` is **one** evidence-backed chain (3–5 short steps).
- `model_commonality` and `model_differences` contrast the tested models.
  Commonality is corroborated facts. Differences are unique, contested, or
  outlier disclosures. Name models and claim IDs already in the JSON.
- Plain prose only in string values — never `**bold**`, headings, or markdown lists.
- Do not use: delve, testament, landscape, underscore, pivotal, robust, leverage, revolutionize, comprehensive, it is important to note, in today's world.

## Data

{{ report_data_json }}
