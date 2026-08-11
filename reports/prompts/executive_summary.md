# Executive snapshot copy

Using `report_data` JSON, return **only** a JSON object (no markdown fences, no `**` labels).

```json
{
  "headline": "What AI systems reveal",
  "summary": "2–4 sentence plain-English executive overview. No markdown.",
  "top_finding_blurb": "2–3 sentences on the lead finding. No markdown.",
  "why_it_matters": "1–2 plain-English sentences on practical impact.",
  "confidence_label": "High | Medium | Low",
  "confidence_rationale": "One short sentence on why that confidence.",
  "public_sources": [
    "Named public source or filing type (e.g. FEC filings)",
    "Second public source",
    "Third public source"
  ],
  "inference_chain": [
    "Step 1 grounded in evidence",
    "Step 2",
    "Step 3"
  ],
  "defensive_action": "One concrete recommended defensive action.",
  "exposure_teaser": "One short teaser sentence for the longer exposure chain (do not dump the full chain).",
  "exposure_chain": ["optional longer step 1", "step 2", "step 3"],
  "what_else": ["bullet 1", "bullet 2", "bullet 3"]
}
```

Rules:
- Do not invent findings. Prefer precision and disagreements already scored.
- `public_sources` must be real outlets/filings named in the findings (FEC, OpenSecrets, House disclosures, named newsrooms, court dockets, etc.). Exactly 3 when possible.
- `inference_chain` is **one** evidence-backed chain (3–5 short steps).
- Plain prose only in string values — never `**bold**`, headings, or markdown lists.

## Data

{{ report_data_json }}
