# Cluster and reconcile claims

You receive many claim objects from multi-LLM extraction. Merge paraphrases of
the **same atomic fact**, update `corroboration` (distinct `source_model`
labels), `source_count` (distinct citations across the cluster), raise
`confidence` when multiple LLMs and/or multiple citations agree, and assign
final `status`:

- CORROBORATED — ≥ {{ corroboration_min_sources }} distinct LLMs agree
- CONTESTED — sources materially disagree
- OUTLIER — surprising / extreme vs consensus (keep; do not drop)
- UNVERIFIED — weak grounding
- MODEL-SPECIFIC — distinctive to one model

Confidence boost (applied on top of extraction confidence, capped at 5):
+1 for ≥2 LLMs, +1 for ≥3 LLMs, +1 for ≥2 citations, +1 for ≥3 citations.

Never drop contested, outlier, sensitive, or single-source findings.

Return JSON: `{ "findings": [ ...merged claims... ], "disagreements": [ ... ] }`.

## Input

{{ claims_json }}
