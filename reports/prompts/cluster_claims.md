# Cluster and reconcile claims

You receive many claim objects from multi-LLM extraction. Merge paraphrases of
the **same atomic fact**, update `corroboration` counts (distinct `source_model`
labels), and assign final `status`:

- CORROBORATED — ≥ {{ corroboration_min_sources }} distinct sources agree
- CONTESTED — sources materially disagree
- OUTLIER — surprising / extreme vs consensus (keep; do not drop)
- UNVERIFIED — weak grounding
- MODEL-SPECIFIC — distinctive to one model

Never drop contested, outlier, sensitive, or single-source findings.

Return JSON: `{ "findings": [ ...merged claims... ], "disagreements": [ ... ] }`.

## Input

{{ claims_json }}
