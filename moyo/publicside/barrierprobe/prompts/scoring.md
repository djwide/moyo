# Scoring Prompt Templates

These notes document how `barrier_analyzer.py` ranks semantic distance between
private and public index content.

---

## BARRIER_DISTANCE_SCORING

Used internally by `barrier_analyzer.py` — documents the semantic distance thresholds
applied to rank potential barrier breaches.

| Distance | Risk Level | Interpretation |
|----------|------------|----------------|
| ≤ 0.10   | HIGH       | Near-identical content; likely direct information leak |
| ≤ 0.30   | MEDIUM     | Substantial overlap; review for indirect exposure |
| ≤ 0.50   | LOW        | Moderate similarity; monitor for aggregation risk |
| > 0.50   | NONE       | Barrier intact for this phrase pair |
