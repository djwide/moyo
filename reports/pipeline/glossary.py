"""Reader glossary printed at the end of the Basis Report and Exposure Snapshot.

Definitions mirror ``design-system/terminology.html`` so the standalone
companion page and the PDFs never drift apart.
"""

from __future__ import annotations

from typing import Any

_GROUPS: list[dict[str, Any]] = [
    {
        "title": "Findings And Identifiers",
        "terms": [
            (
                "Finding / claim",
                "An atomic factual statement taken from one model answer, with "
                "scores, a status, a source model, and a verbatim evidence excerpt.",
            ),
            (
                "C#### — Claim ID",
                "Stable identifier for a finding (e.g. C0082), used in the "
                "inventory, evidence graph, and remediation lists.",
            ),
            (
                "Q## — Query ID",
                "The exploration prompt that elicited the answer (e.g. Q18).",
            ),
            (
                "CL### — Cluster ID",
                "Group of paraphrased findings expressing the same fact across "
                "models or languages. Clustering sets the corroboration count.",
            ),
            (
                "CH### — Chain ID",
                "A related group of findings that reads as one exposure: several "
                "claims that reinforce the same conclusion.",
            ),
            (
                "Category",
                "Thematic bucket for the claim (for example proprietary-adjacent, "
                "public fact, campaign finance). It does not affect scores.",
            ),
            (
                "Raw excerpt / evidence",
                "Exact text copied from the exploration transcript with start and "
                "end line numbers, so every finding stays auditable.",
            ),
        ],
    },
    {
        "title": "Evidence Status",
        "terms": [
            (
                "Source-linked",
                "The model supplied an external source. Moyo has not confirmed "
                "that the source supports the claim.",
            ),
            (
                "Unvalidated inferences",
                "Findings without confirmed source support. Includes "
                "cross-model corroborated and single-model leads. Cross-model "
                "repetition is not factual validation.",
            ),
            (
                "Cross-model corroborated",
                "More than one model stated the claim, and no source URL was recovered. "
                "An unvalidated inference until checked against the record.",
            ),
            (
                "Single-model lead",
                "One model stated the claim, and no source URL was recovered. "
                "An unvalidated inference that requires independent research.",
            ),
            (
                "Contested",
                "The record disagrees on a material point, not merely on wording. "
                "This status stands on its own, even when a URL is present.",
            ),
        ],
    },
    {
        "title": "Research Significance",
        "terms": [
            (
                "High",
                "The claim would matter if it holds. This is not an evidence status.",
            ),
            (
                "Medium",
                "Worth a look, and still not evidence that the claim is true.",
            ),
            (
                "Low",
                "Retained for context.",
            ),
        ],
    },
    {
        "title": "Sources And Citations",
        "terms": [
            (
                "Source model",
                "The model, and where relevant the language tag, that produced the "
                "answer — for example Kimi (Moonshot kimi-k2.6) (French).",
            ),
            (
                "Source cite",
                "Short display form for model attribution: a model alias plus the "
                "number of other corroborating models. Kimi means this model only; "
                "Kimi + 5 means the primary model plus five peers.",
            ),
            (
                "S# — Source reference",
                "A real-world citation carried by the model answer (publication, "
                "filing, or URL). Full labels and URLs are listed in the Sources "
                "and Citations section.",
            ),
            (
                "Cited by",
                "How many findings in this run rest on that real-world source.",
            ),
        ],
    },
    {
        "title": "Charts And Metrics",
        "terms": [
            (
                "Findings by model",
                "Two bars per model: every finding it stated, and how many of "
                "those are high significance. Counts, not a weighted score.",
            ),
            (
                "Cross-model corroboration rate",
                "Share of distinct findings stated by two or more models.",
            ),
            (
                "Reproduction matrix",
                "Columns are high-significance findings; rows are models. A "
                "filled cell means that model stated the finding. Numbered "
                "columns are annotated under the chart.",
            ),
            (
                "Average scores vs corpus",
                "Dossier dot plot on a 1–5 line. The dot is this model's average; "
                "the tick is the average across all models. Specificity, "
                "sensitivity, novelty, and confidence are extraction scores; "
                "corroboration is how many models stated a finding.",
            ),
            (
                "Claim Support Graph",
                "How model outputs connect to claims and how claims group into "
                "higher-level exposures.",
            ),
            (
                "Findings Count",
                "Total claims retained after processing. Nothing is dropped for "
                "being unusual.",
            ),
            (
                "LLMs tested",
                "Number of distinct model sources that contributed answers.",
            ),
        ],
    },
    {
        "title": "Method",
        "terms": [
            (
                "Fuzz mode",
                "Rewriting regime used to vary the prompt (for example basic or "
                "multilingual).",
            ),
            (
                "Technique",
                "How a prompt was reworded before retrieval: original, paraphrase, "
                "abstract, summarize, translate, typo, or shuffle. Basic scans "
                "stay in English (original / paraphrase). Abstract and the other "
                "techniques are a la carte. Translate runs only when an extra "
                "language is selected (multilingual scan).",
            ),
            (
                "Attempted / received / substantive / claims",
                "Coverage of the retrieval roster: models the scan tried, models "
                "that returned any body, models whose visible answer was usable "
                "evidence, and models that contributed at least one extracted "
                "claim. llms_tested is the substantive count, not the configured "
                "roster.",
            ),
            (
                "Remediation / follow-up",
                "Recommended control or investigation step, mapped to the Idea "
                "Security Verification Framework (ISVF) control catalog.",
            ),
        ],
    },
]


def glossary_groups(audience: str = "organization") -> list[dict[str, Any]]:
    """Glossary structure consumed by the report templates."""
    del audience
    return [
        {
            "title": group["title"],
            "terms": [
                {"term": term, "definition": definition}
                for term, definition in group["terms"]
            ],
        }
        for group in _GROUPS
    ]
