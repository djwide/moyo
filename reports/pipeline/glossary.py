"""Reader glossary printed at the end of the Basis Report and Exposure Snapshot.

Definitions mirror ``design-system/terminology.html`` so the standalone
companion page and the PDFs never drift apart.
"""

from __future__ import annotations

from typing import Any

_GROUPS: list[dict[str, Any]] = [
    {
        "title": "Findings and identifiers",
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
        "title": "Status labels",
        "terms": [
            (
                "Corroborated",
                "At least two distinct models agree on the same atomic fact.",
            ),
            (
                "Contested",
                "Sources disagree on a material point, not merely on wording.",
            ),
            (
                "Outlier",
                "Diverges sharply from consensus, or is unusually specific or extreme.",
            ),
            (
                "Unverified",
                "Plausible but weakly grounded; limited support in the exploration.",
            ),
            (
                "Model-specific",
                "Distinctive to one model family or one language-tagged run of it.",
            ),
        ],
    },
    {
        "title": "Score dimensions (1 low – 5 high)",
        "terms": [
            (
                "Sensitivity",
                "How high-stakes the disclosure is: policy risk, brand harm, "
                "privacy, safety, proprietary exposure. Sensitivity 4–5 counts as "
                "high-sensitivity in report totals.",
            ),
            (
                "Specificity",
                "How concrete and actionable the detail is — names, quantities, "
                "procedures — versus vague allusion.",
            ),
            (
                "Novelty",
                "How surprising the claim is relative to the expected public "
                "consensus on the topic.",
            ),
            (
                "Interestingness",
                "Editorial priority for a reader reviewing exposure.",
            ),
            (
                "Confidence",
                "How clearly the evidence shows the claim was actually stated "
                "(extraction and grounding confidence), drawn as filled dots.",
            ),
            (
                "Corroboration",
                "Number of distinct sources supporting the same clustered fact.",
            ),
        ],
    },
    {
        "title": "Severity bands",
        "terms": [
            ("High", "Sensitivity 4–5. Primary exposure concern."),
            ("Medium", "Sensitivity 3. Material detail worth review."),
            ("Low", "Sensitivity 2. Limited sensitivity, retained for completeness."),
            ("Info", "Sensitivity 1. Context or low-stakes public fact."),
        ],
    },
    {
        "title": "Sources and citations",
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
        "title": "Charts and metrics",
        "terms": [
            (
                "Finding classification profile",
                "Average specificity, sensitivity, corroboration, novelty, and "
                "confidence across all extracted claims.",
            ),
            (
                "Findings by LLM",
                "Each tested model scored by how many findings it produced and "
                "how sensitive those findings are. Bar height is the sum of "
                "finding sensitivities; color shows the high / medium / low / "
                "informational mix.",
            ),
            (
                "Model heatmap",
                "Which models produced disclosures in which categories.",
            ),
            (
                "Claim support graph",
                "How model outputs connect to claims and how claims group into "
                "higher-level exposures.",
            ),
            (
                "Model exposure dots",
                "Relative contribution of each model to overall exposure, scaled "
                "against the highest-scoring model in the run.",
            ),
            (
                "Findings count",
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
                "How a prompt was reworded before retrieval: paraphrase, "
                "translate, summarize, abstract, or typo.",
            ),
            (
                "Remediation / follow-up",
                "Recommended control or investigation step, mapped to the Idea "
                "Security Verification Framework (ISVF) control catalog.",
            ),
        ],
    },
]


def glossary_groups() -> list[dict[str, Any]]:
    """Glossary structure consumed by the report templates."""
    return [
        {
            "title": group["title"],
            "terms": [{"term": term, "definition": definition} for term, definition in group["terms"]],
        }
        for group in _GROUPS
    ]
