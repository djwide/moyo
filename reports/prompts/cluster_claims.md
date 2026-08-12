# Group claims that state the same atomic fact

You are clustering extracted claims. Decide which claims are paraphrases of the
**same atomic fact** (same who/what/when/amount/event), even when wording differs.

Return **only** JSON (no markdown fences):

```json
{ "groups": [["C0001", "C0004"], ["C0002"], ["C0003", "C0005", "C0008"]] }
```

## Rules

- Every input `claim_id` must appear in **exactly one** group.
- Put claims in the same group only when they assert the same concrete fact.
- Different facts about the same person/topic stay in **separate** groups
  (e.g. a Bank of China account vs late STOCK Act filings).
- Prefer merging when numbers/dates match or one claim is a stricter version of
  another (e.g. "$100,001–$250,000" vs "Bank of China account").
- Do not invent claim_ids. Use only ids from the input list.
- If unsure, keep claims in separate singleton groups.

## Claims

{{ claims_json }}
