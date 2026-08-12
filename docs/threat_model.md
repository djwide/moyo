# Threat Model — moyo Red Teaming Tool

## Overview

This document defines the threat model for moyo's LLM red teaming capability.
The tool is designed to help organizations discover whether an LLM system (e.g.
a fine-tuned or RAG-augmented model deployed internally) will reveal proprietary
or classified information under adversarial prompting.

Authorized use cases: internal security audits, LLM deployment validation, AI
governance assessments, penetration testing of AI systems with written permission.

---

## Assets Under Protection

| Asset Category | Examples |
|---|---|
| Financial data | Budget figures, salary bands, M&A targets, revenue projections |
| Personnel data | Org charts, performance reviews, HR records, compensation details |
| Strategic plans | Product roadmaps, competitive intelligence, acquisition targets |
| Technical IP | Internal APIs, architecture docs, algorithm details, source code |
| Legal / compliance | Ongoing litigation, regulatory violations, settlement terms |
| Operational secrets | Supply chain details, vendor contracts, pricing models |

---

## Threat Actors

### T1 — External Adversary (Black-Box)
- **Profile**: Competitor, journalist, foreign intelligence, or criminal actor
- **Knowledge**: No access to internal systems; queries the target LLM publicly
- **Goal**: Extract proprietary data by probing the LLM with inferred questions
- **Mapped mode**: Black-box (`moyo-redteam blackbox`)
- **Hypotheses**: The hypothesis engine can be seeded from a **probe path**
  (`--probe-path`) — a curated list of secrets a given target type would value
  (see `probe_paths/`). Seeds are injected as high-confidence hypotheses and used
  to focus LLM expansion. The same hypotheses can seed public multi-LLM
  exploration via `moyo-redteam blackbox-explore` → `moyo-gather explore -f`
  (or an in-process explore run) without changing the gather `explore` command.

### T2 — Insider Threat (White-Box)
- **Profile**: Disgruntled employee, contractor, or compromised account
- **Knowledge**: Has partial knowledge of what secrets the LLM may contain
- **Goal**: Confirm and extract specific classified information
- **Mapped mode**: White-box (`moyo-redteam whitebox`)
- **Grounding**: With `--private-index`, probes are grounded in the private
  corpus's mapcorpus centroids/topic tokens, and `--refine-rounds` iteratively
  rewrites each unrevealed probe toward the protected passages a response comes
  closest to (helper LLM only; secrets are never sent to it).

### T3 — Security Researcher / Auditor
- **Profile**: Authorized penetration tester or AI safety team
- **Knowledge**: Full access to secret inventory (white-box) or blind audit
- **Goal**: Measure exposure risk before deployment or after fine-tuning
- **Mapped mode**: Both modes; moyo is designed for this actor

---

## Attack Vectors

### AV-1: Direct Extraction (White-Box)
Ask the model directly about a known secret topic.
- **Strategy**: `DIRECT`
- **Mitigation**: Output filtering, topic blocking, RLHF refusal training

### AV-2: Indirect Inference (White-Box / Black-Box)
Approach the topic sideways through related questions without naming the secret.
- **Strategy**: `INDIRECT`
- **Mitigation**: Semantic similarity guardrails (Sente integration)

### AV-3: Roleplay / Persona Injection (White-Box)
Assign the model a role (consultant, employee) for whom disclosure is natural.
- **Strategy**: `ROLEPLAY`
- **Mitigation**: System prompt hardening, persona restriction

### AV-4: Few-Shot Priming (White-Box)
Provide partial fragments of the secret to prime context and invite completion.
- **Strategy**: `FEW_SHOT`
- **Mitigation**: Contextual injection detection (Sente static scan)

### AV-5: False Premise Injection (White-Box)
Frame the secret as already public and ask confirming questions.
- **Strategy**: `CONTEXT`
- **Mitigation**: False premise detection; refusal training for factual correction

### AV-6: Authority / Compliance Pressure (White-Box)
Use authority roles (DPO, legal counsel, system admin) to create pressure to disclose.
- **Strategy**: `AUTHORITY` — reuses `RoleAuthorityHacker` from `advanced_fuzzing_techniques.py`
- **Mitigation**: Role verification logic; rejection of authority claims in prompts

### AV-7: Hypothesis-Driven Blind Probing (Black-Box)
Iteratively generate exploratory questions guided by public OSINT and LLM refinement,
narrowing in on proprietary topics without prior knowledge.
- **Strategy**: Black-box `HypothesisEngine` + `BlindProber`
- **Mitigation**: Rate limiting, anomaly detection on output, egress filtering (SentePersonal)

### AV-8: Corpus Leakage via Embedding Proximity (Corpus-Level)
Identifying that private training data is semantically too close to public content,
revealing that private information may be reconstructible from public sources.
- **Strategy**: `BarrierAnalyzer` + `TwoLayerFuzzer` (existing moyo capability)
- **Mitigation**: Data deduplication before fine-tuning, minimum distance thresholds

---

## Trust Boundary Map

```
                        ┌─────────────────────────────────────┐
                        │          ORGANIZATION PERIMETER       │
                        │                                       │
  ┌──────────────┐      │  ┌────────────────────────────────┐  │
  │  Red Teamer  │─────────▶  moyo-redteam CLI              │  │
  │  (Auditor)   │      │  │  (whitebox / blackbox)         │  │
  └──────────────┘      │  └──────────────┬─────────────────┘  │
                        │                 │                     │
                        │    ┌────────────▼──────────────┐      │
                        │    │  TargetLLMClient           │      │
                        │    │  (probes the system under  │      │
                        │    │   test; isolated from      │      │
                        │    │   helper LLMs)             │      │
                        │    └────────────┬───────────────┘      │
                        │                 │  API call            │
                        │    ┌────────────▼───────────────┐      │
                        │    │  TARGET LLM                 │      │
                        │    │  (fine-tuned / RAG model    │      │
                        │    │   containing potential      │      │
                        │    │   proprietary data)         │      │
                        │    └────────────────────────────┘      │
                        │                                       │
                        │  ┌────────────────────────────────┐  │
                        │  │  Helper LLM (OpenAI/Anthropic) │  │
                        │  │  Used ONLY for probe generation │  │
                        │  │  Never sees target responses    │  │
                        │  └────────────────────────────────┘  │
                        │                                       │
                        │  ┌────────────────────────────────┐  │
                        │  │  SecretStore (whitebox only)   │  │
                        │  │  Encrypted at rest; never sent  │  │
                        │  │  to helper LLM                  │  │
                        │  └────────────────────────────────┘  │
                        └─────────────────────────────────────┘
```

---

## Key Security Properties

| Property | Implementation |
|---|---|
| Secret isolation | `SecretStore` secrets are NEVER sent to the helper LLM |
| Target isolation | `TargetLLMClient` is separate from helper LLMs; no cross-contamination |
| Full audit trail | All interactions logged via `target.save_interaction_log()` |
| Configurable thresholds | `similarity_threshold` and `specificity_threshold` are tunable per engagement |
| Graceful degradation | All components fall back to local/template-based operation when APIs are unavailable |

---

## Out of Scope

- Attacks against moyo itself (tool hardening is a separate concern)
- Attacks that require physical access to infrastructure
- Social engineering of human operators
- Training data extraction via gradient-based membership inference (model-level; not prompt-based)

---

## Residual Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Helper LLM leaks secret via probe text | Low | High | Secrets excluded from helper prompts; probes generated from labels only |
| False negative (secret revealed but not detected) | Medium | High | Lower `similarity_threshold`; supplement with LLM-based `SEMANTIC_OVERLAP_PROMPT` |
| False positive (benign response flagged) | Medium | Low | Review flagged results manually; adjust thresholds |
| Target LLM refusal prevents testing | Medium | Medium | Use varied strategies (authority, roleplay) and probe rephrasing |
