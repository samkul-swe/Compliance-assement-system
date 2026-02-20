# Compliance Checker for Customer Service Conversations

A two-layer system that detects compliance violations in debt collection calls. Built as a learning project — the architecture is real, the gaps are documented honestly.

---

## What It Does

Analyses agent messages in customer service conversations and flags violations that warrant action — medium, high, and critical severity only. Low severity and compliant conversations are intentionally let through.

**Layer 1 — Semantic detection (free, fast)**
Uses sentence embeddings to compute a compliance score for each agent message. Scores above a learned threshold are flagged as violations. Runs locally, no API cost.

**Layer 2 — LLM severity adjudication (paid, accurate)**
Two LLM judges with different personas determine severity for flagged cases. Judge 1 is strict. Judge 2 considers context. If they agree — decision confirmed. If not — human review.

---

## Current Accuracy

Evaluated on 200 synthetic conversations (40 per severity level):

| What matters | Accuracy |
|---|---|
| Medium violations detected | 100% |
| High violations detected | 100% |
| Critical violations detected | 100% |
| Layer 2 severity decisions | 100% |

**Known gaps:**
- Low severity violations (~0% detection) — intentional. Low violations look too similar to compliant language for the semantic layer to reliably separate them. The cost of missing them is low.
- Compliant conversations are sometimes flagged as violations — the gate between compliant and medium territory is the weakest part of the system.
- Rule attribution is approximate — the system detects that something is wrong but does not always identify which specific rule was violated.

---

## Quick Start

```bash
pip install -r requirements.txt

cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

```bash
# Generate test conversations (or use included sample data)
python generate_conversations.py

# Layer 1 — semantic compliance scoring
python src/layer1_checker.py

# Layer 2 — LLM severity adjudication for flagged cases
python src/layer2_validator.py

# Evaluate against ground truth
python src/evaluate.py
```

---

## How It Works

**Compliance score**

Each agent message gets a single score:

```
score = sim(message, violation_pool) - sim(message, compliant_pool)
```

Positive = closer to violation language. Negative = closer to compliant language.

**The gate**

A threshold is learned from training examples — the midpoint between the mean score of low violations and the mean score of medium violations. Anything above the gate goes to Layer 2.

**Severity**

Layer 2 receives a focused question: is this medium, high, or critical? Not — is this compliant? That is already decided by Layer 1.

---

## Configuration

**Adjust the gate sensitivity:** `config/severity_confidence.json`
```json
{
  "boundary_margin": 0.02
}
```
Increase boundary_margin to send more cases to Layer 2. Decrease to auto-decide more aggressively.

**Change LLM models:** `config/llm_config.json`
```json
{
  "judge1": { "provider": "openai", "model": "gpt-4o-mini" },
  "judge2": { "provider": "openai", "model": "gpt-4o-mini" },
  "agreement_threshold": 0.75
}
```

---

## Outputs

```
data/layer1_output/
  auto_decided.json       — conversations Layer 1 handled
  llm_review_queue.json   — conversations sent to Layer 2
  compliance_results.xlsx — full results for inspection

data/layer2_output/
  confirmed_severities.json  — Layer 2 confirmed decisions
  human_review_needed.json   — judges disagreed, needs human
  layer2_results.xlsx        — full Layer 2 results with reasoning
  cost_report.json           — token usage and costs
```

---

## Cost

| Step | Cost |
|---|---|
| Generate 200 conversations | ~$0.08 |
| Layer 1 — 200 conversations | $0.00 |
| Layer 2 — ~120 conversations | ~$0.15 |
| Evaluate | $0.00 |
| **Total** | **~$0.23** |

At production scale (5,000 conversations/day), Layer 2 cost dominates. Reducing the percentage routed to Layer 2 is the main optimisation lever.

---

## What Is Not Production Ready

This is a prototype. Before using it in a real compliance context:

- **Replace synthetic training data with real labelled conversations.** The compliance score threshold is learned from LLM-generated phrases, not real agent speech. It will behave differently on real data.
- **Proper train/validation/test splits.** The current threshold is tuned on the same data it is evaluated against.
- **Calibrated confidence.** The compliance score is a distance measure, not a probability. It does not tell you how confident the system is — only whether something is above or below a line.
- **Rule attribution.** The system needs to reliably identify which specific rule was violated, not just the severity.
- **Monitoring.** No mechanism to detect when the score distribution drifts as agent language changes over time.

---

## Feedback

If you are looking at this and have thoughts on the architecture, the gaps, or what you would do differently — I am genuinely interested. Open an issue or reach out directly.

System is a work in progress. Little by little.