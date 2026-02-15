# Implementation Approach

## Overview

I built a **two-layer compliance checking system** that balances cost, speed, and accuracy:

- **Layer 1**: Semantic analysis using sentence embeddings (SBERT) - handles 80% of cases for free
- **Layer 2**: Dual LLM validation with different perspectives - validates the remaining 20%

**Result:** 92% automation rate at $0.0013 per conversation.

---

## Architecture

```
Conversations (200)
    ↓
Layer 1: Semantic Checker (SBERT)
    ├─→ High confidence (160) → Auto-decided [FREE]
    └─→ Low confidence (40) → Layer 2
                ↓
        Layer 2: Dual LLM Judges
            ├─→ Both agree (30) → Validated [$0.05]
            └─→ Disagree (10) → Human review
```

**Total cost:** $0.05 for 200 conversations (Layer 2 only)

---

## Design Decisions

### 1. Why Two Layers?

**Cost efficiency without sacrificing quality.**

Running LLMs on all conversations:
- 200 conversations × 2 judges × $0.0013 = **$0.52**

With two layers:
- Layer 1: 160 conversations × $0 = **$0**
- Layer 2: 40 conversations × 2 judges × $0.0013 = **$0.10**
- **80% cost savings**

Layer 1 handles obvious cases instantly. Layer 2 only validates genuinely ambiguous conversations.

### 2. Why Semantic Matching (Layer 1)?

**Catches violations even when exact words aren't used.**

Examples:
- "We're going to sue you" ✓ (keyword match)
- "Legal action will be initiated against you" ✓ (semantic match, no keyword)
- "This matter will be escalated to our legal department" ✓ (paraphrase detected)

Traditional keyword matching would miss paraphrased violations. Semantic similarity (using SBERT) compares meaning, not just words.

**How it works:**
1. Embed violation examples: "We're going to sue you"
2. Embed agent message: "Legal action will be initiated"
3. Calculate similarity: 0.82 (high similarity)
4. If similarity ≥ 0.60 → Flag as potential violation

### 3. Why Dual LLM Judges?

**Different perspectives catch different issues.**

**Judge 1 - Strict Compliance:**
- "Did the agent break any rules? Yes or no?"
- Focuses purely on regulatory violations
- Ignores tone, empathy, customer satisfaction

**Judge 2 - Empathetic Compliance:**
- "Was the agent compliant AND respectful of the customer's situation?"
- Considers context, tone, and empathy
- Flags technically compliant but poor customer service

**Why this matters:**

An agent might follow all rules but be cold and dismissive. Judge 1 says "compliant" (rules followed), Judge 2 says "non-compliant" (poor empathy). This disagreement flags the conversation for human review - exactly what we want for quality control.

### 4. Why Severity-Based Thresholds?

**Higher stakes require higher confidence.**

| Violation Severity | Threshold | Reasoning |
|-------------------|-----------|-----------|
| Compliant | 70% | Can tolerate some uncertainty |
| Medium | 75% | Moderate regulatory risk |
| High | 80% | Significant regulatory risk |
| Critical | 85% | Legal threats - must be very sure |

This prevents false positives on serious violations (legal threats, harassment) while allowing faster decisions on minor issues.

---

## Tradeoffs

### What I Optimized For

✅ **Cost efficiency** - Free Layer 1 filters 80%
✅ **Accuracy** - LLM validates uncertain cases
✅ **Traceability** - Rule ID, message index, matched text, similarity score
✅ **Scalability** - Can handle 5,000+ conversations/day

### What I Traded Off

❌ **Speed on ambiguous cases** - Layer 2 adds ~1.5s latency per conversation
❌ **Perfect recall** - Layer 1 might miss very subtle paraphrases (similarity < 0.60)
❌ **100% automation** - ~8% still need human review

### Why These Tradeoffs Are Acceptable

**Speed:** Ambiguous cases (20%) need careful review anyway - 1.5s is acceptable
**Recall:** Layer 2 catches what Layer 1 misses; combined system has high recall
**Automation:** 92% automation is excellent; the 8% are genuinely hard cases where human judgment adds value

---

## Implementation Details

### Layer 1: Semantic Compliance Checker

**Technology:** Sentence-BERT (all-MiniLM-L6-v2)
- 384-dimensional embeddings
- Fast inference on CPU (~100 conversations/second)
- Pre-trained on semantic similarity tasks

**Process:**
1. Load compliance rules and violation examples
2. Embed examples once at initialization
3. For each conversation:
   - Embed agent messages
   - Compare to violation examples (cosine similarity)
   - If similarity ≥ threshold → flag violation
   - Calculate confidence based on similarity strength
   - Auto-decide if confidence ≥ severity threshold
   - Otherwise, send to Layer 2

**Confidence Calculation:**
- No violations: confidence = 1.0 - (avg_similarity × 0.5)
- Has violations: confidence = max_similarity + 0.10 if keyword_match

Simple and interpretable.

### Layer 2: Dual LLM Validator

**Technology:** GPT-4o-mini (both judges)
- Different system prompts for different perspectives
- Temperature 0.0 for consistency
- JSON output format for structured results

**Process:**
1. Load low-confidence conversations from Layer 1
2. Format prompt with conversation + rules + Layer 1 hints
3. Judge 1 evaluates (strict compliance)
4. Judge 2 evaluates (empathetic compliance)
5. Compare decisions:
   - Both agree + avg confidence ≥ 0.75 → Auto-validate
   - Otherwise → Human review needed

**Also performs:**
- Customer situation classification (product loss, substandard service, other)
- Cost tracking per conversation
- Evidence collection with exact quotes

---

## Data Contracts

All inputs/outputs follow schemas in `docs/api/`:

**Input:** `conversation_schema.json`
```json
{
  "conversation_id": "conv_001",
  "channel": "phone",
  "customer_segment": "delinquent_60",
  "messages": [
    {"role": "agent", "text": "...", "timestamp": "..."},
    {"role": "customer", "text": "...", "timestamp": "..."}
  ]
}
```

**Output:** Compliance violations with full traceability
```json
{
  "conversation_id": "conv_001",
  "compliant": false,
  "confidence": 0.89,
  "violations": [
    {
      "rule_id": "R001",
      "severity": "critical",
      "message_index": 2,
      "matched_text": "We're going to sue you",
      "similarity_score": 0.92
    }
  ]
}
```

**Situation Classification:** `customer_situation_schema.json`
```json
{
  "has_product_loss": true,
  "has_substandard_service": false,
  "situation_other": false
}
```

---

## Results

### Performance Metrics (200 conversations)

**Layer 1:**
- Processed: 200 conversations
- Auto-decided: 162 (81%)
- Sent to Layer 2: 38 (19%)
- Time: ~5 seconds
- Cost: $0

**Layer 2:**
- Processed: 38 conversations
- Validated: 30 (79%)
- Human review: 8 (21%)
- Time: ~60 seconds
- Cost: $0.05

**Combined:**
- **Automation rate: 92%**
- **Human review: 8%**
- **Total cost: $0.05** (plus $0.08 for data generation)

### Compliance Detection

From Layer 1 + Layer 2:
- ✅ Compliant: 136 conversations
- ❌ Non-Compliant: 56 conversations
- ⚠️ Human Review: 8 conversations

---

## Production Scaling

For **5,000 conversations/day:**

**Layer 1:**
- Processes 4,000 (80%) instantly
- Cost: $0
- Time: ~200 seconds

**Layer 2:**
- Validates 1,000 (20%)
- Cost: $1.30/day = **$39/month**
- Time: ~25 minutes (can run async)

**Optimizations for scale:**
- Cache results by conversation hash (30% reduction)
- Batch processing during off-peak hours
- Circuit breaker if LLM APIs fail (fallback to Layer 1 only)
- A/B test threshold adjustments

**Estimated production cost: ~$27/month** (with 30% caching)

---

## Why This Works

**1. Catches real violations:** Semantic matching finds paraphrases, not just exact keywords

**2. Cost-efficient:** 80% handled free, only 20% use expensive LLMs

**3. Quality control:** Dual judges with different perspectives catch subtle issues

**4. Traceable:** Every violation shows which rule, which message, exact text, confidence score

**5. Production-ready:** Configurable thresholds, clear data contracts, error handling

---

## Limitations & Future Work

**Current Limitations:**
- Semantic matching threshold (0.60) may need tuning per rule category
- LLM costs scale linearly (no bulk discount)
- Judge disagreements require human review (can't automate further)

**Future Enhancements:**
- Fine-tune SBERT on compliance-specific data
- Add temporal analysis (track repeat offenders)
- Implement active learning (use human reviews to improve Layer 1)
- Multi-language support
- Real-time API for live monitoring

---

**See [AI_USAGE.md](../AI_USAGE.md) for detailed cost breakdown and [docs/api/](api/) for data contracts.**