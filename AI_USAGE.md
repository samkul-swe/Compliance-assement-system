# AI Usage Log

---

## Models and Tools Used

| Model / Tool | Purpose |
|--------------|---------|
| Claude 3.5 Sonnet (claude.ai) | Architecture design, code review, debugging |
| GPT-4o-mini (OpenAI API) | Conversation generation, Layer 2 dual validation |
| Sentence-BERT (all-MiniLM-L6-v2) | Layer 1 semantic compliance checking |

---

## Token Usage and Cost

### Development

| Model | Tokens (approx) | Cost |
|-------|-----------------|------|
| Claude 3.5 Sonnet | ~65,000 | $0.00 (free tier) |

### Data Generation

| Model | Input Tokens | Output Tokens | Cost |
|-------|--------------|---------------|------|
| GPT-4o-mini | ~100,000 | ~60,000 | $0.08 |

### Layer 2 Validation (38 conversations)

| Model | Input Tokens | Output Tokens | Cost |
|-------|--------------|---------------|------|
| GPT-4o-mini (Judge 1) | ~34,000 | ~11,000 | $0.0246 |
| GPT-4o-mini (Judge 2) | ~34,000 | ~11,000 | $0.0246 |
| **Subtotal** | **~68,000** | **~22,000** | **$0.0493** |

### Total Assessment Cost

| Component | Total Tokens | Cost |
|-----------|--------------|------|
| Development | ~65,000 | $0.00 |
| Generation | ~160,000 | $0.08 |
| Validation | ~90,000 | $0.05 |
| **TOTAL** | **~315,000** | **$0.13** |

**Per conversation cost (Layer 2 validation): $0.0013**

---

## Scaling to Production

### Production Estimate: 5,000 conversations/day

**Layer 1 (Semantic):**
- Processes: 5,000 conversations
- Auto-decides: 4,000 (80%)
- Cost: $0 (runs locally)

**Layer 2 (Dual LLM):**
- Processes: 1,000 (20%)
- Calls: 2,000 (dual judges)
- Cost: $1.30/day = **$39/month**

**Total Production Cost: $39/month**

### Optimization Strategies

**Caching (30% reduction):**
- Cache results by conversation hash
- Expected savings: ~$12/month
- **New cost: $27/month**

**Model Selection:**
- Switch to Gemini Flash (50% cheaper)
- **New cost: $20/month**

**Adaptive Dual Judges:**
- Use single judge for medium confidence (10%)
- Use dual judges only for very low confidence (5%)
- **New cost: $15/month**

**Optimized Production Cost: $15-20/month**

### Rate Limits & Reliability

- OpenAI rate limit: 3,500 req/min
- Our load: 1.4 req/min
- **No rate limit concerns**

**Reliability:**
- Retry logic (3 attempts)
- Circuit breaker on 5% error rate
- Fallback to Layer 1 only if APIs fail

---

## Cost Controls

**Guardrails:**
- Daily spend limit: $2.00
- Per-conversation token limit: 2,000
- Alert on cost spikes

**Monitoring:**
- Track cost per conversation (target: < $0.002)
- Track automation rate (target: > 85%)
- Alert if human review queue > 100

---

## Key Decisions

**Why GPT-4o-mini?**
- Good balance of cost ($0.15/1M input) and quality
- Reliable JSON output format
- Fast response times

**Why Sentence-BERT?**
- Free, runs locally
- Good semantic understanding
- Fast inference on CPU

**Why Dual Judges?**
- Different perspectives (strict vs empathetic)
- Only 20% of conversations need it
- Marginal cost increase for quality improvement