# MLE Assessment: Compliance for Customer Communication

A two-layer compliance checker for collections conversations: semantic analysis (free) + dual LLM validation (for ambiguous cases).

---

## 🚀 Quick Start

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Add your API key
cp .env.example .env
# Edit .env and add OPENAI_API_KEY=sk-...
```

### Run the System

```bash
# Step 1: Generate test data (optional - sample data included)
python generate_conversations.py

# Step 2: Layer 1 - Fast semantic analysis (free)
python src/layer1_checker.py

# Step 3: Layer 2 - LLM validation for uncertain cases
python src/layer2_validator.py
```

---

## 📊 What It Does

**Layer 1** (Semantic Analysis)
- Analyzes all conversations using sentence embeddings
- Fast initial screening (~5 seconds for 200 conversations)
- Sends uncertain cases to Layer 2
- Cost: $0 (runs locally)
- **Note:** Requires calibration on domain-specific data for optimal accuracy

**Layer 2** (Dual LLM Validation)
- Two AI judges evaluate conversations
- Judge 1: Strict rule compliance
- Judge 2: Empathetic + compliant
- Both must agree to auto-validate
- Cost: ~$0.0013 per conversation
- **Accuracy:** 100% on validated cases (19/19 correct in testing)

**Result:** Layer 2 provides high-quality validation for cases Layer 1 routes to it

---

## 📁 Key Outputs

**For Decisions:**
- `data/layer1_output/auto_decided.json` - High-confidence results
- `data/layer2_output/validated_decisions.json` - LLM-validated results

**For Human Review:**
- `data/layer1_output/compliance_results.xlsx` - All Layer 1 results
- `data/layer2_output/human_review_needed.xlsx` - Cases where judges disagreed
- `data/layer2_output/layer2_all_results.xlsx` - Complete Layer 2 results

**Tracking:**
- `data/layer2_output/cost_report.json` - Token usage and costs

---

## 💰 Costs

| Component | Cost | Notes |
|-----------|------|-------|
| Generate 200 conversations | $0.08 | LLM-generated test data |
| Layer 1 (200 conversations) | $0.00 | Runs locally |
| Layer 2 (~40 conversations) | $0.05 | High-accuracy validation |
| **Total** | **$0.13** | |

**Layer 2 Validation Accuracy:** 100% (19/19 cases correctly validated in testing)

**Production scale (5,000/day):** ~$39/month (assuming 20% routed to Layer 2)

---

## ⚙️ Configuration

**Adjust Layer 1 thresholds:** Edit `config/severity_confidence.json`
```json
{
  "semantic_threshold": 0.60,
  "confidence_thresholds": {
    "compliant": 0.70,
    "medium": 0.75,
    "high": 0.80,
    "critical": 0.85
  }
}
```

**Change LLM models:** Edit `config/llm_config.json`
```json
{
  "judge1": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "system_prompt": "You are a strict compliance analyst..."
  },
  "judge2": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "system_prompt": "You are an empathetic compliance analyst..."
  }
}
```

---

## 📖 Documentation

- **[docs/APPROACH.md](docs/APPROACH.md)** - Design decisions and tradeoffs
- **[AI_USAGE.md](AI_USAGE.md)** - Models used, costs, scaling strategy
- **[PROACTIVE_CONTROLS.md](PROACTIVE_CONTROLS.md)** - Proactive prevention strategies
- **[docs/api/](docs/api/)** - Data contract schemas

---

## 🛡️ Proactive Controls (Beyond Detection)

This system detects violations after they occur. For prevention, see **[PROACTIVE_CONTROLS.md](PROACTIVE_CONTROLS.md)** which outlines:

- **Pre-conversation intelligence** - Brief agents with customer risk profile before engaging
- **Channel-specific controls** - Real-time warnings for chat/email, templates for phone
- **Strategic empathy** - Differentiated approach based on customer history and value
- **The 10-second pause** - Simple technique for phone calls to show empathy

**Key insight:** Prevention requires customer context. High-value customers get enhanced service, high-risk customers get strict compliance adherence, all customers get baseline respect.

---

## 🎯 What Gets Flagged

**Compliance violations:**
- Legal threats (sue, garnish, arrest)
- Contacting family/employer
- Abusive language
- Pressure tactics
- Misrepresentation

**Customer situations:**
- Product loss (never received, double charged)
- Substandard service (features broken, poor support)
- Financial hardship (job loss, medical bills)

---

**Total setup time:** ~5 minutes  
**Total runtime:** ~3 minutes for 200 conversations