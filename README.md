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
python src/generate_conversations.py

# Step 2: Layer 1 - Fast semantic analysis (free)
python src/layer1_checker.py

# Step 3: Layer 2 - LLM validation for uncertain cases
python src/layer2_validator.py
```

---

## 📊 What It Does

**Layer 1** (Semantic Analysis)
- Analyzes all conversations using sentence embeddings
- Auto-decides clear cases (~80%)
- Sends uncertain cases to Layer 2
- Cost: $0 (runs locally)

**Layer 2** (Dual LLM Validation)
- Two AI judges evaluate ambiguous conversations
- Judge 1: Strict rule compliance
- Judge 2: Empathetic + compliant
- Both must agree to auto-validate
- Cost: ~$0.0013 per conversation

**Result:** ~92% automated, ~8% need human review

---

## 📁 Key Outputs

**For Decisions:**
- `data/layer1_output/auto_decided.json` - High-confidence results
- `data/layer2_output/validated_decisions.json` - LLM-validated results

**For Human Review:**
- `data/layer1_output/layer1_results.xlsx` - All Layer 1 results
- `data/layer2_output/human_review_needed.xlsx` - Cases where judges disagreed
- `data/layer2_output/layer2_all_results.xlsx` - Complete Layer 2 results

**Tracking:**
- `data/layer2_output/cost_report.json` - Token usage and costs

---

## 💰 Costs

| Component | Cost |
|-----------|------|
| Generate 200 conversations | $0.08 |
| Layer 1 (200 conversations) | $0.00 |
| Layer 2 (~40 conversations) | $0.05 |
| **Total** | **$0.13** |

**Production scale (5,000/day):** ~$39/month

---

## ⚙️ Configuration

**Adjust Layer 1 thresholds:** Edit `config/confidence_thresholds.json`
```json
{
  "confidence_thresholds": {
    "compliant": 0.70,
    "medium": 0.75,
    "high": 0.80,
    "critical": 0.85
  }
}
```

**Change LLM models:** Edit `config/layer2_llm_config.json`
```json
{
  "judge1": {"model": "gpt-4o-mini", ...},
  "judge2": {"model": "claude-3-haiku-20240307", ...}
}
```

---

## 📖 Documentation

- **[APPROACH.md](docs/APPROACH.md)** - Design decisions and tradeoffs
- **[AI_USAGE.md](AI_USAGE.md)** - Models used, costs, scaling strategy
- **[docs/api/](docs/api/)** - Data contract schemas

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