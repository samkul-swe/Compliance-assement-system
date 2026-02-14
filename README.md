# MLE Assessment: Compliance for Customer Communication

A self-contained take-home assessment for Machine Learning Engineer candidates at a consumer finance startup. This repo contains the problem brief, mock data, data contracts, and expected deliverables.

## Quick start

1. **Read the brief**: [ASSESSMENT.md](ASSESSMENT.md) — problem statement, tasks, and deliverables.
2. **Explore data**: `data/` — sample conversations and compliance rules (see `docs/api/` for contracts).
3. **Implement**: Use `src/` (or `notebooks/`) for code, `prompts/` for any LLM prompts, `docs/` for presentation/notes.
4. **Track AI usage**: Fill [AI_USAGE.md](AI_USAGE.md) with models, token counts, cost, and scaling commentary.

### Run the reference implementation (optional)

The repo includes a minimal reference script so you can run and see results without building backends.

```bash
# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run compliance check + situation classifier on sample data
python src/run_assessment.py
```

**Environment variables** (optional):

- `OPENAI_API_KEY` or similar — only if you choose to use an LLM API in your solution; the reference script works without it.

## Repo structure

| Path | Purpose |
|------|---------|
| `ASSESSMENT.md` | Problem statement, deliverables, AI usage expectations |
| `README.md` | This file — how to run and navigate the repo |
| `AI_USAGE.md` | Template for models, tools, token/cost, scaling notes (you fill this) |
| `data/` | Sample conversations (JSON/JSONL), compliance rules, schemas |
| `docs/` | Your presentation, notes, summary, thinking process |
| `docs/api/` | Data contracts and mock API descriptions |
| `prompts/` | Prompts used for LLM-based checks or classifiers (you add) |
| `src/` | Code — reference implementation + your code |
| `RUBRIC.md` | Evaluator rubric (for hiring team; do not share with candidates) |

## Data contracts

- **Conversations**: See `data/conversations.json` and `docs/api/conversation_schema.json`. Each conversation has `conversation_id`, `messages` (role, text, timestamp), and optional metadata.
- **Compliance rules**: See `data/compliance_rules.json` and `docs/api/compliance_rules_schema.json`. Rules have id, category, description, severity, and optional keyword/regex hints.
- **Customer situation** (optional): Document your input/output contract in `docs/` if you implement a situation classifier (e.g. product loss vs substandard service).

APIs can be mocked (e.g. read from `data/`); we care that you design to stable contracts.

## Deliverables checklist

- [ ] Runnable code with clear entrypoint and setup instructions
- [ ] Short presentation/notes/summary (in `docs/`)
- [ ] Prompts and thinking process (in `prompts/` or `docs/`)
- [ ] `AI_USAGE.md` filled: models, tools, token/cost estimate, scaling-to-prod commentary

Submit a single self-contained repo (or zip) with everything needed to run and evaluate your work.
