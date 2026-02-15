# AI Usage Log (Candidate — fill this out)

Use this template to report how you used AI for the assessment. Keep cost minimal; we care about reasoning and design.

---

## Models and tools used

| Model / tool | Purpose (e.g. code gen, prompts, analysis) |
|--------------|--------------------------------------------|
| _e.g. Claude 3.5 Sonnet via Cursor_ | _e.g. Draft compliance rule logic_ |
| _e.g. GPT-4o mini_ | _e.g. Classify sample conversations_ |
| _…_ | _…_ |

---

## Workflows / skills

- _e.g. “Used Cursor agent to generate initial compliance checker; then edited by hand.”_
- _e.g. “Custom prompt in `prompts/compliance_check.txt` run via OpenAI API.”_

---

## Token usage and cost (approximate)

| Model | Input tokens (approx) | Output tokens (approx) | Est. cost ($) |
|-------|------------------------|-------------------------|---------------|
| _e.g. Claude 3.5 Sonnet_ | _e.g. 15k_ | _e.g. 2k_ | _e.g. 0.08_ |
| _…_ | _…_ | _…_ | _…_ |
| **Total** | | | **_e.g. 0.12_** |

---

## Scaling to production — commentary

_Short paragraph: how would you manage cost, rate limits, caching, model choice, and reliability when this runs at scale (e.g. thousands of conversations per day)?_

Example topics:

- Caching (e.g. same conversation text → reuse compliance result).
- Model choice (smaller/cheaper model for simple rules, larger only when needed).
- Batching and async processing.
- Guardrails and fallbacks (e.g. rule-based first, LLM only for edge cases).



Estimated cost for 200 conversations:
   ~160,000 tokens
   ~$0.0528 USD