# Prompts

This directory contains prompts used for LLM-based components.

## Files

- **conversation_generation.md** - Generates realistic synthetic conversations
  - Used by: `generate_conversations.py`
  - Model: GPT-4o-mini
  - Purpose: Create training/test data with compliant, non-compliant, and vague cases

- **compliance_validation.md** - Validates compliance in ambiguous cases
  - Used by: `src/layer2_validator.py`
  - Models: GPT-4o-mini (Judge 1 & Judge 2)
  - Purpose: Expert judgment on borderline conversations

## Prompt Structure

All prompts follow the same structure:
- **__ASK__** - What we want the LLM to do
- **__CONTEXT__** - Background and why it matters
- **__CONSTRAINTS__** - Rules and requirements
- **__EXAMPLE__** - Concrete examples of expected output

This structure ensures consistent, high-quality responses.