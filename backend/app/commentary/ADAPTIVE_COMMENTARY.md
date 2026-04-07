# Adaptive Commentary

This module extends the existing tactical commentary pipeline with a small, deterministic audience-adaptation layer.

## What It Adds

- audience profiles for `Beginner`, `Intermediate`, and `Expert`
- explicit level selection plus rule-based fallback inference
- minimal personalization hooks:
  - `commentary_level`
  - `commentary_verbosity`
  - `educational_mode`
  - `commentary_style`
- deterministic fallback commentary that changes systematically by level
- prompt-level adaptation when the Ollama commentary model is available
- comparison utilities for side-by-side adaptive examples

## Main Files

- `adaptive.py`
  audience profile resolution, adaptation policy, deterministic fallback rendering, and lightweight evaluation helpers
- `tac_commentary.py`
  integrates adaptive guidance into tactical commentary generation
- `commentary_service.py`
  resolves audience profile from persisted match metadata and app settings
- `routes/upload.py`
  accepts optional adaptive commentary preferences on upload
- `scripts/generate_adaptive_commentary_examples.py`
  generates Beginner / Intermediate / Expert examples for the same tactical situation

## Profile Fields

These values can be stored in `match.tracking_artifacts`:

- `commentary_level`: `Beginner`, `Intermediate`, or `Expert`
- `commentary_verbosity`: `low`, `medium`, or `high`
- `educational_mode`: `true` or `false`
- `commentary_style`: `neutral`, `friendly`, `analytical`, or `coach`

If no valid explicit level is present, the resolver falls back to simple rules using the other signals. If no useful signals are available, it defaults to `Intermediate`.

## Example Demo

Run from the `backend` directory:

```bash
python scripts/generate_adaptive_commentary_examples.py
```

Outputs:

- `artifacts/adaptive_commentary_examples/comparison.json`
- `artifacts/adaptive_commentary_examples/validation.json`
- `artifacts/adaptive_commentary_examples/comparison.md`

## Upload API

The upload route now accepts these optional form fields in addition to `commentary_level`:

- `commentary_verbosity`
- `educational_mode`
- `commentary_style`

Existing clients that only send `commentary_level` remain compatible.

## Future Extension Points

- replace the rule-based audience inference with a learned classifier
- connect profile resolution to real user accounts or session history
- add stronger text-quality evaluation metrics
- expand style control beyond a single tone field
