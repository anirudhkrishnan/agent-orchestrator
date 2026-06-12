# Example Run Data

> **ILLUSTRATIVE DATA ONLY** — these are synthetic judge scores, not results
> from a real bake-off. Numbers are hand-crafted to demonstrate the tooling.
> Do not draw conclusions about real model quality from this data.

## Structure

```
examples/
  example-run-oss/
    judge-batch.json   # JudgeBatch shape — 6 tasks × 3 scenarios × 5 samples (N=5)
    judge-scores.json  # Synthetic judge scores for ollama/qwen3:8b
  example-run-frontier/
    judge-scores.json  # Synthetic scores for claude-sonnet-4-6 + claude-haiku-4-5
```

## Running the CLI against this data

```bash
# Dry-run comparison for ingest_pipeline
PYTHONPATH=src python -m orchestrator.tiered dry-run \
    examples/example-run-oss examples/example-run-frontier \
    --workflow ingest_pipeline

# Dry-run for all example workflows
PYTHONPATH=src python -m orchestrator.tiered dry-run \
    examples/example-run-oss examples/example-run-frontier

# Build routing-tiered.json
PYTHONPATH=src python -m orchestrator.tiered build-table \
    examples/example-run-oss examples/example-run-frontier \
    --out /tmp/routing-tiered.json
```

## Replacing with real data

1. Run your own OSS bake-off with `python -m orchestrator.evaluation run ...`
2. Have the judge score outputs → `judge-batch.json` + `judge-scores.json`
3. Do the same for frontier models
4. Point the CLI at your run directories instead of `examples/`
