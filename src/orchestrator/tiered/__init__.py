"""Tiered routing module.

Provides three-mode (tiered / frontier-only / OSS-only) dry-run comparison and
machine-readable routing-table generation for any agent workflow.

Quickstart::

    from orchestrator.tiered.dry_run import analyze_workflow, render_dry_run_report
    from orchestrator.tiered.dry_run import load_oss_quality, load_frontier_quality
    from orchestrator.tiered.routing_table import build_routing_table

Public surface:

* :mod:`orchestrator.tiered.dry_run` — per-workflow cost/quality comparison.
* :mod:`orchestrator.tiered.routing_table` — machine-readable routing-tiered.json.
* :mod:`orchestrator.tiered.cli` — ``dry-run`` and ``build-table`` CLI commands.

CLI entry-point::

    python -m orchestrator.tiered dry-run \\
        examples/example-run-oss examples/example-run-frontier \\
        --workflow ingest_pipeline

    python -m orchestrator.tiered build-table \\
        examples/example-run-oss examples/example-run-frontier \\
        --out routing-tiered.json
"""
