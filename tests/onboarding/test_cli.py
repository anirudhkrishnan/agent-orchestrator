"""End-to-end CLI tests for `python -m orchestrator.onboarding init`."""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.onboarding.cli import main


# Re-use a slim template fixture matching the real one's load-bearing markers.
_TEMPLATE_FIXTURE = """\
# Orchestration Plan — <APP_NAME>

**Status:** draft
**Last bake-off:** <ISO_TIMESTAMP>
**Routing.json slots used:** <COMMA_SEPARATED_LIST>
**Owner:** <NAME>
**Onboarded by:** <NAME> on <DATE>

## 1. Task inventory

| Task | Input shape | Output shape | Frequency (per app run) | Latency tolerance | Notes |
|---|---|---|---|---|---|
| <task_name> | <input_shape> | <output_shape> | <freq> | <tolerance> | <notes> |
"""


def _make_template(tmp_path: Path) -> Path:
    t = tmp_path / "template.md"
    t.write_text(_TEMPLATE_FIXTURE)
    return t


def _make_workspace(tmp_path: Path, *, with_llm_call: bool = True) -> Path:
    ws = tmp_path / "my-app"
    ws.mkdir()
    skill_body = "# SKILL\n"
    if with_llm_call:
        skill_body += "\nWhen triggered, invoke the model to summarize.\n"
    (ws / "SKILL.md").write_text(skill_body)
    return ws


# --- help / no-args --------------------------------------------------------


def test_cli_no_args_prints_help_returns_zero(capsys):
    rc = main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "init" in out


def test_cli_help_flag(capsys):
    try:
        main(["--help"])
    except SystemExit as e:
        assert e.code == 0
    out = capsys.readouterr().out
    assert "init" in out


# --- init success path -----------------------------------------------------


def test_cli_init_writes_plan(tmp_path: Path, capsys):
    ws = _make_workspace(tmp_path)
    template = _make_template(tmp_path)
    rc = main([
        "init",
        "--app", "my-app",
        "--workspace", str(ws),
        "--template", str(template),
    ])
    assert rc == 0
    plan = ws / "orchestration-plan.md"
    assert plan.exists()
    body = plan.read_text()
    assert "# Orchestration Plan — my-app" in body
    out = capsys.readouterr().out
    assert "scaffold written" in out
    assert "Next steps" in out


def test_cli_init_seeds_inventory_from_scan(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    template = _make_template(tmp_path)
    rc = main([
        "init", "--app", "my-app",
        "--workspace", str(ws), "--template", str(template),
    ])
    assert rc == 0
    body = (ws / "orchestration-plan.md").read_text()
    # The scanner picked up "invoke the model" — the inventory table has
    # at least one task_NN_ row and no placeholder.
    assert "task_01" in body
    assert "<task_name> | <input_shape>" not in body


def test_cli_init_with_no_scan_keeps_placeholder(tmp_path: Path, capsys):
    ws = _make_workspace(tmp_path, with_llm_call=False)
    template = _make_template(tmp_path)
    rc = main([
        "init", "--app", "my-app",
        "--workspace", str(ws), "--template", str(template),
        "--no-scan",
    ])
    assert rc == 0
    body = (ws / "orchestration-plan.md").read_text()
    # Without scanning, we keep the TODO placeholder row.
    assert "TODO" in body
    out = capsys.readouterr().out
    assert "no llm-call patterns" in out.lower()


def test_cli_init_writes_to_custom_out_path(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    template = _make_template(tmp_path)
    out = tmp_path / "custom" / "elsewhere.md"
    out.parent.mkdir()
    rc = main([
        "init", "--app", "my-app",
        "--workspace", str(ws),
        "--template", str(template),
        "--out", str(out),
    ])
    assert rc == 0
    assert out.exists()
    # And the default location is NOT written.
    assert not (ws / "orchestration-plan.md").exists()


def test_cli_init_records_custom_onboarded_by(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    template = _make_template(tmp_path)
    rc = main([
        "init", "--app", "my-app",
        "--workspace", str(ws),
        "--template", str(template),
        "--onboarded-by", "example_user",
    ])
    assert rc == 0
    body = (ws / "orchestration-plan.md").read_text()
    assert "Onboarded by:** example_user" in body


# --- default (packaged) template -------------------------------------------


def test_cli_init_succeeds_without_template_flag(tmp_path: Path, capsys):
    """Regression: `init` must work with NO --template. The old default walked
    parent directories above the installed package and never resolved; the
    template now ships inside the package and is found via importlib.resources."""
    ws = _make_workspace(tmp_path)
    rc = main([
        "init",
        "--app", "my-app",
        "--workspace", str(ws),
    ])
    assert rc == 0
    body = (ws / "orchestration-plan.md").read_text()
    assert "# Orchestration Plan — my-app" in body
    assert "<APP_NAME>" not in body
    # The scanner row replaced the packaged template's placeholder row.
    assert "task_01" in body
    out = capsys.readouterr().out
    assert "Next steps" in out


def test_default_template_is_packaged_and_has_load_bearing_markers():
    """The packaged template must exist and carry every marker the scaffold
    substitutes or replaces."""
    from orchestrator.onboarding.cli import _default_template_path

    p = _default_template_path()
    assert p.exists(), f"packaged template missing: {p}"
    text = p.read_text()
    assert "# Orchestration Plan — <APP_NAME>" in text
    assert "**Last bake-off:** <ISO_TIMESTAMP>" in text
    assert "**Onboarded by:** <NAME> on <DATE>" in text
    # The exact inventory placeholder row _replace_inventory_table looks for.
    assert (
        "| <task_name> | <input_shape> | <output_shape> | <freq> | "
        "<tolerance> | <notes> |"
    ) in text


# --- init failure paths ---------------------------------------------------


def test_cli_init_refuses_existing_plan_without_overwrite(tmp_path: Path, capsys):
    ws = _make_workspace(tmp_path)
    template = _make_template(tmp_path)
    (ws / "orchestration-plan.md").write_text("# existing\n")
    rc = main([
        "init", "--app", "my-app",
        "--workspace", str(ws),
        "--template", str(template),
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "already exists" in err
    # File still has original content.
    assert (ws / "orchestration-plan.md").read_text() == "# existing\n"


def test_cli_init_overwrite_replaces_existing(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    template = _make_template(tmp_path)
    (ws / "orchestration-plan.md").write_text("# existing\n")
    rc = main([
        "init", "--app", "my-app",
        "--workspace", str(ws),
        "--template", str(template),
        "--overwrite",
    ])
    assert rc == 0
    body = (ws / "orchestration-plan.md").read_text()
    assert "# Orchestration Plan — my-app" in body


def test_cli_init_returns_2_when_workspace_missing(tmp_path: Path, capsys):
    template = _make_template(tmp_path)
    rc = main([
        "init", "--app", "x",
        "--workspace", str(tmp_path / "no-such-dir"),
        "--template", str(template),
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "workspace does not exist" in err


def test_cli_init_returns_2_when_workspace_not_dir(tmp_path: Path, capsys):
    template = _make_template(tmp_path)
    f = tmp_path / "file.txt"
    f.write_text("x")
    rc = main([
        "init", "--app", "x",
        "--workspace", str(f),
        "--template", str(template),
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not a directory" in err


def test_cli_init_returns_3_when_template_missing(tmp_path: Path, capsys):
    ws = _make_workspace(tmp_path)
    rc = main([
        "init", "--app", "x",
        "--workspace", str(ws),
        "--template", str(tmp_path / "no-template.md"),
    ])
    assert rc == 3
    err = capsys.readouterr().err
    assert "template not found" in err


# --- argparse plumbing ----------------------------------------------------


def test_cli_init_required_args(capsys):
    """`init` without --app and --workspace should fail at argparse layer."""
    with pytest.raises(SystemExit):
        main(["init"])
    err = capsys.readouterr().err
    assert "--app" in err or "required" in err.lower()
