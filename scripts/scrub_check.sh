#!/usr/bin/env bash
# scrub_check.sh — fail if tracked files contain personal/private leakage.
#
# Two layers:
#   1. Generic patterns any public checkout can verify: absolute home paths
#      ('/Users/'), personal email domains ('@gmail'), local workspace paths
#      ('Documents/workspace'). GitHub commit no-reply addresses
#      ('users.noreply') are explicitly allowed.
#   2. Optional maintainer-local denylist: a `.scrub-denylist` file at the repo
#      root (gitignored — never committed), one case-insensitive term per line.
#      Lets a maintainer scan for terms that must never ship without publishing
#      the terms themselves. Absent in CI, where only layer 1 runs.
#
# Scans TRACKED files only (git ls-files) — untracked local artifacts are not
# shipped. Exit 0 = clean, exit 1 = at least one hit.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

# NUL-safe tracked-file list; exclude the denylist itself defensively (it is
# gitignored, so it should never be tracked in the first place) and this
# script (its own pattern definitions would otherwise flag themselves).
files=()
while IFS= read -r -d '' f; do
  [ "$f" = ".scrub-denylist" ] && continue
  [ "$f" = "scripts/scrub_check.sh" ] && continue
  files+=("$f")
done < <(git ls-files -z)

status=0

# --- Layer 1: generic leak patterns (fixed strings, case-sensitive) ---------
generic_patterns=(
  '/Users/'
  '@gmail'
  'Documents/workspace'
)

for pat in "${generic_patterns[@]}"; do
  hits="$(grep -nIF -- "$pat" "${files[@]}" 2>/dev/null | grep -v 'users\.noreply' || true)"
  if [ -n "$hits" ]; then
    echo "[scrub] generic pattern '$pat' found in tracked files:"
    printf '%s\n' "$hits" | sed 's/^/  /'
    status=1
  fi
done

# --- Layer 2: maintainer-local denylist (fixed strings, case-insensitive) ---
if [ -f .scrub-denylist ]; then
  while IFS= read -r term || [ -n "$term" ]; do
    term="${term%$'\r'}"
    case "$term" in '' | '#'*) continue ;; esac
    hits="$(grep -inIF -- "$term" "${files[@]}" 2>/dev/null || true)"
    if [ -n "$hits" ]; then
      echo "[scrub] denylist term '$term' found in tracked files:"
      printf '%s\n' "$hits" | sed 's/^/  /'
      status=1
    fi
  done <.scrub-denylist
else
  echo "[scrub] no .scrub-denylist at repo root — running generic layer only."
fi

if [ "$status" -eq 0 ]; then
  echo "[scrub] clean — no leak patterns in tracked files."
else
  echo "[scrub] FAILED — scrub the hits above before publishing." >&2
fi
exit "$status"
