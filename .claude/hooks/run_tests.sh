#!/bin/bash
# PostToolUse hook: run the tests that cover the file just edited.
#
# Design goals, in order:
#   1. catch a regression immediately, while the edit is still in context;
#   2. cost nothing when it passes — SILENT on success, so a multi-file change
#      doesn't pipe the same "N passed" banner back N times;
#   3. be fast — the matching subset, not the whole suite. The agent runs the
#      whole suite itself when the change spans areas or before handing over
#      (see the table in CLAUDE.md).
#
# Reads the tool payload on stdin. Docs/config-only edits exit immediately.
payload=$(cat)
file=$(printf '%s' "$payload" | python3 -c \
  "import sys,json;d=json.load(sys.stdin);print(d.get('tool_input',{}).get('file_path',''))" \
  2>/dev/null)

cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}" || exit 0

# --- route the edited path to the tests that cover it ----------------------
case "$file" in
  *src/solarjv_analyzer/analysis/*|*src/solarjv_analyzer/spo/spo_analysis.py)
      targets="tests/analysis tests/spo" ;;
  *src/solarjv_analyzer/gui/*|*src/solarjv_analyzer/windows/*|*src/solarjv_analyzer/spo/spo_widget.py)
      targets="tests/gui" ;;
  *src/solarjv_analyzer/procedures/*|*src/solarjv_analyzer/instruments/*)
      targets="tests/procedures tests/instruments" ;;
  *src/solarjv_analyzer/spo/*)
      targets="tests/spo" ;;
  *src/solarjv_analyzer/auth/*)
      targets="tests/auth tests/test_session.py" ;;
  *src/solarjv_analyzer/store/*)
      targets="tests/store" ;;
  *src/solarjv_analyzer/utils/*|*src/solarjv_analyzer/config.py)
      targets="tests/utils tests/config tests/store" ;;
  *src/solarjv_analyzer/main.py)
      targets="tests/gui" ;;
  *tools/*)
      targets="tests/tools" ;;
  *tests/*)
      # An edited test file: run that file alone.
      targets="$file" ;;
  *)
      exit 0 ;;   # docs, settings, anything else — nothing to verify
esac

# Prefer the project venv, so the hook works regardless of the global python.
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python

out=$(QT_QPA_PLATFORM=offscreen "$PY" -m pytest $targets -q -x 2>&1)
status=$?

# No pytest in this interpreter is a setup problem, not a regression — say it
# once, quietly, instead of reporting a fake test failure.
if printf '%s' "$out" | grep -q "No module named pytest"; then
  echo "hook: pytest not available to $PY — tests not run (pip install -e . ?)"
  exit 0
fi

if [ $status -ne 0 ]; then
  # Failure: show enough to act on, and name the follow-up check.
  printf '%s\n' "$out" | tail -30
  echo "--- hook: $targets FAILED (exit $status) ---"
  case "$file" in
    *src/solarjv_analyzer/gui/*|*src/solarjv_analyzer/windows/*|*spo_widget.py)
      echo "hook: after fixing, also run QT_QPA_PLATFORM=offscreen python3 tools/render_preview.py" ;;
  esac
fi
# Success: print nothing.
exit 0
