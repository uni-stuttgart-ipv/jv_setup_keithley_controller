# AGENTS.md

**The project guidance for every coding agent lives in [`CLAUDE.md`](CLAUDE.md).
Read that file, not this one.**

This file used to hold a second copy of those instructions and drifted out of
date — it still described the quintic MPP fit and green controls long after both
were replaced, which is worse than having no guidance at all. It is now a
pointer so there is exactly one set of rules to keep current.

Start here:

- [`CLAUDE.md`](CLAUDE.md) — hard rules, commands, and which check to run for
  the kind of change you're making
- [`docs/architecture.md`](docs/architecture.md) — structure, patterns, where things live
- [`docs/metrics.md`](docs/metrics.md) — metric contracts (read before touching `analysis/` or `spo/`)
- [`docs/ui.md`](docs/ui.md) — theme, Qt/QSS traps, UI verification
- [`docs/testing.md`](docs/testing.md) — fixtures, traps, debugging playbook
- [`AUDIT.md`](AUDIT.md) — known open defects; grep it for your area
