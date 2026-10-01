"""A small JSON journal of publish attempts.

Deliberately advisory: correctness does not depend on it. A finished file that
publishes successfully is deleted from staging, so it can never be published
twice; a file that fails stays in staging and is retried on the next sweep with
no stored state required.

The journal exists for two things the sweeper cannot work out from the
filesystem alone: how many times a file has already failed (so the log does not
fill with the same error every 20 seconds), and which session logs have been
published — logs are never deleted locally, so there is nothing else to tell.

Losing or deleting this file is harmless: attempt counters reset, and a
re-published log is recognised by hash and skipped.
"""

import json
import logging
import os
import threading

logger = logging.getLogger(__name__)

_lock = threading.Lock()


class Journal:
    def __init__(self, path: str):
        self.path = path
        self._data = {"failures": {}, "logs": {}}
        self.load()

    # -- persistence ----------------------------------------------------
    def load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                self._data["failures"] = dict(data.get("failures", {}))
                self._data["logs"] = dict(data.get("logs", {}))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(self._data, handle, indent=2)
            os.replace(tmp, self.path)      # local file — replace is allowed here
        except OSError as exc:
            logger.debug(f"Could not write the store journal: {exc}")

    # -- failures -------------------------------------------------------
    def record_failure(self, path: str, message: str) -> int:
        entry = self._data["failures"].setdefault(path, {"attempts": 0, "error": ""})
        entry["attempts"] += 1
        entry["error"] = message
        return entry["attempts"]

    def clear_failure(self, path: str) -> None:
        self._data["failures"].pop(path, None)

    def attempts(self, path: str) -> int:
        return self._data["failures"].get(path, {}).get("attempts", 0)

    def last_error(self, path: str) -> str:
        return self._data["failures"].get(path, {}).get("error", "")

    def failures(self) -> dict:
        return dict(self._data["failures"])

    # -- session logs ---------------------------------------------------
    def log_published(self, path: str) -> bool:
        return path in self._data["logs"]

    def mark_log_published(self, path: str, target: str) -> None:
        self._data["logs"][path] = target


def open_journal(staging_root: str) -> Journal:
    with _lock:
        return Journal(os.path.join(staging_root, "_publish_journal.json"))
