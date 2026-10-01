#!/usr/bin/env python3
"""
vision_inspector.py — general-purpose vision QA bridge.

Sends an image + a prompt to a Gemini vision model and prints the text
response. Meant to be called from a bash tool / CI hook: everything that
matters comes back as stdout, so the calling agent never needs vision of
its own.

Two modes:
  - Raw:      `vision_inspector.py IMAGE "your prompt"` (unstructured, as before)
  - Scenario: `vision_inspector.py IMAGE --scenario style-check [--verify]`
              builds a grounding-first, evidence-forcing prompt from
              tools/prompt_builder.py (which injects the live design tokens),
              requests structured JSON, and prints a human summary + JSON.
              `--verify` adds a second adversarial pass that tries to refute
              the first pass's findings. `--reference REF` supplies a second
              image for the match-reference scenario.

AUTO-UPDATING, without Google's "-latest" alias:
Google's own docs say the "-latest" alias points at an experimental,
rate-limited channel and isn't recommended for anything you depend on
repeatedly (https://ai.google.dev/gemini-api/docs/models). Instead, this
queries the live model list, filters to stable Flash-tier models that
support generateContent, picks the newest one (by *semantic* version, not
float — 3.10 > 3.9), and caches that choice locally so the model doesn't
silently change mid-session — only when the cache expires (default 7 days)
or a call 404s.

Install:
    pip install -q -U google-genai

Auth (checked in this order):
    export GEMINI_API_KEY=...
    export GOOGLE_API_KEY=...

Scope: this uses the Gemini Developer API (api_key=...), not Vertex AI /
Gemini Enterprise. The image is sent as inline base64 bytes in the
Interactions API `input` list (``{"type": "image", "data": <base64>,
"mime_type": ...}``) — that path works on BOTH the Developer API and
Vertex, unlike client.files.upload(), which is Developer-API-only, so we
don't depend on which billing backend the key came from. The legacy
``client.models.generate_content`` endpoint is being retired; this tool
targets ``client.interactions.create``.

The `google.genai` import is deliberately lazy (inside the functions that
need it) so the module's pure model-selection helpers can be unit-tested
and the CLI can report a clear "install google-genai" error without a
traceback if the SDK isn't present.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Sibling module for scenario-driven prompts. Pure (no google-genai import),
# and the theme tokens are imported lazily inside it, so this never pulls the
# app package or the SDK in at import time.
import prompt_builder  # noqa: E402

# Hard fallback if live discovery fails entirely (network down, bad key,
# API shape changed). This is a safety net, not the primary mechanism —
# update it occasionally, but the script works even if you never touch it.
PINNED_FALLBACK_MODEL = "gemini-3.6-flash"

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = Path(
    os.environ.get("VISION_MODEL_CACHE", REPO_ROOT / ".cache" / "gemini_vision_model.json")
)
CACHE_TTL_SECONDS = int(os.environ.get("VISION_MODEL_CACHE_TTL", 7 * 24 * 3600))  # 7 days

# Matches stable, non-preview/experimental Flash chat models, e.g.
#   models/gemini-3.0-flash, models/gemini-2.5-flash
# The trailing `$` anchors the "-flash" suffix, so "-flash-latest" (the
# alias we must NOT depend on) and "-flash-thinking" etc. never match.
# Excludes lite/image/tts/embedding/thinking/live/exp/preview/robotics
# variants — not what you want for "describe this screenshot" QA.
_STABLE_FLASH_RE = re.compile(r"^models/gemini-(\d+(?:\.\d+)?)-flash$")
_EXCLUDE_RE = re.compile(r"(lite|image|tts|embed|thinking|live|exp|preview|robotics)")

_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("Error: set GEMINI_API_KEY or GOOGLE_API_KEY.", file=sys.stderr)
        sys.exit(1)
    return key


def _flash_version_tuple(name: str) -> tuple[int, ...] | None:
    """Return the semantic version of a `models/gemini-X.Y-flash` name, or
    None if it isn't a stable Flash model. Tuple comparison makes
    `3.10` sort after `3.9` (a float comparison would get this wrong)."""
    match = _STABLE_FLASH_RE.match(name)
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def _select_newest_stable_flash(names) -> str | None:
    """Pure selection: given model names, return the newest stable Flash
    model's short name (no `models/` prefix), or None. Separated from the
    network call so it can be unit-tested without an API key."""
    best: tuple[tuple[int, ...], str] | None = None
    for name in names:
        if _EXCLUDE_RE.search(name):
            continue
        version = _flash_version_tuple(name)
        if version is None:
            continue
        short_name = name.removeprefix("models/")
        if best is None or version > best[0]:
            best = (version, short_name)
    return best[1] if best else None


def _discover_latest_stable_flash(client) -> str | None:
    """Query the live model list and return the newest stable Flash model
    name, or None if discovery fails or nothing matches. The Interactions
    API shares model names with generateContent, so filtering on
    ``generateContent`` in ``supported_actions`` still selects the models
    ``interactions.create`` accepts."""
    try:
        names = []
        for m in client.models.list():
            name = getattr(m, "name", "") or ""
            actions = getattr(m, "supported_actions", None) or []
            if "generateContent" not in actions:
                continue
            names.append(name)
    except Exception as e:
        print(f"Warning: model discovery failed ({e}); will fall back.", file=sys.stderr)
        return None
    return _select_newest_stable_flash(names)


def _load_cache() -> dict | None:
    if not CACHE_PATH.exists():
        return None
    try:
        data = json.loads(CACHE_PATH.read_text())
        if time.time() - data.get("resolved_at", 0) < CACHE_TTL_SECONDS:
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _save_cache(model_name: str) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(
            json.dumps({"model": model_name, "resolved_at": time.time()})
        )
    except OSError as e:
        print(f"Warning: couldn't write model cache ({e}).", file=sys.stderr)


def _invalidate_cache() -> None:
    try:
        CACHE_PATH.unlink(missing_ok=True)
    except OSError:
        pass


def resolve_model(client, force_refresh: bool = False) -> str:
    """Return the model name to use. Refreshes from live discovery at most
    once per CACHE_TTL_SECONDS, so a session's before/after comparisons
    stay on the same model instead of drifting mid-run.

    The chosen model + source is printed to stderr on EVERY call — model
    selection is auditable, never silent."""
    source = "cache"
    if not force_refresh:
        cached = _load_cache()
        if cached:
            model = cached["model"]
            print(f"[vision_inspector] using model: {model} ({source})", file=sys.stderr)
            return model

    discovered = _discover_latest_stable_flash(client)
    model = discovered or PINNED_FALLBACK_MODEL
    source = "discovered" if discovered else "pinned fallback"
    if discovered:
        _save_cache(model)
    print(f"[vision_inspector] using model: {model} ({source})", file=sys.stderr)
    return model


def _response_text(response) -> str:
    """Return the Interaction's output text (interactions API)."""
    text = getattr(response, "output_text", None)
    return str(text).strip() if text else ""


def _mime_type_for(image_path: Path) -> str:
    return _MIME_BY_SUFFIX.get(image_path.suffix.lower(), "image/png")


def analyze_image(
    image_path: str,
    prompt: str,
    model: str | None = None,
    refresh_model: bool = False,
    json_mode: bool = False,
    additional_images: list[str] | None = None,
) -> str:
    # Lazy import: the SDK is only needed when actually calling the API, so
    # `--help`, cache logic, and unit tests work without it installed.
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"No image at {image_path}")

    for extra in additional_images or []:
        if not Path(extra).exists():
            raise FileNotFoundError(f"No image at {extra}")

    try:
        from google import genai
    except ImportError:
        print(
            "Error: google-genai is not installed. Run `pip install -U google-genai`.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = genai.Client(api_key=_get_api_key())

    if model:
        print(f"[vision_inspector] using model: {model} (explicit --model)", file=sys.stderr)
    resolved = model or resolve_model(client, force_refresh=refresh_model)

    # Inline base64 bytes — works on the Developer API and Vertex, and
    # avoids leaving a 48h-orphaned file on the Files service. The prompt
    # goes first, then the candidate image, then any reference images.
    input_parts = _build_input(prompt, [path] + [Path(e) for e in (additional_images or [])])

    response = _generate(client, resolved, input_parts, json_mode=json_mode)
    text = _response_text(response)
    if not text:
        raise RuntimeError(
            "model returned no text (empty or blocked response); "
            "check the image and prompt"
        )
    return text


def _build_input(prompt: str, image_paths: list[Path]) -> list[dict]:
    """Build the interactions-API `input` list: the prompt text followed by
    one inline-base64 image dict per image (candidate first, then any
    reference images). This is the wire shape the Interactions API accepts
    (``{"type": "image", "data": <base64>, "mime_type": ...}``)."""
    import base64

    parts: list[dict] = [{"type": "text", "text": prompt}]
    for p in image_paths:
        parts.append(
            {
                "type": "image",
                "data": base64.b64encode(p.read_bytes()).decode("ascii"),
                "mime_type": _mime_type_for(p),
            }
        )
    return parts


def _generate(client, model: str, input_parts: list[dict], json_mode: bool = False):
    """Generate, re-resolving the model once if it 404s (cached model may
    have been deprecated since last refresh)."""
    kwargs: dict = {"model": model, "input": input_parts}
    if json_mode:
        # Force structured JSON so a "claim" is a parseable field, not prose.
        # The legacy `response_mime_type` field is deprecated; JSON is
        # requested via the `response_format` text-format descriptor instead.
        kwargs["response_format"] = {"type": "text", "mime_type": "application/json"}

    try:
        return client.interactions.create(**kwargs)
    except Exception as e:  # noqa: BLE001 — deprecation errors span several SDK exception types
        status = (
            getattr(e, "status", None)
            or getattr(e, "status_code", None)
            or getattr(e, "code", None)
        )
        msg = str(e)
        is_gone = (
            status == 404
            or "no longer available" in msg
            or "NOT_FOUND" in msg
            or "not found" in msg.lower()
        )
        if not is_gone:
            raise
        # Cached (or pinned) model is gone — force one live re-resolve and
        # retry before giving up.
        print(
            f"[vision_inspector] model {model} no longer available, re-resolving...",
            file=sys.stderr,
        )
        _invalidate_cache()
        fresh = resolve_model(client, force_refresh=True)
        kwargs["model"] = fresh
        return client.interactions.create(**kwargs)


def _parse_json(text: str) -> dict:
    """Robustly extract a JSON object from a model response (strips markdown
    fences and surrounding prose). Raises ValueError if no object is found."""
    t = text.strip()
    if t.startswith("```"):
        t = t.lstrip("`")
        if t[:4].lower() == "json":
            t = t[4:].lstrip()
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("model did not return a JSON object")
    try:
        return json.loads(t[start : end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"could not parse model JSON: {e}")


def _render_summary(parsed: dict) -> str:
    """Human-readable digest of the structured verdicts/diffs."""
    lines = []
    summary = parsed.get("summary")
    if summary:
        lines.append(str(summary))

    verdicts = parsed.get("verdicts", [])
    diffs = parsed.get("diffs", [])
    retracted = parsed.get("retracted", [])

    if verdicts:
        fails = [v for v in verdicts if not v.get("pass", True)]
        lines.append(f"{len(fails)} violation(s) of {len(verdicts)} checks")
        for v in fails:
            lines.append(
                f"  FAIL: {v.get('location', '?')} — {v.get('widget', '?')}: "
                f"observed {v.get('observed', '?')}, expected {v.get('expected', '?')} "
                f"(confidence {v.get('confidence', '?')})"
            )

    if diffs:
        lines.append(f"{len(diffs)} difference(s) from reference")
        for d in diffs:
            lines.append(
                f"  {str(d.get('severity', '?')).upper()}: {d.get('element', '?')}: "
                f"candidate={d.get('candidate', '?')} vs reference={d.get('reference', '?')} "
                f"— {d.get('action', '?')}"
            )

    if retracted:
        lines.append(f"{len(retracted)} claim(s) retracted under adversarial review")

    return "\n".join(lines)


def _run_scenario(
    image: str,
    scenario: str,
    verify: bool,
    reference: str | None,
    json_only: bool,
    model: str | None,
    refresh_model: bool,
    theme: str = "main",
    images: list[str] | None = None,
) -> int:
    if scenario == "consistency-check":
        if not images or len(images) < 2:
            raise ValueError("consistency-check requires --images with 2 or more files")
        prompt = prompt_builder.build_prompt(scenario, theme=theme)
        primary, additional = images[0], images[1:]
    else:
        prompt = prompt_builder.build_prompt(scenario, reference=reference, theme=theme)
        primary, additional = image, ([reference] if reference else [])

    text = analyze_image(
        primary,
        prompt,
        model=model,
        refresh_model=refresh_model,
        json_mode=True,
        additional_images=additional,
    )
    parsed = _parse_json(text)
    parsed.setdefault("scenario", scenario)

    if verify:
        adv = prompt_builder.build_adversarial_prompt(json.dumps(parsed), scenario)
        text2 = analyze_image(
            primary,
            adv,
            model=model,
            refresh_model=False,  # reuse the model resolved above (cache hit)
            json_mode=True,
            additional_images=additional,
        )
        parsed = _parse_json(text2)
        parsed.setdefault("scenario", scenario)

    if json_only:
        print(json.dumps(parsed, indent=2))
    else:
        print(_render_summary(parsed))
        print()
        print("--- JSON ---")
        print(json.dumps(parsed, indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a Gemini vision model about an image."
    )
    parser.add_argument("image", nargs="?", default=None, help="Path to the image file")
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="What to ask about it (raw mode). Omit when using --scenario.",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        choices=sorted(prompt_builder.SCENARIOS),
        help="Scenario-driven prompt (style-check, layout-check, match-reference, "
        "consistency-check).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Add a second adversarial pass that tries to refute the first "
        "pass's findings (scenario mode only).",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Second image for the match-reference scenario.",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=None,
        help="Multiple images (2+) for the consistency-check scenario.",
    )
    parser.add_argument(
        "--theme",
        choices=["main", "auth"],
        default="main",
        help="Which spec to evaluate against: the main design system (default) "
        "or the auth/login-flow palette.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print only the structured JSON (scenario mode only).",
    )
    parser.add_argument(
        "--refresh-model",
        action="store_true",
        help="Force live re-discovery instead of using the cache",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model selection with an explicit model name "
        "(bypasses discovery/cache; for debugging only)",
    )
    args = parser.parse_args()

    if args.scenario:
        if args.prompt:
            parser.error("provide either a prompt string (raw mode) OR --scenario, not both")
        if args.scenario == "consistency-check" and (not args.images or len(args.images) < 2):
            parser.error("--scenario consistency-check requires --images with 2 or more files")
        try:
            return _run_scenario(
                args.image,
                args.scenario,
                verify=args.verify,
                reference=args.reference,
                json_only=args.json_only,
                model=args.model,
                refresh_model=args.refresh_model,
                theme=args.theme,
                images=args.images,
            )
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:  # API errors, no-text, etc.
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    if not args.prompt:
        parser.error("provide a prompt string, or use --scenario")

    try:
        print(
            analyze_image(
                args.image,
                args.prompt,
                model=args.model,
                refresh_model=args.refresh_model,
            )
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:  # API errors, no-text, etc.
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
