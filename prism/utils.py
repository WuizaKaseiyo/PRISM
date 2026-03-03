"""Shared utilities for PRISM."""
from __future__ import annotations

import json
import re
from typing import Any


def extract_json_from_text(text: str) -> dict[str, Any] | None:
    """Extract JSON object from text using multiple strategies.

    3-strategy fallback:
    1. Direct JSON.parse of entire text
    2. ```json code block extraction
    3. Balanced brace counting for embedded JSON
    """
    try:
        # Strategy 1: direct parse
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Strategy 2: ```json blocks
        matches = re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

        # Strategy 3: balanced brace counting
        for candidate in _find_json_objects(text):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    except Exception:
        pass

    return None


def _find_json_objects(text: str) -> list[str]:
    """Find JSON objects in text using balanced brace counting.

    Handles quoted strings to avoid counting braces inside them.
    """
    objects: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            depth = 1
            start = i
            i += 1
            while i < len(text) and depth > 0:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                elif text[i] == '"':
                    i += 1
                    while i < len(text) and text[i] != '"':
                        if text[i] == "\\":
                            i += 1
                        i += 1
                i += 1
            if depth == 0:
                objects.append(text[start:i])
        else:
            i += 1
    return objects
