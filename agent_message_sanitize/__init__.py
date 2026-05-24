"""
agent-message-sanitize: sanitize LLM message lists before sending.

Strips null bytes, truncates oversized content, removes None-content messages,
and coerces non-string content to string. Returns a sanitized copy with a
change log. Never mutates the input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Change:
    """Record of a single sanitization operation applied to one message."""

    index: int  # message index in the original list
    field: str  # e.g. "content"
    description: str  # human-readable description of what changed


@dataclass
class SanitizeResult:
    """Result of sanitizing a list of messages."""

    messages: list[dict]  # sanitized copy (never the same objects as input)
    changes: list[Change] = field(default_factory=list)  # log of what changed


class MessageSanitizer:
    """Sanitize LLM message dicts before sending to a model API."""

    def __init__(
        self,
        max_content_chars: int = 100_000,
        strip_null_bytes: bool = True,
        normalize_whitespace: bool = False,
        remove_none_content: bool = True,
        coerce_content_to_str: bool = True,
    ) -> None:
        self.max_content_chars = max_content_chars
        self.strip_null_bytes = strip_null_bytes
        self.normalize_whitespace = normalize_whitespace
        self.remove_none_content = remove_none_content
        self.coerce_content_to_str = coerce_content_to_str

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sanitize(self, messages: list[dict]) -> SanitizeResult:
        """Sanitize a list of message dicts.

        Returns a SanitizeResult with a new list and a log of all changes.
        The original list and its dicts are never mutated.
        """
        sanitized: list[dict] = []
        all_changes: list[Change] = []

        for original_index, msg in enumerate(messages):
            new_msg, changes = self.sanitize_one(msg, index=original_index)
            all_changes.extend(changes)

            # Decide whether to include the message in the output.
            content = new_msg.get("content", _SENTINEL)
            if content is _SENTINEL:
                # No "content" key at all — always keep the message.
                sanitized.append(new_msg)
            elif content is None and self.remove_none_content:
                # Drop the message; record a removal change.
                all_changes.append(
                    Change(
                        index=original_index,
                        field="content",
                        description="message removed: content is None",
                    )
                )
            else:
                sanitized.append(new_msg)

        return SanitizeResult(messages=sanitized, changes=all_changes)

    def sanitize_one(self, message: dict, index: int = 0) -> tuple[dict, list[Change]]:
        """Sanitize a single message dict.

        Returns (new_dict, changes). Never mutates the input dict.
        Operations are applied in this order:
          1. coerce_content_to_str
          2. strip_null_bytes
          3. normalize_whitespace
          4. max_content_chars truncation
        """
        # Shallow-copy so we never touch the caller's dict.
        new_msg = dict(message)
        changes: list[Change] = []

        # No "content" key — nothing to sanitize.
        if "content" not in new_msg:
            return new_msg, changes

        content = new_msg["content"]

        # 1. Coerce non-str, non-None content to str.
        if self.coerce_content_to_str and content is not None and not isinstance(content, str):
            original_repr = repr(content)
            content = str(content)
            changes.append(
                Change(
                    index=index,
                    field="content",
                    description=f"coerced to str from {original_repr}",
                )
            )

        # Remaining operations only apply to str content.
        if isinstance(content, str):
            # 2. Strip null bytes.
            if self.strip_null_bytes and "\x00" in content:
                content = content.replace("\x00", "")
                changes.append(
                    Change(
                        index=index,
                        field="content",
                        description="stripped null bytes (\\x00)",
                    )
                )

            # 3. Normalize whitespace.
            if self.normalize_whitespace:
                normalized = re.sub(r"\s+", " ", content).strip()
                if normalized != content:
                    content = normalized
                    changes.append(
                        Change(
                            index=index,
                            field="content",
                            description="normalized whitespace",
                        )
                    )

            # 4. Truncate to max_content_chars.
            if len(content) > self.max_content_chars:
                content = content[: self.max_content_chars] + "..."
                changes.append(
                    Change(
                        index=index,
                        field="content",
                        description=f"truncated to {self.max_content_chars} chars",
                    )
                )

        new_msg["content"] = content
        return new_msg, changes

    def sanitize_text(self, text: str) -> tuple[str, list[str]]:
        """Convenience method: sanitize a plain string (not a message dict).

        Returns (sanitized_text, list_of_change_descriptions).
        Applies: strip_null_bytes, normalize_whitespace, max_content_chars truncation.
        """
        descriptions: list[str] = []

        if self.strip_null_bytes and "\x00" in text:
            text = text.replace("\x00", "")
            descriptions.append("stripped null bytes (\\x00)")

        if self.normalize_whitespace:
            normalized = re.sub(r"\s+", " ", text).strip()
            if normalized != text:
                text = normalized
                descriptions.append("normalized whitespace")

        if len(text) > self.max_content_chars:
            text = text[: self.max_content_chars] + "..."
            descriptions.append(f"truncated to {self.max_content_chars} chars")

        return text, descriptions


# Internal sentinel to distinguish "key absent" from "key present with None value".
_SENTINEL = object()

__all__ = ["Change", "SanitizeResult", "MessageSanitizer"]
