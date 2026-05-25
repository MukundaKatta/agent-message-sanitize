"""
agent_message_sanitize — sanitize LLM message lists before sending.

Removes None/empty content, normalizes roles, strips bad blocks,
and optionally collapses adjacent same-role messages.
Zero dependencies (stdlib only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Message = dict[str, Any]
ContentBlock = dict[str, Any]

VALID_ROLES = frozenset({"user", "assistant", "system", "tool"})

# Role aliases → canonical
ROLE_ALIASES: dict[str, str] = {
    "human": "user",
    "bot": "assistant",
    "ai": "assistant",
    "model": "assistant",
}

VALID_BLOCK_TYPES = frozenset({
    "text", "image", "tool_use", "tool_result",
    "thinking", "redacted_thinking", "document",
})


@dataclass
class SanitizeResult:
    """Result of a sanitize operation."""

    messages: list[Message]
    removed_count: int = 0
    modified_count: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True if no messages were removed or modified."""
        return self.removed_count == 0 and self.modified_count == 0

    def __len__(self) -> int:
        return len(self.messages)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_role(role: Any) -> str:
    """Normalize a role string to a canonical value."""
    if not isinstance(role, str):
        return str(role)
    r = role.strip().lower()
    return ROLE_ALIASES.get(r, r)


def _is_empty_string(value: Any) -> bool:
    return isinstance(value, str) and not value.strip()


def _is_empty_content(content: Any) -> bool:
    """Return True if content is effectively empty."""
    if content is None:
        return True
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        return len(content) == 0
    return False


def _text_of(block: ContentBlock) -> str:
    if isinstance(block, dict):
        return block.get("text", "") or ""
    if hasattr(block, "text"):
        return str(block.text)
    return ""


def _sanitize_content_blocks(
    blocks: list[Any],
    *,
    remove_empty_text: bool,
    allowed_types: frozenset[str] | None,
    warnings: list[str],
    msg_index: int,
) -> tuple[list[ContentBlock], bool]:
    """
    Sanitize a list of content blocks.

    Returns (cleaned_blocks, was_modified).
    """
    cleaned: list[ContentBlock] = []
    modified = False
    for block in blocks:
        if not isinstance(block, dict):
            # Skip non-dict blocks
            warnings.append(f"message[{msg_index}]: non-dict block dropped")
            modified = True
            continue
        btype = block.get("type", "")
        # Filter by allowed types
        if allowed_types is not None and btype not in allowed_types:
            warnings.append(f"message[{msg_index}]: block type {btype!r} dropped")
            modified = True
            continue
        # Strip empty text blocks
        if remove_empty_text and btype == "text":
            if _is_empty_string(_text_of(block)):
                warnings.append(f"message[{msg_index}]: empty text block dropped")
                modified = True
                continue
        cleaned.append(block)
    return cleaned, modified


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sanitize_messages(
    messages: list[Any],
    *,
    normalize_roles: bool = True,
    remove_none_content: bool = True,
    remove_empty_content: bool = True,
    remove_empty_text_blocks: bool = True,
    allowed_block_types: frozenset[str] | None = None,
    collapse_adjacent_same_role: bool = False,
    collapse_separator: str = "\n\n",
    remove_unknown_keys: bool = False,
    keep_keys: frozenset[str] = frozenset({"role", "content"}),
) -> SanitizeResult:
    """
    Sanitize a list of LLM messages.

    Args:
        messages: Input message list.
        normalize_roles: Map aliases (``human`` → ``user``, ``ai`` → ``assistant``).
        remove_none_content: Drop messages with ``content: None``.
        remove_empty_content: Drop messages with empty string or empty list content.
        remove_empty_text_blocks: Remove text blocks with only whitespace.
        allowed_block_types: If set, drop blocks whose type is not in this set.
        collapse_adjacent_same_role: Merge consecutive same-role messages.
        collapse_separator: Separator used when collapsing.
        remove_unknown_keys: If True, strip all keys except ``keep_keys``.
        keep_keys: Keys to keep when ``remove_unknown_keys`` is True.

    Returns:
        :class:`SanitizeResult`
    """
    result: list[Message] = []
    warnings: list[str] = []
    removed = 0
    modified = 0

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            warnings.append(f"message[{i}]: non-dict message dropped")
            removed += 1
            continue

        msg = dict(msg)  # shallow copy

        # Normalize role
        role = msg.get("role")
        if normalize_roles and isinstance(role, str):
            new_role = _normalize_role(role)
            if new_role != role:
                msg["role"] = new_role
                modified += 1

        # Normalize content
        content = msg.get("content")

        if content is None:
            if remove_none_content:
                warnings.append(f"message[{i}]: None content dropped")
                removed += 1
                continue
            # None but not removing it — skip empty-content check below
        elif remove_empty_content and _is_empty_content(content):
            warnings.append(f"message[{i}]: empty content dropped")
            removed += 1
            continue

        # Sanitize content blocks
        if isinstance(content, list):
            cleaned, was_mod = _sanitize_content_blocks(
                content,
                remove_empty_text=remove_empty_text_blocks,
                allowed_types=allowed_block_types,
                warnings=warnings,
                msg_index=i,
            )
            if was_mod:
                msg["content"] = cleaned
                modified += 1
            # Drop message if all blocks were removed
            if not cleaned:
                warnings.append(f"message[{i}]: all blocks removed, message dropped")
                removed += 1
                continue

        # Strip unknown keys
        if remove_unknown_keys:
            before_keys = set(msg.keys())
            msg = {k: v for k, v in msg.items() if k in keep_keys}
            if set(msg.keys()) != before_keys:
                modified += 1

        result.append(msg)

    # Collapse adjacent same-role messages
    if collapse_adjacent_same_role and result:
        collapsed: list[Message] = [result[0]]
        for msg in result[1:]:
            prev = collapsed[-1]
            if msg.get("role") == prev.get("role"):
                # Merge content
                prev_content = prev.get("content", "")
                cur_content = msg.get("content", "")
                if isinstance(prev_content, str) and isinstance(cur_content, str):
                    prev["content"] = prev_content + collapse_separator + cur_content
                elif isinstance(prev_content, list) and isinstance(cur_content, list):
                    prev["content"] = prev_content + cur_content
                elif isinstance(prev_content, str) and isinstance(cur_content, list):
                    prev["content"] = [{"type": "text", "text": prev_content}] + cur_content
                elif isinstance(prev_content, list) and isinstance(cur_content, str):
                    prev["content"] = prev_content + [{"type": "text", "text": cur_content}]
                modified += 1
            else:
                collapsed.append(msg)
        removed += len(result) - len(collapsed)
        result = collapsed

    return SanitizeResult(
        messages=result,
        removed_count=removed,
        modified_count=modified,
        warnings=warnings,
    )


def clean_messages(messages: list[Any], **kwargs: Any) -> list[Message]:
    """
    Sanitize messages and return the cleaned list directly.

    Equivalent to ``sanitize_messages(messages, **kwargs).messages``.
    """
    return sanitize_messages(messages, **kwargs).messages


def drop_empty(messages: list[Any]) -> list[Message]:
    """Remove messages with None or empty content."""
    return clean_messages(
        messages,
        normalize_roles=False,
        remove_none_content=True,
        remove_empty_content=True,
        remove_empty_text_blocks=True,
    )


def normalize_roles(messages: list[Any]) -> list[Message]:
    """Normalize role aliases to canonical names only."""
    return clean_messages(
        messages,
        normalize_roles=True,
        remove_none_content=False,
        remove_empty_content=False,
        remove_empty_text_blocks=False,
    )
