import pytest
from agent_message_sanitize import (
    SanitizeResult,
    sanitize_messages,
    clean_messages,
    drop_empty,
    normalize_roles,
    ROLE_ALIASES,
)


# ---------------------------------------------------------------------------
# Basic sanitize
# ---------------------------------------------------------------------------

def test_sanitize_returns_result():
    r = sanitize_messages([{"role": "user", "content": "hi"}])
    assert isinstance(r, SanitizeResult)

def test_sanitize_clean_messages_unchanged():
    msgs = [{"role": "user", "content": "hello"}]
    r = sanitize_messages(msgs)
    assert r.messages == [{"role": "user", "content": "hello"}]
    assert r.ok

def test_sanitize_empty_list():
    r = sanitize_messages([])
    assert r.messages == []
    assert r.ok

def test_sanitize_preserves_multi_turn():
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "thanks"},
    ]
    r = sanitize_messages(msgs)
    assert len(r.messages) == 3


# ---------------------------------------------------------------------------
# None content removal
# ---------------------------------------------------------------------------

def test_remove_none_content():
    msgs = [
        {"role": "user", "content": None},
        {"role": "user", "content": "valid"},
    ]
    r = sanitize_messages(msgs)
    assert len(r.messages) == 1
    assert r.removed_count == 1

def test_keep_none_content_when_disabled():
    msgs = [{"role": "user", "content": None}]
    r = sanitize_messages(msgs, remove_none_content=False)
    assert len(r.messages) == 1

def test_none_content_adds_warning():
    msgs = [{"role": "user", "content": None}]
    r = sanitize_messages(msgs)
    assert any("None content" in w for w in r.warnings)


# ---------------------------------------------------------------------------
# Empty content removal
# ---------------------------------------------------------------------------

def test_remove_empty_string_content():
    msgs = [{"role": "user", "content": ""}]
    r = sanitize_messages(msgs)
    assert r.removed_count == 1

def test_remove_whitespace_only_content():
    msgs = [{"role": "user", "content": "   "}]
    r = sanitize_messages(msgs)
    assert r.removed_count == 1

def test_remove_empty_list_content():
    msgs = [{"role": "user", "content": []}]
    r = sanitize_messages(msgs)
    assert r.removed_count == 1

def test_keep_empty_content_when_disabled():
    msgs = [{"role": "user", "content": ""}]
    r = sanitize_messages(msgs, remove_empty_content=False)
    assert len(r.messages) == 1


# ---------------------------------------------------------------------------
# Role normalization
# ---------------------------------------------------------------------------

def test_normalize_human_to_user():
    msgs = [{"role": "human", "content": "hi"}]
    r = sanitize_messages(msgs)
    assert r.messages[0]["role"] == "user"
    assert r.modified_count >= 1

def test_normalize_ai_to_assistant():
    msgs = [{"role": "ai", "content": "hello"}]
    r = sanitize_messages(msgs)
    assert r.messages[0]["role"] == "assistant"

def test_normalize_bot_to_assistant():
    msgs = [{"role": "bot", "content": "hello"}]
    r = sanitize_messages(msgs)
    assert r.messages[0]["role"] == "assistant"

def test_normalize_model_to_assistant():
    msgs = [{"role": "model", "content": "hi"}]
    r = sanitize_messages(msgs)
    assert r.messages[0]["role"] == "assistant"

def test_normalize_roles_false():
    msgs = [{"role": "human", "content": "hi"}]
    r = sanitize_messages(msgs, normalize_roles=False)
    assert r.messages[0]["role"] == "human"

def test_normalize_case_insensitive():
    msgs = [{"role": "HUMAN", "content": "hi"}]
    r = sanitize_messages(msgs)
    assert r.messages[0]["role"] == "user"

def test_known_role_unchanged():
    msgs = [{"role": "user", "content": "hi"}]
    r = sanitize_messages(msgs)
    assert r.messages[0]["role"] == "user"
    assert r.ok


# ---------------------------------------------------------------------------
# Content block sanitization
# ---------------------------------------------------------------------------

def test_empty_text_block_removed():
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": ""},
        {"type": "text", "text": "real content"},
    ]}]
    r = sanitize_messages(msgs)
    assert len(r.messages[0]["content"]) == 1

def test_whitespace_text_block_removed():
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "   "},
    ]}]
    r = sanitize_messages(msgs)
    assert r.removed_count == 1  # whole message dropped (no blocks left)

def test_non_dict_block_removed():
    msgs = [{"role": "user", "content": [
        "not a dict",
        {"type": "text", "text": "valid"},
    ]}]
    r = sanitize_messages(msgs)
    assert len(r.messages[0]["content"]) == 1

def test_allowed_block_types_filters():
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "hello"},
        {"type": "image", "source": {}},
    ]}]
    r = sanitize_messages(msgs, allowed_block_types=frozenset({"text"}))
    assert len(r.messages[0]["content"]) == 1
    assert r.messages[0]["content"][0]["type"] == "text"

def test_all_blocks_removed_drops_message():
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": ""},
    ]}]
    r = sanitize_messages(msgs)
    assert r.removed_count == 1
    assert len(r.messages) == 0


# ---------------------------------------------------------------------------
# Non-dict messages
# ---------------------------------------------------------------------------

def test_non_dict_message_dropped():
    msgs = ["not a dict", {"role": "user", "content": "ok"}]
    r = sanitize_messages(msgs)
    assert len(r.messages) == 1
    assert r.removed_count == 1


# ---------------------------------------------------------------------------
# remove_unknown_keys
# ---------------------------------------------------------------------------

def test_remove_unknown_keys():
    msgs = [{"role": "user", "content": "hi", "metadata": {"id": 1}}]
    r = sanitize_messages(msgs, remove_unknown_keys=True)
    assert set(r.messages[0].keys()) == {"role", "content"}

def test_keep_keys_custom():
    msgs = [{"role": "user", "content": "hi", "name": "Alice"}]
    r = sanitize_messages(
        msgs,
        remove_unknown_keys=True,
        keep_keys=frozenset({"role", "content", "name"}),
    )
    assert "name" in r.messages[0]


# ---------------------------------------------------------------------------
# Collapse adjacent same-role
# ---------------------------------------------------------------------------

def test_collapse_adjacent_same_role_string():
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
    ]
    r = sanitize_messages(msgs, collapse_adjacent_same_role=True)
    assert len(r.messages) == 1
    assert "first" in r.messages[0]["content"]
    assert "second" in r.messages[0]["content"]

def test_collapse_different_roles_not_merged():
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    r = sanitize_messages(msgs, collapse_adjacent_same_role=True)
    assert len(r.messages) == 2

def test_collapse_three_same_role():
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    r = sanitize_messages(msgs, collapse_adjacent_same_role=True)
    assert len(r.messages) == 1

def test_collapse_separator():
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
    ]
    r = sanitize_messages(msgs, collapse_adjacent_same_role=True, collapse_separator="---")
    assert "---" in r.messages[0]["content"]

def test_collapse_list_content():
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "a"}]},
        {"role": "user", "content": [{"type": "text", "text": "b"}]},
    ]
    r = sanitize_messages(msgs, collapse_adjacent_same_role=True)
    assert len(r.messages) == 1
    assert len(r.messages[0]["content"]) == 2


# ---------------------------------------------------------------------------
# SanitizeResult
# ---------------------------------------------------------------------------

def test_result_len():
    msgs = [{"role": "user", "content": "hi"}]
    r = sanitize_messages(msgs)
    assert len(r) == 1

def test_result_ok_true():
    msgs = [{"role": "user", "content": "hi"}]
    r = sanitize_messages(msgs)
    assert r.ok

def test_result_ok_false_when_modified():
    msgs = [{"role": "human", "content": "hi"}]
    r = sanitize_messages(msgs)
    assert not r.ok


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def test_clean_messages_returns_list():
    msgs = [{"role": "user", "content": "hi"}]
    result = clean_messages(msgs)
    assert isinstance(result, list)

def test_drop_empty_removes_none():
    msgs = [
        {"role": "user", "content": None},
        {"role": "user", "content": "ok"},
    ]
    result = drop_empty(msgs)
    assert len(result) == 1

def test_normalize_roles_fn():
    msgs = [{"role": "human", "content": "hi"}]
    result = normalize_roles(msgs)
    assert result[0]["role"] == "user"

def test_role_aliases_exported():
    assert "human" in ROLE_ALIASES
    assert ROLE_ALIASES["human"] == "user"
