"""Tests for agent_message_sanitize."""

import copy

from agent_message_sanitize import MessageSanitizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sanitizer(**kwargs) -> MessageSanitizer:
    return MessageSanitizer(**kwargs)


# ---------------------------------------------------------------------------
# Basic / happy-path tests
# ---------------------------------------------------------------------------

def test_clean_message_no_changes():
    s = MessageSanitizer()
    msgs = [{"role": "user", "content": "Hello, world!"}]
    result = s.sanitize(msgs)
    assert result.messages == msgs
    assert result.changes == []


def test_empty_messages_list():
    s = MessageSanitizer()
    result = s.sanitize([])
    assert result.messages == []
    assert result.changes == []


def test_sanitize_result_is_new_list():
    s = MessageSanitizer()
    msgs = [{"role": "user", "content": "hi"}]
    result = s.sanitize(msgs)
    assert result.messages is not msgs


def test_sanitize_one_returns_new_dict():
    s = MessageSanitizer()
    msg = {"role": "user", "content": "hi"}
    new_msg, _ = s.sanitize_one(msg)
    assert new_msg is not msg


def test_message_without_content_key_kept_unchanged():
    s = MessageSanitizer()
    msg = {"role": "system"}
    result = s.sanitize([msg])
    assert result.messages == [msg]
    assert result.changes == []


def test_multiple_messages_each_sanitized_independently():
    s = MessageSanitizer(max_content_chars=5)
    msgs = [
        {"role": "user", "content": "short"},
        {"role": "assistant", "content": "longer than five chars"},
    ]
    result = s.sanitize(msgs)
    assert result.messages[0]["content"] == "short"
    assert result.messages[1]["content"] == "longe..."
    assert len(result.changes) == 1
    assert result.changes[0].index == 1


# ---------------------------------------------------------------------------
# Immutability tests
# ---------------------------------------------------------------------------

def test_never_mutates_input_list():
    s = MessageSanitizer(max_content_chars=3)
    original = [{"role": "user", "content": "Hello!"}]
    snapshot = copy.deepcopy(original)
    s.sanitize(original)
    assert original == snapshot


def test_never_mutates_input_dicts():
    s = MessageSanitizer(strip_null_bytes=True)
    msg = {"role": "user", "content": "hi\x00there"}
    original_content = msg["content"]
    s.sanitize([msg])
    assert msg["content"] == original_content


def test_sanitize_one_never_mutates_input():
    s = MessageSanitizer(max_content_chars=2)
    msg = {"role": "user", "content": "hello"}
    s.sanitize_one(msg)
    assert msg["content"] == "hello"


# ---------------------------------------------------------------------------
# strip_null_bytes tests
# ---------------------------------------------------------------------------

def test_strip_null_bytes_true_removes_nulls():
    s = MessageSanitizer(strip_null_bytes=True)
    msgs = [{"role": "user", "content": "he\x00llo\x00"}]
    result = s.sanitize(msgs)
    assert result.messages[0]["content"] == "hello"
    assert any("null" in c.description for c in result.changes)


def test_strip_null_bytes_false_leaves_nulls():
    s = MessageSanitizer(strip_null_bytes=False)
    msgs = [{"role": "user", "content": "he\x00llo"}]
    result = s.sanitize(msgs)
    assert result.messages[0]["content"] == "he\x00llo"
    assert result.changes == []


def test_strip_null_bytes_records_change_with_correct_index():
    s = MessageSanitizer(strip_null_bytes=True)
    msgs = [
        {"role": "system", "content": "clean"},
        {"role": "user", "content": "bad\x00"},
    ]
    result = s.sanitize(msgs)
    assert len(result.changes) == 1
    assert result.changes[0].index == 1


# ---------------------------------------------------------------------------
# coerce_content_to_str tests
# ---------------------------------------------------------------------------

def test_coerce_int_content_to_str():
    s = MessageSanitizer(coerce_content_to_str=True)
    msgs = [{"role": "user", "content": 42}]
    result = s.sanitize(msgs)
    assert result.messages[0]["content"] == "42"
    assert any("coerced" in c.description for c in result.changes)


def test_coerce_false_leaves_non_str_as_is():
    s = MessageSanitizer(coerce_content_to_str=False, remove_none_content=False)
    msgs = [{"role": "user", "content": 99}]
    result = s.sanitize(msgs)
    assert result.messages[0]["content"] == 99
    assert result.changes == []


def test_coerce_records_change_index():
    s = MessageSanitizer(coerce_content_to_str=True)
    msgs = [
        {"role": "system", "content": "ok"},
        {"role": "user", "content": 3.14},
    ]
    result = s.sanitize(msgs)
    assert result.changes[0].index == 1
    assert result.messages[1]["content"] == "3.14"


# ---------------------------------------------------------------------------
# remove_none_content tests
# ---------------------------------------------------------------------------

def test_remove_none_content_true_drops_message():
    s = MessageSanitizer(remove_none_content=True, coerce_content_to_str=False)
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None},
    ]
    result = s.sanitize(msgs)
    assert len(result.messages) == 1
    assert result.messages[0]["content"] == "hi"


def test_remove_none_content_true_records_removal_change():
    s = MessageSanitizer(remove_none_content=True, coerce_content_to_str=False)
    msgs = [{"role": "user", "content": None}]
    result = s.sanitize(msgs)
    assert any("removed" in c.description for c in result.changes)


def test_remove_none_content_false_keeps_none_message():
    s = MessageSanitizer(remove_none_content=False, coerce_content_to_str=False)
    msgs = [{"role": "user", "content": None}]
    result = s.sanitize(msgs)
    assert len(result.messages) == 1
    assert result.messages[0]["content"] is None
    assert result.changes == []


def test_change_index_refers_to_original_list_index():
    s = MessageSanitizer(remove_none_content=True, coerce_content_to_str=False)
    msgs = [
        {"role": "user", "content": None},   # index 0 — will be removed
        {"role": "assistant", "content": None},  # index 1 — will be removed
    ]
    result = s.sanitize(msgs)
    assert len(result.messages) == 0
    removal_changes = [c for c in result.changes if "removed" in c.description]
    indices = {c.index for c in removal_changes}
    assert indices == {0, 1}


# ---------------------------------------------------------------------------
# max_content_chars tests
# ---------------------------------------------------------------------------

def test_long_content_truncated_with_ellipsis():
    s = MessageSanitizer(max_content_chars=10)
    msgs = [{"role": "user", "content": "a" * 20}]
    result = s.sanitize(msgs)
    assert result.messages[0]["content"] == "a" * 10 + "..."
    assert any("truncated" in c.description for c in result.changes)


def test_content_exactly_at_limit_not_truncated():
    s = MessageSanitizer(max_content_chars=5)
    msgs = [{"role": "user", "content": "hello"}]
    result = s.sanitize(msgs)
    assert result.messages[0]["content"] == "hello"
    assert result.changes == []


def test_truncation_total_length_is_limit_plus_three():
    limit = 7
    s = MessageSanitizer(max_content_chars=limit)
    msgs = [{"role": "user", "content": "x" * 100}]
    result = s.sanitize(msgs)
    assert len(result.messages[0]["content"]) == limit + 3


# ---------------------------------------------------------------------------
# normalize_whitespace tests
# ---------------------------------------------------------------------------

def test_normalize_whitespace_true_collapses_spaces():
    s = MessageSanitizer(normalize_whitespace=True)
    msgs = [{"role": "user", "content": "  hello   world  "}]
    result = s.sanitize(msgs)
    assert result.messages[0]["content"] == "hello world"
    assert any("whitespace" in c.description for c in result.changes)


def test_normalize_whitespace_false_leaves_spaces():
    s = MessageSanitizer(normalize_whitespace=False)
    msgs = [{"role": "user", "content": "  hello   world  "}]
    result = s.sanitize(msgs)
    assert result.messages[0]["content"] == "  hello   world  "
    assert result.changes == []


def test_normalize_whitespace_no_change_if_already_clean():
    s = MessageSanitizer(normalize_whitespace=True)
    msgs = [{"role": "user", "content": "hello world"}]
    result = s.sanitize(msgs)
    assert result.messages[0]["content"] == "hello world"
    assert result.changes == []


# ---------------------------------------------------------------------------
# sanitize_one tests
# ---------------------------------------------------------------------------

def test_sanitize_one_applies_operations_in_order():
    # coerce int → "123\x00" would be odd, but with str input:
    # strip_null_bytes happens before normalize_whitespace and truncation.
    s = MessageSanitizer(
        strip_null_bytes=True,
        normalize_whitespace=True,
        max_content_chars=5,
        coerce_content_to_str=True,
    )
    msg = {"role": "user", "content": "ab\x00  cd  ef"}
    new_msg, changes = s.sanitize_one(msg, index=0)
    # After null strip: "ab  cd  ef"
    # After normalize: "ab cd ef"
    # After truncate (limit=5): "ab cd..."
    assert new_msg["content"] == "ab cd..."
    assert len(changes) >= 2  # at least null strip + truncation


def test_sanitize_one_index_propagated_in_changes():
    s = MessageSanitizer(strip_null_bytes=True)
    msg = {"role": "user", "content": "a\x00b"}
    _, changes = s.sanitize_one(msg, index=7)
    assert changes[0].index == 7


# ---------------------------------------------------------------------------
# sanitize_text tests
# ---------------------------------------------------------------------------

def test_sanitize_text_strips_null_bytes():
    s = MessageSanitizer(strip_null_bytes=True)
    text, descs = s.sanitize_text("hel\x00lo")
    assert text == "hello"
    assert any("null" in d for d in descs)


def test_sanitize_text_truncates():
    s = MessageSanitizer(max_content_chars=3)
    text, descs = s.sanitize_text("abcdef")
    assert text == "abc..."
    assert any("truncated" in d for d in descs)


def test_sanitize_text_normalizes_whitespace():
    s = MessageSanitizer(normalize_whitespace=True)
    text, descs = s.sanitize_text("  foo   bar  ")
    assert text == "foo bar"
    assert any("whitespace" in d for d in descs)


def test_sanitize_text_clean_no_descriptions():
    s = MessageSanitizer()
    text, descs = s.sanitize_text("clean text")
    assert text == "clean text"
    assert descs == []
