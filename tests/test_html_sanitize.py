from utils.html_sanitize import sanitize_html


def test_sanitize_html_strips_leading_junk():
    raw = "\ufeff⚠⚠️  \n<p>Hello</p>"
    assert sanitize_html(raw).startswith("<p>Hello")


def test_sanitize_html_balances_paragraphs():
    raw = "<p>Line 1<p>Line 2</p>"
    sanitized = sanitize_html(raw)
    assert sanitized.endswith("</p>")
    assert sanitized.count("<p") == sanitized.count("</p>")


def test_sanitize_html_keeps_balanced_html_unchanged():
    raw = "<div><p>Content</p></div>"
    assert sanitize_html(raw) == raw


def test_sanitize_html_handles_non_string_inputs():
    assert sanitize_html(None) == ""
    assert sanitize_html(123).endswith("123")
