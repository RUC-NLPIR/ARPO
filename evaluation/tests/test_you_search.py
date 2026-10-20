"""Tests for the optional You.com search backend. Run from evaluation/: pytest tests"""
import asyncio
import os
import sys
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "tools"))

import you_search  # noqa: E402
from you_search import you_search_request  # noqa: E402

PAYLOAD = {
    "results": {
        "web": [
            {
                "url": "https://example.com/a",
                "title": "A",
                "description": "desc a",
                "snippets": ["snip 1", "snip 2"],
                "page_age": "2026-01-02T00:00:00",
            },
            {"url": "https://example.com/b", "title": "B", "description": "desc b"},
        ]
    }
}


def _resp(status, payload=None, headers=None):
    r = mock.Mock()
    r.status_code = status
    r.json.return_value = payload or {}
    r.headers = headers or {}
    return r


def test_keyless_uses_get(monkeypatch):
    monkeypatch.delenv("YDC_API_KEY", raising=False)
    s = mock.Mock()
    s.get.return_value = _resp(200, PAYLOAD)
    out = you_search_request("q", count=5, session=s)
    url = s.get.call_args.args[0]
    kw = s.get.call_args.kwargs
    assert url == you_search.KEYLESS_URL
    assert kw["params"] == {"query": "q", "count": 5}
    assert kw["headers"]["User-Agent"] == "youdotcom-integration/ruc-nlpir-arpo"
    assert "X-API-Key" not in kw["headers"]
    s.post.assert_not_called()
    assert out["organic"][0] == {
        "title": "A",
        "link": "https://example.com/a",
        "description": "desc a snip 1 snip 2",
        "datePublished": "2026-01-02T00:00:00",
        "site_name": "",
    }
    assert out["organic"][1]["description"] == "desc b"


def test_keyed_uses_post_with_env_key(monkeypatch):
    monkeypatch.setenv("YDC_API_KEY", "k")
    s = mock.Mock()
    s.post.return_value = _resp(200, PAYLOAD)
    out = you_search_request("q", session=s)
    assert s.post.call_args.args[0] == you_search.KEYED_URL
    assert s.post.call_args.kwargs["headers"]["X-API-Key"] == "k"
    assert len(out["organic"]) == 2


def test_bad_key_returns_none_without_retry(monkeypatch):
    for code in (401, 402, 403):
        s = mock.Mock()
        s.post.return_value = _resp(code)
        assert you_search_request("q", api_key="bad", session=s) is None
        assert s.post.call_count == 1


def test_429_honors_retry_after(monkeypatch):
    monkeypatch.delenv("YDC_API_KEY", raising=False)
    s = mock.Mock()
    s.get.side_effect = [_resp(429, headers={"Retry-After": "2"}), _resp(200, PAYLOAD)]
    with mock.patch.object(you_search.time, "sleep") as sleep:
        out = you_search_request("q", session=s)
    sleep.assert_called_once_with(2.0)
    assert len(out["organic"]) == 2


def test_empty_results():
    s = mock.Mock()
    s.get.return_value = _resp(200, {"results": {"web": []}})
    assert you_search_request("q", api_key="", session=s) == {"general": {}, "organic": []}


def test_gives_up_after_retries():
    s = mock.Mock()
    s.get.return_value = _resp(500)
    with mock.patch.object(you_search.time, "sleep"):
        assert you_search_request("q", api_key="", max_retries=2, session=s) is None
    assert s.get.call_count == 2


def test_you_search_tool_formats_like_bing(tmp_path):
    # Load the tools package without its __init__, which pulls in vllm via the SDS tool.
    import importlib
    import types

    pkg = types.ModuleType("arpo_tools")
    pkg.__path__ = [os.path.join(os.path.dirname(__file__), "..", "src", "tools")]
    sys.modules.setdefault("arpo_tools", pkg)
    YouSearchTool = importlib.import_module("arpo_tools.search_tool").YouSearchTool

    tool = YouSearchTool(search_cache_file=str(tmp_path / "c.db"), max_results=2)
    assert tool.name == "you_search" and tool.trigger_tag == "search"
    tool.search_cache_manager.add_to_cache = mock.AsyncMock()
    with mock.patch("arpo_tools.you_search.you_search_request", return_value=you_search._to_organic(PAYLOAD)) as m:
        out = asyncio.run(tool.execute("q"))
    assert m.call_args.kwargs["api_key"] is None
    assert out == "Page 1: desc a snip 1 snip 2\nPage 2: desc b"
