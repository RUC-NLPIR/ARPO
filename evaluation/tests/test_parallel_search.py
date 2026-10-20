"""Exercise the production resolver/dispatcher/processor with scripted model output.

The local HTTP fixture speaks MCP; the real MCP SDK constructs and parses every
request. No model weights or external search service are needed for this suite.
"""

import asyncio
import contextlib
import io
import json
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.inference_engine import (
    AsyncInference,
    AsyncInferenceCompletion,
    AsyncInferenceCompletionSDS,
)
from src.prompt_manager import PromptManager
from src.tools import parallel_search_tool
from src.tools.parallel_search_tool import MCP_URL, USER_AGENT, ParallelSearchTool
from src.tools.search_provider import resolve_search_provider
from src.tools.search_tool import BingSearchTool


class MCPFixture:
    def __init__(self):
        self.requests = []
        self.mode = "normal"
        self.call_event = threading.Event()
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, data=None):
                body = json.dumps(data).encode() if data is not None else b""
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    self.wfile.write(body)

            def do_DELETE(self):
                self.reply(405)

            def do_GET(self):
                self.reply(405)

            def do_POST(self):
                message = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                fixture.requests.append((self.path, dict(self.headers), message))
                method = message["method"]
                if method.startswith("notifications/"):
                    self.reply(202)
                    return
                if method == "initialize":
                    result = {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "search-fixture", "version": "1"},
                    }
                elif method == "tools/list":
                    result = {
                        "tools": [
                            {
                                "name": name,
                                "inputSchema": {"type": "object", "properties": {}},
                            }
                            for name in ("web_search", "web_fetch")
                        ]
                    }
                elif method == "tools/call":
                    fixture.call_event.set()
                    if fixture.mode == "slow":
                        time.sleep(2)
                    if fixture.mode == "http_error":
                        self.reply(503, {"error": "unavailable"})
                        return
                    if fixture.mode in ("quota", "error"):
                        text = (
                            "Anonymous quota exceeded"
                            if fixture.mode == "quota"
                            else "Search unavailable"
                        )
                        result = {
                            "content": [{"type": "text", "text": text}],
                            "isError": True,
                        }
                    else:
                        data = {
                            "results": [
                                {
                                    "title": "Reference",
                                    "url": "https://example.org/reference",
                                    "excerpts": ["Parallel fact: 42."],
                                },
                                {
                                    "title": "Other",
                                    "url": "https://example.org/other",
                                    "excerpts": ["Other fact."],
                                },
                            ]
                        }
                        if fixture.mode == "empty":
                            data = {"results": []}
                        elif fixture.mode == "fetch_error":
                            data = {
                                "results": [],
                                "errors": [
                                    {
                                        "url": "https://example.org",
                                        "error": "fetch failed",
                                    }
                                ],
                            }
                        elif fixture.mode == "bad_payload":
                            data = {"unexpected": []}
                        result = {
                            "content": [{"type": "text", "text": json.dumps(data)}]
                        }
                        if fixture.mode != "text_only":
                            result["structuredContent"] = data
                else:
                    raise AssertionError(method)
                self.reply(
                    200, {"jsonrpc": "2.0", "id": message["id"], "result": result}
                )

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.url = f"http://127.0.0.1:{self.server.server_port}/mcp"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def calls(self):
        return [m["params"] for _, _, m in self.requests if m["method"] == "tools/call"]


class TokenizerFixture:
    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(message["content"] for message in messages)


class ScriptedModel:
    def __init__(self, steps):
        self.steps = steps
        self.prompts = []

    async def generate(self, prompt, *args, **kwargs):
        self.prompts.append(prompt)
        await asyncio.sleep(0)
        index = prompt.count("<result>") - PromptManager(
            "search"
        ).get_system_prompt().count("<result>")
        output = (
            f"<search>{self.steps[index]}</search>"
            if index < len(self.steps)
            else r"<answer>\boxed{42}</answer>"
        )
        return SimpleNamespace(choices=[SimpleNamespace(text=output)])


def arguments(**overrides):
    params = dict(
        search_provider="auto",
        bing_api_key=None,
        bing_zone="saved_zone",
        search_max_results=2,
        search_result_length=1000,
        bing_requests_per_second=2,
        search_cache_file=":memory:",
        bing_max_retries=2,
        bing_retry_delay=0,
        conda_path=None,
        conda_env=None,
        python_max_concurrent=1,
        max_python_times=0,
        max_search_times=3,
        sampling_params=None,
        sampling_params_nostop=None,
    )
    params.update(overrides)
    return SimpleNamespace(**params)


def inference(args=None, steps=None, cls=AsyncInference):
    # Replace only model/tokenizer setup. Execute production search resolution,
    # tool registration, get_processor, process_sample and the complete loop.
    engine = cls.__new__(cls)
    engine.args = args or arguments()
    engine.prompt_manager = PromptManager("search")
    engine.tokenizer = TokenizerFixture()
    engine.vllm_pool = ScriptedModel(steps or ["test query"])
    engine.tool_executor = engine._create_tool_executor()
    return engine


class ParallelSearchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = MCPFixture()
        self.fixture.__enter__()
        self.redirect = patch.object(parallel_search_tool, "MCP_URL", self.fixture.url)
        self.redirect.start()
        self.quiet = contextlib.redirect_stdout(io.StringIO())
        self.quiet.__enter__()
        self.stderr = contextlib.redirect_stderr(io.StringIO())
        self.stderr.__enter__()

    async def asyncTearDown(self):
        self.redirect.stop()
        self.stderr.__exit__(None, None, None)
        self.quiet.__exit__(None, None, None)
        await asyncio.to_thread(self.fixture.__exit__, None, None, None)

    async def test_no_key_resolver_to_model_consumption_and_identity(self):
        self.assertEqual(MCP_URL, "https://search.parallel.ai/mcp")
        with patch.dict(
            "os.environ",
            {"BING_API_KEY": "ambient", "PARALLEL_API_KEY": "ambient"},
            clear=True,
        ):
            engine = inference(
                steps=["first query", "https://example.org/reference", "followup query"]
            )
            self.assertIsInstance(
                engine.tool_executor.get_tool("search"), ParallelSearchTool
            )
            first, second = await asyncio.gather(
                engine.process_sample("first question", "42"),
                engine.process_sample("second question", "42"),
            )
            # Same question in a new research run must also get a new MCP ID.
            await engine.process_sample("first question", "42")
        self.assertEqual(first["prediction"], "42")
        self.assertIn("URL: https://example.org/reference", first["output"])
        self.assertIn("Parallel fact: 42.", second["output"])
        self.assertTrue(
            any("Parallel fact: 42." in prompt for prompt in engine.vllm_pool.prompts)
        )
        calls = self.fixture.calls()
        identities = {}
        for call in calls:
            sid = call["arguments"]["session_id"]
            identities.setdefault(sid, []).append(call["name"])
            if call["name"] == "web_fetch":
                self.assertEqual(
                    call["arguments"]["urls"], ["https://example.org/reference"]
                )
            else:
                self.assertEqual(
                    call["arguments"]["search_queries"],
                    [call["arguments"]["objective"]],
                )
        self.assertEqual(len(identities), 3)
        for sid, names in identities.items():
            self.assertEqual(len(sid), 36)
            self.assertEqual(names, ["web_search", "web_fetch", "web_search"])
        for path, headers, message in self.fixture.requests:
            self.assertEqual(path, "/mcp")
            lowered = {k.lower(): v for k, v in headers.items()}
            self.assertEqual(lowered["user-agent"], USER_AGENT)
            self.assertNotIn("authorization", lowered)
        self.assertTrue(
            {"initialize", "tools/list", "tools/call"}.issubset(
                {m["method"] for _, _, m in self.fixture.requests}
            )
        )

    async def test_completion_mode_consumes_results_and_respects_limits(self):
        engine = inference(
            arguments(max_search_times=1),
            steps=["first query", "second query"],
            cls=AsyncInferenceCompletion,
        )
        result = await engine.process_sample("bounded research", "42")
        self.assertEqual(result["prediction"], "42")
        self.assertIn("Parallel fact", result["output"])
        self.assertIn("maximum search limit", result["output"])
        self.assertEqual(len(self.fixture.calls()), 1)

    async def test_saved_bing_and_explicit_choices(self):
        engine = inference(arguments(bing_api_key="saved-key"))
        tool = engine.tool_executor.get_tool("search")
        self.assertIsInstance(tool, BingSearchTool)
        self.assertEqual(tool.name, "bing_search")
        self.assertEqual(tool._api_key, "saved-key")
        self.assertEqual(tool._zone, "saved_zone")
        self.assertEqual(tool._max_retries, 2)
        engine = inference(
            arguments(bing_api_key="saved-key", search_provider="parallel")
        )
        self.assertEqual(
            engine.tool_executor.get_tool("search").name, "parallel_search"
        )
        self.assertEqual(
            resolve_search_provider(arguments(bing_api_key="saved-key"), browser=True),
            "bing",
        )
        with self.assertRaises(ValueError):
            inference(arguments(search_provider="bing"))
        with self.assertRaises(ValueError):
            inference(arguments(), cls=AsyncInferenceCompletionSDS)
        with self.assertRaises(ValueError):
            inference(
                arguments(search_provider="parallel", bing_api_key="saved-key"),
                cls=AsyncInferenceCompletionSDS,
            )
        self.assertEqual(self.fixture.requests, [])

    async def test_cli_and_sds_saved_parameters(self):
        import infer

        command = [
            "infer.py",
            "--infer_mode",
            "completion",
            "--endpoints",
            "http://localhost:8000/v1",
            "--model_path",
            "/model",
        ]
        with patch.object(sys, "argv", command):
            args = infer.parse_arguments()
        self.assertIsNone(args.bing_api_key)
        self.assertEqual(resolve_search_provider(args), "parallel")
        for choice in ("bing", "parallel", "none"):
            with patch.object(sys, "argv", command + ["--search_provider", choice]):
                self.assertEqual(infer.parse_arguments().search_provider, choice)
        args = arguments(
            bing_api_key="saved-key",
            url_cache_file="saved-url.db",
            summ_model_path="/saved-model",
            summ_model_urls=["http://saved-model/v1"],
            summ_model_name="saved-model",
        )
        browser_tool = Mock(trigger_tag="search")
        constructor = Mock(return_value=browser_tool)
        # The SDS model/browser implementation is an existing optional dependency;
        # verify its exact constructor parameters without loading model weights.
        with patch.dict(
            sys.modules,
            {
                "src.tools.search_tool_sds": SimpleNamespace(
                    BingSearchToolSDS=constructor
                )
            },
        ):
            engine = inference(args, cls=AsyncInferenceCompletionSDS)
        self.assertIs(engine.tool_executor.get_tool("search"), browser_tool)
        self.assertEqual(
            constructor.call_args.kwargs,
            {
                "api_key": "saved-key",
                "zone": "saved_zone",
                "max_results": 2,
                "result_length": 1000,
                "requests_per_second": 2,
                "search_cache_file": ":memory:",
                "max_retries": 2,
                "retry_delay": 0,
                "url_cache_file": "saved-url.db",
                "summ_model_path": "/saved-model",
                "summ_model_urls": ["http://saved-model/v1"],
                "summ_model_name": "saved-model",
            },
        )

    async def test_disable_and_existing_zero_call_limit(self):
        engine = inference(arguments(search_provider="none", bing_api_key="saved-key"))
        self.assertIsNone(engine.tool_executor.get_tool("search"))
        await engine.process_sample("no permission to search", "42")
        engine = inference(arguments(max_search_times=0))
        await engine.process_sample("zero search calls", "42")
        self.assertEqual(self.fixture.requests, [])

    async def test_errors_quota_and_retry_keep_conversation(self):
        engine = inference()
        stat = {"input": "retry question", "output": "", "logs": []}
        processor = engine.get_processor(stat, None)
        sid = processor.search_session_id
        for mode in ("error", "quota", "bad_payload", "fetch_error", "http_error"):
            self.fixture.mode = mode
            await processor.call_search("test query")
            self.assertIn("Tool execute failed", stat["logs"][-1])
            self.assertEqual(self.fixture.calls()[-1]["arguments"]["session_id"], sid)
        self.fixture.mode = "normal"
        await processor.call_search("test query")
        self.assertIn("Parallel fact", stat["logs"][-1])
        self.assertEqual(self.fixture.calls()[-1]["arguments"]["session_id"], sid)
        self.assertEqual(
            len(self.fixture.calls()), 6
        )  # No hidden retries/provider fallback.

    async def test_timeout_and_cancellation(self):
        self.fixture.mode = "slow"
        engine = inference()
        tool = engine.tool_executor.get_tool("search")
        processor = engine.get_processor(
            {"input": "slow question", "output": "", "logs": []}, None
        )
        timeout_task = asyncio.create_task(
            engine.tool_executor.execute(
                "search",
                "slow query",
                timeout=1,
                session_id=processor.search_session_id,
            )
        )
        self.assertTrue(await asyncio.to_thread(self.fixture.call_event.wait, 3))
        result = await timeout_task
        self.assertIn("Tool execute failed", result)
        self.fixture.call_event.clear()
        task = asyncio.create_task(
            tool.execute(
                "slow query", timeout=10, session_id=processor.search_session_id
            )
        )
        self.assertTrue(await asyncio.to_thread(self.fixture.call_event.wait, 3))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.fixture.mode = "normal"
        await processor.call_search("retry after cancellation")
        self.assertIn("Parallel fact", processor.sample_stat["logs"][-1])
        self.assertEqual(
            {c["arguments"]["session_id"] for c in self.fixture.calls()},
            {processor.search_session_id},
        )

    async def test_output_bounds_text_payload_and_invalid_urls(self):
        engine = inference(arguments(search_max_results=1, search_result_length=8))
        tool = engine.tool_executor.get_tool("search")
        self.fixture.mode = "text_only"
        result = await tool.execute("test query", session_id="x" * 36)
        self.assertIn("URL: https://example.org/reference", result)
        self.assertIn("Parallel", result)
        self.assertNotIn("fact", result)
        self.assertNotIn("Page 2", result)
        count = len(self.fixture.requests)
        for url in (
            "https://user:secret@example.org",
            "https://",
            "https://example.org one two",
        ):
            with self.assertRaises(ValueError):
                await tool.execute(url, session_id="x" * 36)
        self.assertEqual(len(self.fixture.requests), count)
        self.fixture.mode = "empty"
        self.assertEqual(
            await tool.execute("empty query", session_id="x" * 36),
            "No search results found.",
        )


if __name__ == "__main__":
    unittest.main()
