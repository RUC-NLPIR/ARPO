"""Anonymous web search and URL extraction through Parallel's HTTP MCP server."""

import asyncio
import json
from urllib.parse import urlsplit

from .base_tool import BaseTool

MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "ARPO/ParallelSearch (https://github.com/RUC-NLPIR/ARPO)"


class ParallelSearchTool(BaseTool):
    """Use the existing <search> tag for queries or a single HTTP(S) URL.

    The processor supplies a conversation ID; the shared tool owns no per-sample
    state. A fresh MCP connection can therefore serve each call without changing
    the conversation's identity or mixing concurrent samples.
    """

    def __init__(self, max_results=10, result_length=1000):
        if max_results < 1 or result_length < 1:
            raise ValueError("Search result limits must be positive")
        self.max_results = max_results
        self.result_length = result_length

    @property
    def name(self):
        return "parallel_search"

    @property
    def trigger_tag(self):
        return "search"

    async def execute(self, query, timeout=60, *, session_id=None, **kwargs):
        if not isinstance(session_id, str) or not 1 <= len(session_id) <= 100:
            raise ValueError("Parallel search requires the current sample's session_id")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query must not be empty")
        query = query.strip()
        arguments = {"session_id": session_id}
        if query.lower().startswith(("http:", "https:")):
            if not self._valid_url(query):
                raise ValueError("Fetch requires a single valid HTTP(S) URL")
            tool_name = "web_fetch"
            arguments["urls"] = [query]
        else:
            tool_name = "web_search"
            arguments.update(objective=query, search_queries=[query])
        return await asyncio.wait_for(
            self._request(tool_name, arguments, timeout), timeout
        )

    async def _request(self, tool_name, arguments, timeout):
        # Import only when selected. Existing Bing installations need no MCP SDK.
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(
            MCP_URL,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            sse_read_timeout=timeout,
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                if tool_name not in {tool.name for tool in tools.tools}:
                    raise RuntimeError(
                        f"Parallel Search MCP does not offer {tool_name}"
                    )
                result = await session.call_tool(tool_name, arguments)
        if result.isError:
            detail = " ".join(
                block.text for block in result.content if block.type == "text"
            )
            raise RuntimeError(f"Parallel Search MCP: {detail}")
        data = result.structuredContent
        if data is None:
            text = "\n".join(
                block.text for block in result.content if block.type == "text"
            )
            data = json.loads(text)
        return self._format_results(data)

    @staticmethod
    def _valid_url(url):
        try:
            parsed = urlsplit(url)
            return (
                parsed.scheme in ("http", "https")
                and bool(parsed.hostname)
                and not parsed.username
                and not parsed.password
                and not any(char.isspace() for char in url)
            )
        except ValueError:
            return False

    def _format_results(self, data):
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise ValueError("Parallel Search MCP returned an invalid results payload")
        pages = []
        for result in data["results"][: self.max_results]:
            if not isinstance(result, dict):
                raise ValueError("Parallel Search MCP returned an invalid result")
            url = result.get("url", "")
            if not isinstance(url, str) or not self._valid_url(url):
                continue
            excerpts = result.get("excerpts") or []
            if not isinstance(excerpts, list) or any(
                not isinstance(item, str) for item in excerpts
            ):
                raise ValueError("Parallel Search MCP returned invalid excerpts")
            snippet = "\n".join(excerpts)[: self.result_length]
            title = result.get("title") or url
            pages.append(f"Page {len(pages) + 1}: {title}\nURL: {url}\n{snippet}")
        if not pages:
            errors = data.get("errors")
            if errors:
                raise RuntimeError(
                    f"Parallel Search MCP could not retrieve results: {errors}"
                )
            return "No search results found."
        return "\n\n".join(pages)
