import sys
import os

sys.path.append(os.getcwd())
import time
import langid
import asyncio
import requests
import aiolimiter
from typing import Union, Dict
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor

from .base_tool import BaseTool
from .cache_manager import PreprocessCacheManager


class BingSearchTool(BaseTool):
    """BingSearchTool"""

    _executor = ThreadPoolExecutor(max_workers=os.cpu_count() * 2)

    def __init__(
        self,
        api_key: str,
        zone: str = "your_zone",
        max_results: int = 10,
        result_length: int = 1000,
        location: str = "cn",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        requests_per_second: float = 2.0,
        search_cache_file="",
    ):
        self._api_key = api_key
        self._zone = zone
        self._max_results = max_results
        self._result_length = result_length
        self._location = location
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        self._limiter = aiolimiter.AsyncLimiter(
            max_rate=requests_per_second, time_period=1.0
        )
        self.search_cache_manager = PreprocessCacheManager(search_cache_file)

    @property
    def name(self) -> str:
        return "bing_search"

    @property
    def trigger_tag(self) -> str:
        return "search"

    def _call_request(self, query, headers, payload, timeout):
        error_cnt = 0
        while True:
            if error_cnt >= self._max_retries:
                print(
                    f"query: {query} has tried {error_cnt} times without success, just skip it."
                )
                break
            try:
                response = requests.post(
                    "https://api.brightdata.com/request",
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                search_results = response.json()
                return search_results
            except requests.exceptions.Timeout:
                error_cnt += 1
                print(
                    f"error_cnt: {error_cnt}, Bing Web Search request timed out ({timeout} seconds) for query: {query}"
                )
                time.sleep(5)
            except requests.exceptions.RequestException as e:
                error_cnt += 1
                print(
                    f"error_cnt: {error_cnt}, Error occurred during Bing Web Search request: {e}, payload: {payload}"
                )
                time.sleep(5)
        return None

    def _pack_query(self, query):
        if langid.classify(query)[0] == "zh":
            mkt, setLang = "zh-CN", "zh"
        else:
            mkt, setLang = "en-US", "en"
        input_obj = {"q": query, "mkt": mkt, "setLang": setLang}
        encoded_query = urlencode(input_obj)
        return encoded_query

    def _make_request(self, query: str, timeout: int):
        """
        Send a request to the Brightdata API.

        Args:
            query: The search query.
            timeout: Request timeout in seconds.
            
        Returns:
            The response object.
        """
        encoded_query = self._pack_query(query)
        target_url = f"https://www.bing.com/search?{encoded_query}&brd_json=1&cc={self._location}"

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {"zone": self._zone, "url": target_url, "format": "raw"}
        result = self._call_request(query, headers, payload, timeout)
        return result

    async def postprocess_search_result(self, query, response, **kwargs) -> str:
        data = response
        if "organic" not in data:
            data["chunk_content"] = []
            result = self._format_results(data)
        else:
            chunk_content_list = []
            seen_snippets = set()
            for result_item in data["organic"]:
                snippet = result_item.get("description", "").strip()
                if snippet and snippet not in seen_snippets:
                    chunk_content_list.append(snippet)
                    seen_snippets.add(snippet)
            data["chunk_content"] = chunk_content_list
            result = self._format_results(data)
        return result

    async def execute(self, query: str, timeout: int = 60, **kwargs) -> str:
        """
        Execute a Bing search query with support for cache and semantic similarity cache hits.

        Args:
            query: The search query text.
            timeout: Request timeout in seconds.
            model: SBERT model used for semantic search.
            threshold: Minimum similarity threshold.
            top_k: Number of top most similar cached entries to consider.

        Returns:
            A string containing the search result.
        """
        hit_cache = self.search_cache_manager.hit_cache(query)
        if hit_cache:
            print("hit cache: ", query)
            response = hit_cache
        else:
            loop = asyncio.get_event_loop()

            async with self._limiter:
                response = await loop.run_in_executor(
                    self._executor, lambda: self._make_request(query, timeout)
                )
            if response is None:
                return f"Bing search failed: {query}"
            await self.search_cache_manager.add_to_cache(query, response)
        assert response is not None
        return await self.postprocess_search_result(query, response, **kwargs)

    def _format_results(self, results: Dict) -> Union[str, None]:
        """Format the search results."""
        if not results.get("chunk_content"):
            return None

        formatted = []
        for idx, snippet in enumerate(results["chunk_content"][: self._max_results], 1):
            snippet = snippet[: self._result_length]
            formatted.append(f"Page {idx}: {snippet}")
        return "\n".join(formatted)


class YouSearchTool(BingSearchTool):
    """You.com search with the same interface as BingSearchTool.

    ``api_key`` is optional: leave it empty to use the keyless free tier.
    """

    def __init__(self, api_key: str = "", **kwargs):
        kwargs.pop("zone", None)
        super().__init__(api_key=api_key, **kwargs)

    @property
    def name(self) -> str:
        return "you_search"

    def _make_request(self, query: str, timeout: int):
        from .you_search import you_search_request

        return you_search_request(
            query,
            count=self._max_results,
            api_key=self._api_key or None,
            timeout=timeout,
            max_retries=self._max_retries,
            retry_delay=self._retry_delay,
        )
