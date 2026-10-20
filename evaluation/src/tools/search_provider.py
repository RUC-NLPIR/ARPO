"""Resolve evaluation search without overriding configured providers."""


def resolve_search_provider(args, *, browser=False):
    provider = getattr(args, "search_provider", "auto")
    if provider == "none":
        return provider
    if provider not in ("auto", "bing", "parallel"):
        raise ValueError(f"Unknown search provider: {provider}")
    if provider == "auto":
        provider = (
            "bing" if browser or getattr(args, "bing_api_key", None) else "parallel"
        )
    if browser and provider != "bing":
        raise ValueError(
            "completion_sds requires Bing; use default/completion for Parallel"
        )
    if provider == "bing" and not getattr(args, "bing_api_key", None):
        raise ValueError("Bing search requires --bing_api_key")
    return provider


def create_search_tool(args, *, browser=False):
    provider = resolve_search_provider(args, browser=browser)
    if provider == "none":
        return None
    if provider == "parallel":
        from .parallel_search_tool import ParallelSearchTool

        return ParallelSearchTool(
            max_results=args.search_max_results,
            result_length=args.search_result_length,
        )

    from .search_tool import BingSearchTool

    params = dict(
        api_key=args.bing_api_key,
        zone=args.bing_zone,
        max_results=args.search_max_results,
        result_length=args.search_result_length,
        requests_per_second=args.bing_requests_per_second,
        search_cache_file=args.search_cache_file,
        max_retries=args.bing_max_retries,
        retry_delay=args.bing_retry_delay,
    )
    if browser:
        from .search_tool_sds import BingSearchToolSDS

        params.update(
            url_cache_file=args.url_cache_file,
            summ_model_path=args.summ_model_path,
            summ_model_urls=args.summ_model_urls,
            summ_model_name=args.summ_model_name,
        )
        return BingSearchToolSDS(**params)
    return BingSearchTool(**params)
