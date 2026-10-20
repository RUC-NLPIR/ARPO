"""Load provider dependencies only when that provider is selected."""

from importlib import import_module

_MODULES = {
    "BaseTool": "base_tool",
    "PythonTool": "python_tool",
    "BingSearchTool": "search_tool",
    "BingSearchToolSDS": "search_tool_sds",
    "BingSearchToolSDScn": "search_tool_sds_cn",
    "ParallelSearchTool": "parallel_search_tool",
    "ToolExecutor": "tool_executor",
}
__all__ = list(_MODULES)


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(f".{_MODULES[name]}", __name__), name)
    globals()[name] = value
    return value
