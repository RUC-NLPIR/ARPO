# Evaluation search providers

The English evaluation entrypoint (`infer.py`) supports free web search and URL
extraction with [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp).
Anonymous search uses Fast mode and needs no account or API key. This changes the
search backend for new, unconfigured `default` and `completion` runs. Changing
search providers can change evaluation results; select Bing explicitly when
reproducing an experiment that used Bing.

In your existing evaluation environment (Python 3.10 or newer), install the MCP
client dependencies:

```bash
cd evaluation
python -m pip install -r requirements-search.txt
```

The full evaluation environment is listed in `requirement.txt` (singular). The
MCP requirements retain compatibility with its Pydantic 2.10 pin. Model serving,
model/tokenizer paths, data and the Python interpreter setup still follow the
repository's evaluation instructions.

For example, with a running model server and the appropriate local model and
dataset paths:

```bash
python infer.py --infer_mode completion \
  --endpoints http://localhost:8000/v1 --model_path /path/to/model \
  --data_path ./data --dataset_name hotpotqa --output_path ./outputs \
  --conda_path /path/to/conda --conda_env evaluation \
  --search_provider parallel
```

`--search_provider` selects the backend:

| Value | Behavior |
| --- | --- |
| `auto` (default) | Uses Bing when `--bing_api_key` is supplied; otherwise uses Parallel in `default` and `completion`. |
| `parallel` | Uses anonymous Parallel Search MCP, even if a Bing key is supplied. |
| `bing` | Uses the existing Bright Data/Bing tool; requires `--bing_api_key`. Existing zone, retry, rate and cache settings apply. |
| `none` | Does not register a search tool. |

`completion_sds` retains its Bing plus local summarization pipeline and requires a
Bing key unless search is disabled. It rejects `parallel`; choose `default` or
`completion` to use Parallel. Existing scripts that pass a Bing key continue to
use Bing. The Chinese entrypoint and training search configuration are unchanged.
The integration does not read provider keys from the environment, and it does not
switch providers after an error.

The existing `<search>query</search>` calls return titles, source URLs and excerpts
inside `<result>`. A single HTTP(S) URL inside `<search>` invokes MCP `web_fetch`
instead. This provides extraction without the SDS summarization pipeline. Output
is bounded by `--search_max_results` and `--search_result_length` (excerpt characters
per page). Existing search-call limits, including `--max_search_times 0`, still
apply. No additional model tool tag is needed.

Each research sample gets one opaque MCP conversation ID, reused for its search,
fetch and follow-up calls. Independent or concurrent samples get different IDs.
Calls honor the processor timeout and task cancellation. Service errors are
returned through the existing tool error path; anonymous usage has service-managed
limits and is intended for exploration and light use.

The provider is loaded only when selected. The deterministic suite exercises the
real resolver, dispatcher and sample processor with scripted model output and a
local HTTP MCP fixture. It does not load model weights or measure model quality:

```bash
python -m pip install -r tests/requirements.txt
python -m unittest discover -s tests -v
```
