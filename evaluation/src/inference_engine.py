import asyncio
import time
import json
import os

from vllm import SamplingParams
from transformers import AutoTokenizer
from typing import Dict, Any, Optional
from tqdm.asyncio import tqdm as async_tqdm

from .prompt_manager import PromptManager
from .data_loader import DataLoader
from .tools.tool_executor import ToolExecutor
from .tools import PythonTool, BingSearchTool, BingSearchToolSDS, YouSearchTool, YouSearchToolSDS
from .vllm_client_pool import VLLMClientPool
from .sample_processor import SampleProcessor, SampleProcessorCompletion


class AsyncInference:

    def __init__(self, args):
        self.args = args
        self.vllm_pool = VLLMClientPool(
            endpoints=args.endpoints,
            api_keys=args.api_keys,
            default_model=args.default_model,
        )
        self.prompt_manager = PromptManager(args.prompt_type)
        self.tool_executor = self._create_tool_executor()
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True
        )
        self.data_loaders = [
            DataLoader(dataset_name, args.dataset_path)
            for dataset_name in args.dataset_name
        ]
        self.args.sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            n=1,
            include_stop_str_in_output=args.include_stop_str_in_output,
        )
        self.args.sampling_params_nostop = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            n=1,
            include_stop_str_in_output=args.include_stop_str_in_output,
            stop=None,
        )
        self.sample_timeout = getattr(args, "sample_timeout", 240)
        print(f"Initialized {self.__class__.__name__}...")

    def _create_tool_executor(self):
        tool_executor = ToolExecutor()
        python_tool = PythonTool(
            conda_path=self.args.conda_path,
            conda_env=self.args.conda_env,
            max_concurrent=self.args.python_max_concurrent,
        )
        tool_executor.register_tool(python_tool)
        use_you = getattr(self.args, "search_provider", "bing") == "you"
        search_cls = YouSearchTool if use_you else BingSearchTool
        search_tool = search_cls(
            api_key=self.args.ydc_api_key if use_you else self.args.bing_api_key,
            zone=self.args.bing_zone,
            max_results=self.args.search_max_results,
            result_length=self.args.search_result_length,
            requests_per_second=self.args.bing_requests_per_second,
            search_cache_file=self.args.search_cache_file,
            max_retries=self.args.bing_max_retries,
            retry_delay=self.args.bing_retry_delay,
        )
        tool_executor.register_tool(search_tool)
        return tool_executor

    def get_processor(self, sample_stat, session_id):
        processor = SampleProcessor(
            self.prompt_manager,
            self.tool_executor,
            self.vllm_pool,
            self.tokenizer,
            self.args,
            sample_stat,
            session_id,
        )
        return processor

    async def process_sample(
        self, question: str, golden_answer: str, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        sample_stat = {
            "instruction": self.prompt_manager.get_system_prompt(),
            "input": question,
            "output": "",
            "prediction": "",
            "answer": golden_answer,
            "logs": [],
            "search_query_history": set(),
        }

        current_task = asyncio.current_task()
        if current_task:
            setattr(current_task, "_current_result", sample_stat)

        processor = self.get_processor(sample_stat, session_id)
        await processor.run()
        processor.log_timing()
        sample_stat = processor.sample_stat
        return sample_stat

    async def process_sample_wrap(self, idx, question, answer):
        try:
            process_task = asyncio.create_task(self.process_sample(question, answer))
            try:
                result = await asyncio.wait_for(
                    process_task,
                    timeout=self.sample_timeout,
                )
                print(f"Finished to process sample {idx}")
            except asyncio.TimeoutError:
                process_task.cancel()
                partial_result = getattr(process_task, "_current_result", None)
                if partial_result:
                    partial_output = partial_result.get("output", "Timeout")
                    partial_prediction = partial_result.get("prediction", "Timeout")
                    print(f"Sample timeout ({self.sample_timeout}): '{question}'")
                else:
                    partial_output = "Timeout"
                    partial_prediction = "Timeout"
                    print(
                        f"Sample timeout (exceeding {self.sample_timeout}s): '{question}'"
                    )
                result = {
                    "instruction": self.prompt_manager.get_system_prompt(),
                    "input": question,
                    "output": f"{partial_output}\n[Timeout (exceeding {self.sample_timeout}s)]",
                    "prediction": partial_prediction or "Timeout",
                    "answer": answer,
                    "logs": [],
                    "search_query_history": set(),
                }
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Sample {idx} error: {str(e)}")
            result = {
                "instruction": self.prompt_manager.get_system_prompt(),
                "input": question,
                "output": f"Error: {str(e)}",
                "prediction": f"Error: {str(e)}",
                "answer": answer,
                "logs": [],
                "search_query_history": set(),
            }
        result["search_query_history"] = list(result["search_query_history"])
        return result

    async def run_inference(self, question):
        return await self.process_sample_wrap(question, question, None)

    async def task_worker(self, task_queue, questions, answers, results):
        while not task_queue.empty():
            try:
                idx = await task_queue.get()
                results[idx] = await self.process_sample_wrap(
                    idx, questions[idx], answers[idx]
                )
            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"Worker error: {str(e)}")
            task_queue.task_done()

    async def run(self):
        for dataloader in self.data_loaders:
            dataset_name = dataloader.dataset_name
            print(
                ">>> Inference dataset: ",
            )
            questions, answers = dataloader.load_data()
            total_examples = min(len(questions), self.args.counts)
            questions = questions[:total_examples]
            answers = answers[:total_examples]
            print(
                f"Total examples: {total_examples}, Max concurrent requests: {self.args.max_concurrent_requests}"
            )
            for turn in self.args.turns:
                print(f">>> Turn {turn}: {dataloader.dataset_name}")
                results = [None] * total_examples
                start_time = time.time()
                task_queue = asyncio.Queue()
                for i in range(total_examples):
                    await task_queue.put(i)
                workers = []
                for _ in range(min(self.args.max_concurrent_requests, total_examples)):
                    workers.append(
                        asyncio.create_task(
                            self.task_worker(task_queue, questions, answers, results)
                        )
                    )
                pbar = async_tqdm(total=total_examples, desc="Processing samples")
                processed = 0
                while processed < total_examples:
                    completed = sum(1 for r in results if r is not None)
                    if completed > processed:
                        pbar.update(completed - processed)
                        processed = completed
                    await asyncio.sleep(0.1)
                pbar.close()
                await task_queue.join()
                for w in workers:
                    w.cancel()
                await asyncio.gather(*workers, return_exceptions=True)
                end_time = time.time()
                print(f"Total Time: {(end_time - start_time) / 60:.2f}min")
                os.makedirs(
                    os.path.join(self.args.output_path, dataset_name), exist_ok=True
                )
                output_file = os.path.join(
                    self.args.output_path,
                    dataset_name,
                    f"{dataset_name}_output_{turn}.json",
                )
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
                print(
                    f"Processed {dataset_name} (Turn {turn}), save results to {output_file}"
                )
        print("Finished to process all datasets!")


class AsyncInferenceCompletion(AsyncInference):

    def get_processor(self, sample_stat, session_id):
        processor = SampleProcessorCompletion(
            self.prompt_manager,
            self.tool_executor,
            self.vllm_pool,
            self.tokenizer,
            self.args,
            sample_stat,
            session_id,
        )
        return processor


class AsyncInferenceCompletionSDS(AsyncInferenceCompletion):
    """using Web browser from SimpleDeepSearcher"""

    def _create_tool_executor(self):
        tool_executor = ToolExecutor()
        python_tool = PythonTool(
            conda_path=self.args.conda_path,
            conda_env=self.args.conda_env,
            max_concurrent=self.args.python_max_concurrent,
        )
        tool_executor.register_tool(python_tool)
        use_you = getattr(self.args, "search_provider", "bing") == "you"
        search_cls = YouSearchToolSDS if use_you else BingSearchToolSDS
        search_tool = search_cls(
            api_key=self.args.ydc_api_key if use_you else self.args.bing_api_key,
            zone=self.args.bing_zone,
            max_results=self.args.search_max_results,
            result_length=self.args.search_result_length,
            requests_per_second=self.args.bing_requests_per_second,
            max_retries=self.args.bing_max_retries,
            retry_delay=self.args.bing_retry_delay,
            search_cache_file=self.args.search_cache_file,
            url_cache_file=self.args.url_cache_file,
            summ_model_path=self.args.summ_model_path,
            summ_model_urls=self.args.summ_model_urls,
            summ_model_name=self.args.summ_model_name,
        )
        tool_executor.register_tool(search_tool)
        return tool_executor
