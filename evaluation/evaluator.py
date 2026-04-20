# evaluation/evaluator.py
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional, Union
from tqdm import tqdm
import json
from datetime import datetime
from pathlib import Path
import time

from evaluation.benchmarks import BaseBenchmark, DataSetItem
from evaluation.benchmark_registry import create_benchmark, get_benchmarks


class Evaluator:
    def __init__(
            self,
            inference_fn: Callable[[str], str],
            workers: int = 1,
            benchmarks: List[str] = None,
            output_dir: str = "./results",
            verbose: bool = False,
            limit: int = None,
            offset: int = 0,
            job_id: Optional[str] = None,
            array_task_id: Optional[int] = None
        ):
        """
        Docstring for __init__

        :param self: Description
        :param inference_fn: Function to get model output given a prompt
        :type inference_fn: Callable[[str], str]
        :param workers: Number of workers to use for evaluation
        :type workers: int
        :param benchmarks: List of benchmark names to evaluate (if None, all available benchmarks are used)
        :type benchmarks: List[str]
        :param job_id: Optional SLURM job ID for organizing results
        :type job_id: Optional[str]
        :param array_task_id: Optional SLURM array task ID
        :type array_task_id: Optional[int]
        """
        self.inference_fn = inference_fn
        self.workers = workers
        self.benchmarks = benchmarks or get_benchmarks()
        self.verbose = verbose
        self.limit = limit
        self.offset = offset
        self.job_id = job_id
        self.array_task_id = array_task_id

        # Create job-specific directory if job_id provided
        if job_id:
            self.output_dir = Path(output_dir) / f"job_{job_id}"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


    def evaluate(self):
        """Evaluate all benchmarks."""
        if self.verbose:
            print("="*60)
            print("Starting Evaluation")
            print("="*60)
            print(f"Benchmarks: {', '.join(self.benchmarks)}")
            print(f"Workers: {self.workers}")
            print(f"Output directory: {self.output_dir}")
            print("="*60)

        
        all_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "workers": self.workers,
                "benchmarks": self.benchmarks,
            },
            "results": {}
        }

        for benchmark_name in self.benchmarks:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Evaluating: {benchmark_name}")
                print(f"{'='*60}")
            benchmark = create_benchmark(benchmark_name)
            dataset = benchmark.load_dataset()
            if self.limit is not None:
                start = self.offset
                end = min(self.offset + self.limit, len(dataset))
                dataset = dataset[start:end]
                print(f"Limiting evaluation to the first {self.limit} samples.")

            if self.verbose:
                print(f"Loaded {len(dataset)} samples")
            
            results = self._evaluate_benchmark(benchmark_name, benchmark, dataset)


            all_results['results'][benchmark_name] = results

            if self.verbose:
                print(f"\n{benchmark_name} Results:")
                for metric, value in results.get("metrics", {}).items():
                    print(f"  {metric}: {value:.4f}")
            
        # Save results to file
        self._save_results(all_results)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("Evaluation Complete!")
            print(f"{'='*60}")
        return all_results
    
    def _evaluate_benchmark(self, benchmark_name: str, benchmark: BaseBenchmark, dataset: List[DataSetItem]) -> Dict[str, Any]:
        """Evaluate a single benchmark with parallel workers."""
        batch_start_time = time.time()

        # with batching
        if getattr(self.inference_fn, "is_batch", False):
            print(f"\n[vLLM Engine] Running batch inference on {len(dataset)} items...")

            # format all prompts and send to vLLM at once
            prompts = [benchmark.get_user_prompt(item.input) for item in dataset]
            batch_outputs = self.inference_fn(prompts)

            # temporarily trick _evaluate_item into pulling from batch_outputs
            original_fn = self.inference_fn
            output_iterator = iter(batch_outputs)
            self.inference_fn = lambda p: next(output_iterator)

            # run sequentially to calculate scores
            results = [
                self._evaluate_item(benchmark, item)
                for item in tqdm(dataset, desc=f"{benchmark_name} (Scoring)")
            ]

            # restore original function
            self.inference_fn = original_fn

        # without batching (original pipeline)
        else:
            if self.workers == 1:
                results = [
                    self._evaluate_item(benchmark, item)
                    for item in tqdm(dataset, desc=f"{benchmark_name}")
                ]
            else:
                results = []
                with ThreadPoolExecutor(max_workers=self.workers) as executor:
                    futures = [
                        executor.submit(self._evaluate_item, benchmark, item)
                        for item in dataset
                    ]

                    for future in tqdm(as_completed(futures), total=len(dataset), desc=f"{benchmark_name}"):
                        result = future.result()
                        results.append(result)

        batch_end_time = time.time()
        batch_duration_s = batch_end_time - batch_start_time

        # Calculate metrics including latency stats
        scores = [r["score"] for r in results]
        latencies = [r.get("metrics", {}).get("latency_ms", 0) for r in results if "metrics" in r]

        avg_score = sum(scores) / len(scores) if scores else 0.0

        metrics = {
            "average_score": avg_score,
            "total_correct": sum(1 for s in scores if s == 1.0),
            "accuracy": sum(1 for s in scores if s == 1.0) / len(scores) if scores else 0.0
        }

        # Add latency metrics if available
        if latencies:
            metrics["avg_latency_ms"] = sum(latencies) / len(latencies)
            metrics["min_latency_ms"] = min(latencies)
            metrics["max_latency_ms"] = max(latencies)
            # Calculate p95 latency
            sorted_latencies = sorted(latencies)
            p95_idx = int(len(sorted_latencies) * 0.95)
            metrics["p95_latency_ms"] = sorted_latencies[p95_idx] if sorted_latencies else 0

        # Add batch-level metadata
        batch_metadata = {
            "batch_duration_s": batch_duration_s,
            "batch_size": len(dataset),
            "is_batch_inference": getattr(self.inference_fn, "is_batch", False),
            "offset": self.offset,
            "limit": self.limit
        }

        return {
            "num_samples": len(dataset),
            "metrics": metrics,
            "batch_metadata": batch_metadata,
            "results": results[:100]  # Keep only first 100 for file size
        }
    
    def _call_inference_with_metrics(self, prompt: str) -> Dict[str, Any]:
        """
        Call inference function and collect metrics.
        Supports both str and dict returns (hybrid approach).
        """
        start_time = time.time()

        # Call the inference function
        output = self.inference_fn(prompt)

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        # Check if output is dict with metrics (advanced) or just str (basic)
        if isinstance(output, dict):
            # Advanced: inference function provided metrics
            text = output.get("text", "")
            metrics = output.get("metrics", {})
            # Ensure latency is present
            if "latency_ms" not in metrics:
                metrics["latency_ms"] = latency_ms
        else:
            # Basic: just a string, create metrics from wrapper
            text = output
            metrics = {
                "latency_ms": latency_ms,
                # Token counting would require tokenizer - skip for now
                # Will be added by models that have tokenizer access
            }

        return {
            "text": text,
            "metrics": metrics
        }

    def _evaluate_item(self, benchmark, item: DataSetItem) -> Dict[str, Any]:
        """Evaluate a single item (can run in parallel)."""
        input_data = item.input

        prompt = benchmark.get_user_prompt(input_data)

        # Call inference with metrics tracking
        output_with_metrics = self._call_inference_with_metrics(prompt)
        at_output = output_with_metrics["text"]
        metrics = output_with_metrics["metrics"]

        parsed_at_output = benchmark.parse_output(at_output)
        score = benchmark.score(item, parsed_at_output)

        return {
            "id": item.id,
            "input": input_data,
            "full_output": at_output,
            "prediction": parsed_at_output,
            "expected": item.output,
            "metadata": item.metadata,
            "score": score,
            "metrics": metrics  # Add metrics to result
        }
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Include array task ID in filename if available
        if self.array_task_id is not None:
            filename = f"eval_results_array_{self.array_task_id}.json"
        else:
            filename = f"eval_results_{timestamp}.json"

        filepath = self.output_dir / filename

        # Add job metadata to results
        results["metadata"]["job_id"] = self.job_id
        results["metadata"]["array_task_id"] = self.array_task_id
        results["metadata"]["offset"] = self.offset
        results["metadata"]["limit"] = self.limit

        # Custom JSON encoder to handle dataclasses and other non-serializable objects
        def default_serializer(obj):
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=default_serializer)

        print(f"\nResults saved to: {filepath}")

