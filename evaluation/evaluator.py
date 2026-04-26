# evaluation/evaluator.py
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional
from tqdm import tqdm
import json
import time
import statistics
from datetime import datetime
from pathlib import Path

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
            offset: int = 0
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
        """
        self.inference_fn = inference_fn
        self.workers = workers
        self.benchmarks = benchmarks or get_benchmarks()
        self.verbose = verbose
        self.limit = limit
        self.offset = offset

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
                metrics = results.get("metrics", {})

                print("  Accuracy Metrics:")
                for metric in ["average_score", "total_correct", "accuracy"]:
                    if metric in metrics:
                        value = metrics[metric]
                        if isinstance(value, float):
                            print(f"    {metric}: {value:.4f}")
                        else:
                            print(f"    {metric}: {value}")

                print("  Latency Metrics:")
                for metric, value in metrics.items():
                    if metric.startswith("latency") or metric == "throughput_items_per_sec":
                        print(f"    {metric}: {value:.4f}")
            
        # Save results to file
        self._save_results(all_results)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("Evaluation Complete!")
            print(f"{'='*60}")
        return all_results
    
    def _evaluate_benchmark(self, benchmark_name: str, benchmark: BaseBenchmark, dataset: List[DataSetItem]) -> Dict[str, Any]:
        """Evaluate a single benchmark with parallel workers."""
        # with batching
        if getattr(self.inference_fn, "is_batch", False):
            print(f"\n[vLLM Engine] Running batch inference on {len(dataset)} items...")

            # format all prompts and send to vLLM at once
            prompts = [benchmark.get_user_prompt(item.input) for item in dataset]

            batch_start_time = time.perf_counter()
            batch_outputs = self.inference_fn(prompts)
            batch_end_time = time.perf_counter()
            batch_latency = batch_end_time - batch_start_time

            # calculate per-item latency (approximate for batch)
            per_item_latency = batch_latency / len(dataset) if dataset else 0.0

            # temporarily trick _evaluate_item into pulling from batch_outputs
            original_fn = self.inference_fn
            output_iterator = iter(batch_outputs)
            self.inference_fn = lambda p: next(output_iterator)

            # run sequentially to calculate scores
            results = []
            for item in tqdm(dataset, desc=f"{benchmark_name} (Scoring)"):
                result = self._evaluate_item(benchmark, item)
                result["latency_seconds"] = per_item_latency
                results.append(result)

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
        
        scores = [r["score"] for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Calculate latency metrics
        latencies = [r.get("latency_seconds", 0.0) for r in results]
        latency_metrics = self._calculate_latency_metrics(latencies)

        return {
            "num_samples": len(dataset),
            "metrics": {
                "average_score": avg_score,
                "total_correct": sum(1 for s in scores if s == 1.0),
                "accuracy": sum(1 for s in scores if s == 1.0) / len(scores) if scores else 0.0,
                **latency_metrics
            },
            "results": results
        }
    
    def _evaluate_item(self, benchmark, item: DataSetItem) -> Dict[str, Any]:
        """Evaluate a single item (can run in parallel)."""
        input_data = item.input

        prompt = benchmark.get_user_prompt(input_data)

        start_time = time.perf_counter()
        at_output = self.inference_fn(prompt)
        end_time = time.perf_counter()
        latency = end_time - start_time

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
            "latency_seconds": latency
        }

    def _calculate_latency_metrics(self, latencies: List[float]) -> Dict[str, float]:
        """Calculate aggregate latency statistics."""
        if not latencies:
            return {
                "latency_mean": 0.0,
                "latency_std": 0.0,
                "total_latency": 0.0,
            }

        total_latency = sum(latencies)

        return {
            "latency_mean": statistics.mean(latencies),
            "latency_std": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            "total_latency": total_latency,
        }

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save results to JSON file."""
        import os
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Get SLURM array task ID if running in SLURM array job
        array_id = os.environ.get('SLURM_ARRAY_TASK_ID', '')
        if array_id:
            filename = f"eval_results_{timestamp}_array{array_id}.json"
        else:
            filename = f"eval_results_{timestamp}.json"

        filepath = self.output_dir / filename

        # Custom JSON encoder to handle dataclasses and other non-serializable objects
        def default_serializer(obj):
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=default_serializer)

        print(f"\nResults saved to: {filepath}")

