# evaluation/evaluator.py
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional
from tqdm import tqdm
import json
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
            verbose: bool = False
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
        return {
            "num_samples": len(dataset),
            "metrics": {
                "average_score": avg_score,
                "total_correct": sum(1 for s in scores if s == 1.0),
                "accuracy": sum(1 for s in scores if s == 1.0) / len(scores) if scores else 0.0
            },
            "results": results[:100] 
        }
    
    def _evaluate_item(self, benchmark, item: DataSetItem) -> Dict[str, Any]:
        """Evaluate a single item (can run in parallel)."""
        input_data = item.input

        prompt = benchmark.get_user_prompt(input_data)
        at_output = self.inference_fn(prompt)
        parsed_at_output = benchmark.parse_output(at_output)
        score = benchmark.score(item, parsed_at_output)
        
        return {
            "id": item.id,
            "input": input_data,
            "full_output": at_output,
            "prediction": parsed_at_output,
            "expected": item.output,
            "metadata": item.metadata,
            "score": score
        }
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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

