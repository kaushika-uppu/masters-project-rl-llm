import torch
from typing import Dict, Any, List
import re
import yaml
from vllm import LLM, SamplingParams
from src.inference.base_inference import BaseInference


class LIMO(BaseInference):
    def __init__(self, model, tokenizer, config):
        self.llm = model 
        self.config = config
        self.sampling_params = SamplingParams(
            temperature=self.config.get('inference', {}).get('temperature', 0.2),
            max_tokens=self.config.get('inference', {}).get('max_new_tokens', 8192),
            stop=["</answer>"]
        )

    def post_process_output(self, raw_output: str) -> str:
        """
        Extracts the answer from LIMO's \boxed{} format and appends it 
        inside <answer></answer> tags so benchmarks can parse it.
        """
        answer = ''
        if "\\boxed{" in raw_output:
            start_idx = raw_output.rfind("\\boxed{") + 7
            depth = 1
            
            # bracket matching algorithm to parse answer
            for i in range(start_idx, len(raw_output)):
                if raw_output[i] == '{':
                    depth += 1
                elif raw_output[i] == '}':
                    depth -= 1
                
                if depth == 0:
                    answer = raw_output[start_idx:i]
            if answer:
                # return raw output to keep reasoning, append final answer within <answer> tags for evalu>
                return raw_output + f"\n\n<answer>\n{answer.strip()}\n</answer>"
            
        return raw_output


    def generate(self, prompts: list) -> list:
        formatted_prompts = [
            f"System: Please reason step by step, and put your final answer within \\boxed{{}}.\nUser: {p}\nAssistant:"
            for p in prompts
        ]

        outputs = self.llm.generate(formatted_prompts, self.sampling_params, use_tqdm=False)

        final_results = []
        for out in outputs:
            raw_text = out.outputs[0].text
            processed_text = self.post_process_output(raw_text)

            # Extract metrics from vLLM RequestOutput
            metrics = self._extract_metrics(out)

            # Return dict with text and metrics
            final_results.append({
                "text": processed_text,
                "metrics": metrics
            })

        return final_results

    def _extract_metrics(self, request_output) -> Dict[str, Any]:
        """Extract detailed metrics from vLLM RequestOutput object."""
        metrics = {}

        # Token counts
        prompt_tokens = len(request_output.prompt_token_ids)
        output_tokens = len(request_output.outputs[0].token_ids)
        metrics["prompt_tokens"] = prompt_tokens
        metrics["output_tokens"] = output_tokens
        metrics["total_tokens"] = prompt_tokens + output_tokens

        # Timing metrics (if available in vLLM version)
        if hasattr(request_output, 'metrics') and request_output.metrics:
            vllm_metrics = request_output.metrics

            # Time to First Token (TTFT) in milliseconds
            if hasattr(vllm_metrics, 'first_token_time') and hasattr(vllm_metrics, 'first_scheduled_time'):
                if vllm_metrics.first_token_time and vllm_metrics.first_scheduled_time:
                    ttft_s = vllm_metrics.first_token_time - vllm_metrics.first_scheduled_time
                    metrics["ttft_ms"] = ttft_s * 1000

            # End-to-end latency in milliseconds
            if hasattr(vllm_metrics, 'finished_time') and hasattr(vllm_metrics, 'first_scheduled_time'):
                if vllm_metrics.finished_time and vllm_metrics.first_scheduled_time:
                    e2e_s = vllm_metrics.finished_time - vllm_metrics.first_scheduled_time
                    metrics["latency_ms"] = e2e_s * 1000

                    # Calculate tokens per second
                    if e2e_s > 0:
                        metrics["tokens_per_second"] = output_tokens / e2e_s

            # Time in queue
            if hasattr(vllm_metrics, 'time_in_queue'):
                if vllm_metrics.time_in_queue:
                    metrics["queue_time_ms"] = vllm_metrics.time_in_queue * 1000

        return metrics

def limo():
    """
    Function that sets up the LIMO environment.
    """
    print("Loading LIMO configuration...")
    with open("configs/limo.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    print(f"Loading model {config['model']['name']}...")
    llm = LLM(
	model=config['model']['name'], 
        tensor_parallel_size=2,
        dtype="bfloat16",
        gpu_memory_utilization=0.95,
        max_model_len=8192
    )

    limo_instance = LIMO(model=llm, tokenizer=None, config=config)
    
    def batch_inference_fn(prompts):
        return limo_instance.generate(prompts)

    batch_inference_fn.is_batch = True
    return batch_inference_fn