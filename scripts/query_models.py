#!/usr/bin/env python3
"""
query_models.py

Script to query LLM APIs for comprehension and generation evaluation.
Supports: OpenAI, Anthropic, Google, HuggingFace, local models via Ollama.

Usage:
    python query_models.py --model gpt-4o --task comprehension --cwe CWE-89
    python query_models.py --model claude-3.5-sonnet --task generation --strategy direct_injection
"""

import os
import json
import time
import argparse
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod

# Rate limiting
import threading
from collections import defaultdict


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    provider: str
    api_key_env: str
    endpoint: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7


# Model registry
MODELS = {
    # OpenAI
    "gpt-4o": ModelConfig("gpt-4o", "openai", "OPENAI_API_KEY"),
    "gpt-4-turbo": ModelConfig("gpt-4-turbo", "openai", "OPENAI_API_KEY"),
    
    # Anthropic
    "claude-3.5-sonnet": ModelConfig("claude-3-5-sonnet-20241022", "anthropic", "ANTHROPIC_API_KEY"),
    "claude-3-opus": ModelConfig("claude-3-opus-20240229", "anthropic", "ANTHROPIC_API_KEY"),
    
    # Google
    "gemini-1.5-pro": ModelConfig("gemini-1.5-pro", "google", "GOOGLE_API_KEY"),
    
    # Open source via Ollama
    "llama-3-70b": ModelConfig("llama3:70b", "ollama", "", endpoint="http://localhost:11434"),
    "codellama-34b": ModelConfig("codellama:34b", "ollama", "", endpoint="http://localhost:11434"),
    "deepseek-33b": ModelConfig("deepseek-coder:33b", "ollama", "", endpoint="http://localhost:11434"),
    "qwen-72b": ModelConfig("qwen:72b", "ollama", "", endpoint="http://localhost:11434"),
    "mixtral-8x22b": ModelConfig("mixtral:8x22b", "ollama", "", endpoint="http://localhost:11434"),
}


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.rpm = requests_per_minute
        self.tokens = requests_per_minute
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    def acquire(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.rpm, self.tokens + elapsed * (self.rpm / 60))
            self.last_update = now
            
            if self.tokens < 1:
                sleep_time = (1 - self.tokens) * (60 / self.rpm)
                time.sleep(sleep_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class ModelClient(ABC):
    """Abstract base class for model clients."""
    
    @abstractmethod
    def query(self, prompt: str, system: Optional[str] = None) -> str:
        pass


class OpenAIClient(ModelClient):
    """OpenAI API client."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.api_key = os.environ.get(config.api_key_env)
        if not self.api_key:
            raise ValueError(f"Missing API key: {config.api_key_env}")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("pip install openai")
    
    def query(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.config.name,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        return response.choices[0].message.content


class AnthropicClient(ModelClient):
    """Anthropic API client."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.api_key = os.environ.get(config.api_key_env)
        if not self.api_key:
            raise ValueError(f"Missing API key: {config.api_key_env}")
        
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("pip install anthropic")
    
    def query(self, prompt: str, system: Optional[str] = None) -> str:
        response = self.client.messages.create(
            model=self.config.name,
            max_tokens=self.config.max_tokens,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class OllamaClient(ModelClient):
    """Ollama local model client."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.endpoint = config.endpoint or "http://localhost:11434"
        
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("pip install requests")
    
    def query(self, prompt: str, system: Optional[str] = None) -> str:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        
        response = self.requests.post(
            f"{self.endpoint}/api/generate",
            json={
                "model": self.config.name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                }
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]


def get_client(model_name: str) -> ModelClient:
    """Get appropriate client for model."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    
    config = MODELS[model_name]
    
    if config.provider == "openai":
        return OpenAIClient(config)
    elif config.provider == "anthropic":
        return AnthropicClient(config)
    elif config.provider == "ollama":
        return OllamaClient(config)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")


def load_prompts(task: str, cwe: Optional[str] = None, strategy: Optional[str] = None) -> List[Dict]:
    """Load prompts from JSON files."""
    prompts_dir = Path(__file__).parent.parent / "prompts"
    
    if task == "comprehension":
        with open(prompts_dir / "comprehension" / "all_comprehension_prompts.json") as f:
            data = json.load(f)
        
        prompts = []
        for cwe_id, cwe_data in data["categories"].items():
            if cwe and cwe_id != cwe:
                continue
            for instance in cwe_data["instances"]:
                prompts.append({
                    "cwe": cwe_id,
                    "instance_id": instance["id"],
                    "prompt": data["prompt_template"].format(
                        language=instance["language"],
                        code=instance["code"]
                    ),
                    "ground_truth": instance["ground_truth"]
                })
        return prompts
    
    elif task == "generation":
        with open(prompts_dir / "generation" / "all_generation_prompts.json") as f:
            data = json.load(f)
        
        prompts = []
        for strat_name, strat_data in data["strategies"].items():
            if strategy and strat_name != strategy:
                continue
            for cwe_id, cwe_prompts in strat_data["prompts_by_cwe"].items():
                if cwe and cwe_id != cwe:
                    continue
                for i, prompt in enumerate(cwe_prompts):
                    prompts.append({
                        "strategy": strat_name,
                        "cwe": cwe_id,
                        "instance_id": i,
                        "prompt": prompt
                    })
        return prompts
    
    else:
        raise ValueError(f"Unknown task: {task}")


def run_evaluation(
    model_name: str,
    task: str,
    cwe: Optional[str] = None,
    strategy: Optional[str] = None,
    output_file: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict]:
    """
    Run evaluation on model.
    
    Args:
        model_name: Model to evaluate
        task: 'comprehension' or 'generation'
        cwe: Filter by CWE (optional)
        strategy: Filter by strategy (optional, generation only)
        output_file: Save results to file
        limit: Maximum prompts to run
    
    Returns:
        List of results
    """
    print(f"Initializing {model_name}...")
    client = get_client(model_name)
    rate_limiter = RateLimiter(requests_per_minute=30)
    
    print(f"Loading {task} prompts...")
    prompts = load_prompts(task, cwe, strategy)
    
    if limit:
        prompts = prompts[:limit]
    
    print(f"Running {len(prompts)} evaluations...")
    results = []
    
    for i, prompt_data in enumerate(prompts):
        rate_limiter.acquire()
        
        try:
            response = client.query(prompt_data["prompt"])
            
            result = {
                "model": model_name,
                "task": task,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                **prompt_data,
                "response": response,
                "success": True,
                "error": None
            }
        except Exception as e:
            result = {
                "model": model_name,
                "task": task,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                **prompt_data,
                "response": None,
                "success": False,
                "error": str(e)
            }
        
        results.append(result)
        
        # Progress
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{len(prompts)}")
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Query LLMs for evaluation")
    parser.add_argument("--model", required=True, help="Model name (e.g., gpt-4o)")
    parser.add_argument("--task", required=True, choices=["comprehension", "generation"])
    parser.add_argument("--cwe", help="Filter by CWE (e.g., CWE-89)")
    parser.add_argument("--strategy", help="Filter by strategy (generation only)")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--limit", type=int, help="Maximum prompts to run")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    args = parser.parse_args()
    
    if args.list_models:
        print("Available models:")
        for name, config in MODELS.items():
            print(f"  {name} ({config.provider})")
        return
    
    results = run_evaluation(
        model_name=args.model,
        task=args.task,
        cwe=args.cwe,
        strategy=args.strategy,
        output_file=args.output,
        limit=args.limit
    )
    
    # Summary
    success = sum(1 for r in results if r["success"])
    print(f"\nCompleted: {success}/{len(results)} successful")


if __name__ == "__main__":
    main()
