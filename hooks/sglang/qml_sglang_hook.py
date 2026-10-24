# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Korea Platform Service Technology Co., Inc. All rights reserved.
"""
QML Custom Hook for SGLang (Python/Triton)
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import asyncio


@dataclass
class SGLangConfig:
    """Configuration for SGLang hook"""
    model_path: str = "/path/to/model"
    host: str = "127.0.0.1"
    port: int = 30000
    dtype: str = "float16"
    tp_size: int = 1


class SGLangHook:
    """Custom hook for SGLang integration"""
    
    def __init__(self, config: Optional[SGLangConfig] = None):
        """Bind the hook to a server address derived from config.

        in: config — SGLangConfig with host/port/model settings; None uses the
            dataclass defaults
        out: None (constructor); connection is deferred to initialize()
        """
        self.config = config or SGLangConfig()
        self.base_url = f"http://{self.config.host}:{self.config.port}"
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize SGLang server connection

        in: (none)
        out: None; marks the hook initialized after a 200 response from
             /health, raises RuntimeError when httpx is not installed or the
             server is unreachable/unhealthy
        """
        try:
            import httpx
            # Check server health
            response = httpx.get(f"{self.base_url}/health")
            if response.status_code == 200:
                self._initialized = True
                print(f"[SGLangHook] Connected to SGLang server: {self.base_url}")
            else:
                raise RuntimeError(f"Server health check failed: {response.status_code}")
        except ImportError:
            raise RuntimeError("httpx not installed. Install with: pip install httpx")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to SGLang server: {e}")
    
    def pre_process(self, input_text: str) -> str:
        """Pre-processing hook

        in: input_text — prompt text about to be sent to the server
        out: the text to actually send (currently returned unmodified)
        side effects: prints a truncated preview of the input
        """
        print(f"[SGLangHook] Pre-processing input: {input_text[:50]}...")
        return input_text
    
    def post_process(self, output_text: str) -> str:
        """Post-processing hook

        in: output_text — generated text returned by the server
        out: the text to hand back to the caller (currently unmodified)
        side effects: prints a truncated preview of the output
        """
        print(f"[SGLangHook] Post-processing output: {output_text[:50]}...")
        return output_text
    
    async def generate_async(self, prompt: str, max_tokens: int = 256) -> str:
        """Async text generation using SGLang

        in: prompt — input text (run through pre_process first)
        in: max_tokens — generation cap forwarded as sampling_params.max_tokens
        out: generated text from POST /generate, run through post_process
        """
        if not self._initialized:
            self.initialize()
        
        import httpx
        
        prompt = self.pre_process(prompt)
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "max_tokens": max_tokens,
                        "temperature": 0.7,
                        "top_p": 0.95
                    }
                },
                timeout=30.0
            )
            
            result = response.json()["text"]
            return self.post_process(result)
    
    async def batch_generate_async(self, prompts: List[str], max_tokens: int = 256) -> List[str]:
        """Async batch generation

        in: prompts — input texts (each run through pre_process first)
        in: max_tokens — per-request generation cap
        out: generated texts from POST /batch_generate, aligned with prompts,
             each run through post_process
        """
        if not self._initialized:
            self.initialize()
        
        import httpx
        
        prompts = [self.pre_process(p) for p in prompts]
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/batch_generate",
                json={
                    "texts": prompts,
                    "sampling_params": {
                        "max_tokens": max_tokens,
                        "temperature": 0.7,
                        "top_p": 0.95
                    }
                },
                timeout=60.0
            )
            
            results = [self.post_process(output) for output in response.json()["texts"]]
            return results
    
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """Sync text generation

        in: prompt — input text
        in: max_tokens — generation cap
        out: generated text (blocks on asyncio.run over generate_async)
        """
        return asyncio.run(self.generate_async(prompt, max_tokens))
    
    def batch_generate(self, prompts: List[str], max_tokens: int = 256) -> List[str]:
        """Sync batch generation

        in: prompts — input texts
        in: max_tokens — per-request generation cap
        out: generated texts aligned with prompts (blocks on asyncio.run over
             batch_generate_async)
        """
        return asyncio.run(self.batch_generate_async(prompts, max_tokens))


# Global hook instance
_hook: Optional[SGLangHook] = None


def get_hook(config: Optional[SGLangConfig] = None) -> SGLangHook:
    """Get global hook instance

    in: config — SGLangConfig used only if the singleton does not exist yet;
        ignored on later calls
    out: the process-wide SGLangHook singleton (created on first call)
    """
    global _hook
    if _hook is None:
        _hook = SGLangHook(config)
    return _hook


def reset_hook() -> None:
    """Reset global hook instance

    in: (none)
    out: None
    side effects: clears the singleton so the next get_hook() builds a new one
    """
    global _hook
    _hook = None


if __name__ == "__main__":
    # Example usage
    hook = SGLangHook()
    hook.initialize()
    result = hook.generate("Hello, how are you?")
    print(f"Response: {result}")