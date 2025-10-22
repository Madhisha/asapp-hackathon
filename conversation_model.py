"""conversation_model.py

This module provides ChatModel which will attempt to load a local
Hugging Face-style model when a filesystem path is given. If that path does
not exist it will try to use an Ollama-managed model by either using the
Ollama Python client (if installed) or by invoking the `ollama` CLI.

This lets the project work whether you have a local HF model or an Ollama
model like `mistral:instruct` installed.
"""
import os
import subprocess
from typing import Optional

try:
    # Optional HF imports
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    _have_transformers = True
except Exception:
    _have_transformers = False

try:
    from ollama import Ollama
    _have_ollama_client = True
except Exception:
    _have_ollama_client = False


class ChatModel:
    def __init__(self, model_name: str = "mistral:instruct"):
        self.model_name = model_name
        self._mode: Optional[str] = None

        # 1) If model_name is a local filesystem path and transformers are
        # available, load it with HF transformers.
        if os.path.exists(model_name) and _have_transformers:
            print(f"Loading local model from {model_name} ...")
            self._mode = 'hf'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.eval()
            print("Local model loaded successfully!")
            return

        # 2) Try Ollama Python client if available
        if _have_ollama_client:
            self.client = Ollama()
            self._mode = 'client'
            print(f"Ollama Python client ready. Using model: {model_name}")
            return

        # 3) Fallback to Ollama CLI
        # Ensure the CLI exists and the model appears installed (best-effort)
        try:
            proc = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
            if model_name not in proc.stdout:
                print(f"Warning: model '{model_name}' not listed by `ollama list`. CLI is present though.")
            else:
                print(f"Ollama CLI ready and model '{model_name}' available.")
            self._mode = 'cli'
        except FileNotFoundError:
            raise RuntimeError("Neither a local HF model was found, nor is the Ollama Python client or `ollama` CLI available. Install one of these or provide a valid local model path.")
        except subprocess.CalledProcessError as e:
            print("Warning: `ollama list` failed:", e)
            # Still set CLI mode; running will surface errors later
            self._mode = 'cli'

    def generate_response(self, prompt: str, max_length: int = 300) -> str:
        """Generate a response using the configured backend.

        - 'hf' uses the transformers model.
        - 'client' uses the Ollama Python client.
        - 'cli' uses the `ollama run <model>` CLI and pipes prompt to stdin.
        """
        if self._mode == 'hf':
            if not _have_transformers:
                raise RuntimeError("Transformers not available to run local HF model")
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(inputs.input_ids, max_length=max_length)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response

        if self._mode == 'client':
            try:
                result = self.client.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
                return getattr(result, 'content', str(result))
            except Exception as e:
                raise RuntimeError(f"Ollama client failed: {e}")

        # CLI mode
        try:
            cmd = ["ollama", "run", self.model_name]
            # Use UTF-8 decode and replace errors to avoid platform codec issues
            proc = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=300,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.strip() or f"ollama exited with code {proc.returncode}")
            return proc.stdout.strip()
        except subprocess.TimeoutExpired as e:
            # Include any partial output for debugging
            partial_out = (e.stdout or '')
            partial_err = (e.stderr or '')
            raise RuntimeError(
                f"Ollama CLI timed out after {e.timeout} seconds. Partial stdout: {partial_out[:1000]!r} Partial stderr: {partial_err[:1000]!r}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Ollama CLI failed to run model '{self.model_name}': {e}")
        