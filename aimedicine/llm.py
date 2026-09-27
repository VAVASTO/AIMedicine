from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from pydantic import ValidationError

from .data import ROOT, save_json


class LLMError(RuntimeError):
    pass


class MistralClient:
    def __init__(self, *, cache_dir=None, max_calls=None, transport=None):
        load_dotenv(ROOT / ".env", override=False)
        self.api_key = os.getenv("MISTRAL_API_KEY", "")
        self.model = os.getenv("MISTRAL_MODEL", "ministral-8b-latest")
        self.base_url = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1").rstrip("/")
        if self.base_url != "https://api.mistral.ai/v1":
            raise LLMError("Only the official HTTPS Mistral endpoint is allowed for this credential.")
        self.temperature = float(os.getenv("MISTRAL_TEMPERATURE", "0.1"))
        self.max_tokens = min(int(os.getenv("MISTRAL_MAX_TOKENS", "2400")), 3200)
        self.max_calls = int(max_calls if max_calls is not None else os.getenv("MISTRAL_MAX_CALLS", "18"))
        self.min_interval = float(os.getenv("MISTRAL_MIN_INTERVAL", "3"))
        self.cache_dir = Path(cache_dir or ROOT / ".cache/llm")
        self.calls = 0
        self.cache_hits = 0
        self.last_request = 0.0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.records = []
        self.http = httpx.Client(timeout=httpx.Timeout(90, connect=15), transport=transport, follow_redirects=False)

    def complete(self, role, system, payload, schema):
        request = {
            "model": os.getenv("MISTRAL_AUDITOR_MODEL", self.model) if role == "auditor" else self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": schema.__name__, "schema": schema.model_json_schema(), "strict": True}},
        }
        digest = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        cache = self.cache_dir / f"{digest}.json"
        if cache.exists():
            saved = json.loads(cache.read_text())
            parsed = schema.model_validate(saved["result"])
            self.cache_hits += 1
            self.records.append({"role": role, "cache": True, "request_hash": digest, "model": saved.get("model")})
            return parsed
        if not self.api_key:
            raise LLMError("MISTRAL_API_KEY is missing. Configure .env or use --mode offline.")
        for attempt in range(3):
            if self.calls >= self.max_calls:
                raise LLMError(f"Per-run API request limit reached ({self.max_calls}); no further requests sent.")
            delay = max(0, self.min_interval - (time.monotonic() - self.last_request))
            if delay:
                time.sleep(delay)
            self.calls += 1
            self.last_request = time.monotonic()
            try:
                response = self.http.post(self.base_url + "/chat/completions", json=request,
                                          headers={"Authorization": "Bearer " + self.api_key})
            except httpx.HTTPError:
                if attempt == 2:
                    raise LLMError("Mistral network request failed after bounded retries.") from None
                time.sleep(2 ** (attempt + 1))
                continue
            if response.status_code in (429, 502, 503, 504) and attempt < 2:
                if response.status_code == 429 and response.headers.get("x-ratelimit-limit-req-minute") == "0":
                    raise LLMError("Mistral HTTP 429: this model has a zero request allowance for this key. Select an available small model; retries stopped.")
                try:
                    delay = min(30.0, max(3.0, float(response.headers.get("retry-after", 2 ** (attempt + 2)))))
                except ValueError:
                    delay = 5.0
                time.sleep(delay)
                continue
            if response.status_code != 200:
                
                raise LLMError(f"Mistral HTTP {response.status_code}; credentials, quota or request need review.")
            try:
                body = response.json()
                usage = body.get("usage", {})
                for key in self.usage:
                    self.usage[key] += int(usage.get(key, 0))
                choice = body["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise LLMError("Mistral output reached token limit; no partial plan accepted.")
                content = choice["message"]["content"]
                if isinstance(content, list):
                    content = "".join(x.get("text", "") for x in content if x.get("type") == "text")
                parsed = schema.model_validate_json(content)
            except (KeyError, ValueError, TypeError, IndexError, AttributeError, ValidationError):
                raise LLMError("Mistral returned an invalid structured result; no clinical fallback was substituted.") from None
            save_json(cache, {"result": parsed.model_dump(), "model": body.get("model", self.model), "usage": usage,
                              "role": role, "request_hash": digest, "prompt_version": "2"})
            self.records.append({"role": role, "cache": False, "request_hash": digest,
                                 "model": body.get("model", self.model), "usage": usage})
            return parsed
        raise LLMError("Mistral request could not complete.")

    def summary(self):
        return {"model": self.model, "api_attempts": self.calls, "cache_hits": self.cache_hits,
                "usage": self.usage, "max_calls": self.max_calls, "max_output_tokens": self.max_tokens,
                "records": self.records}

    def close(self):
        self.http.close()
