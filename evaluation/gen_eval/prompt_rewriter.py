# -*- coding: utf-8 -*-
"""LLM API for GenEval prompt rewrite (cache miss fallback in geneval_lumina_dimoo.py)."""

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, List, Optional


DEFAULT_SYSTEM = """You refine text-to-image prompts for stronger image generation.
Reply with ONLY a compact JSON object: {"prompt": "<refined English prompt>"}.
Preserve all objects, counts, colors, and spatial relations from the user text. Do not invent new objects."""


def _parse_model_output(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    # fenced code block
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "prompt" in obj:
                return str(obj["prompt"]).strip()
        except json.JSONDecodeError:
            pass
    for line in text.splitlines():
        if line.lower().startswith("prompt:"):
            return line.split(":", 1)[1].strip().strip('"').strip("'")
    return text.strip()


class PromptRewriter:
    """OpenAI-compatible HTTP chat; env vars for endpoint and key (see rewrite() doc)."""

    def __init__(self, system: str = "", few_shot_history: Optional[List[Any]] = None):
        self.system = (system or "").strip() or DEFAULT_SYSTEM
        self.few_shot_history = few_shot_history or []
        self._api_key = (
            os.environ.get("GENEVAL_REWRITE_API_KEY")
            or os.environ.get("SILICONFLOW_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        ).strip()
        self._base_url = (
            os.environ.get("GENEVAL_REWRITE_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.siliconflow.cn/v1"
        ).rstrip("/")
        self._model = os.environ.get("GENEVAL_REWRITE_MODEL", "deepseek-ai/DeepSeek-V3").strip()
        self._warned_no_key = False

    def rewrite(self, prompt: str) -> str:
        if not self._api_key:
            if not self._warned_no_key:
                print(
                    "[rewrite] 未设置 GENEVAL_REWRITE_API_KEY / SILICONFLOW_API_KEY / OPENAI_API_KEY，"
                    "跳过改写，使用原始 prompt。"
                )
                self._warned_no_key = True
            return prompt

        url = f"{self._base_url}/chat/completions"
        messages = [{"role": "system", "content": self.system}]
        for turn in self.few_shot_history:
            messages.append(turn)
        messages.append(
            {
                "role": "user",
                "content": f"Original prompt:\n{prompt}\n\nReturn JSON only.",
            }
        )
        body = json.dumps(
            {
                "model": self._model,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 512,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
            print(f"[rewrite] API 调用失败，使用原始 prompt: {e}")
            return prompt

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            print(f"[rewrite] 响应格式异常，使用原始 prompt: {e}")
            return prompt

        refined = _parse_model_output(content)
        return refined if refined else prompt
