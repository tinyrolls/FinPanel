import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, List

import litellm
import yaml
from litellm.integrations.custom_logger import CustomLogger

_MODELS_YAML = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"
_log = logging.getLogger("llm.providers")

litellm.drop_params = True
litellm.suppress_debug_info = True


class ModelProvider(str, Enum):
    OPENAI = "openai"
    OPENAI_COMPATIBLE = "openai_compatible"
    DEEPSEEK = "deepseek"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


_CFG: dict | None = None


def _load_cfg() -> dict:
    global _CFG
    if _CFG is None:
        with _MODELS_YAML.open("r", encoding="utf-8") as fh:
            _CFG = yaml.safe_load(fh) or {}
    return _CFG


def _model_entry(model_name: str) -> dict:
    cfg = _load_cfg()
    models = cfg.get("models", {})
    if model_name not in models:
        raise ValueError(f"Unregistered model '{model_name}'. Add it to {_MODELS_YAML}. Known: {sorted(models)}")
    return {**cfg.get("defaults", {}), **(models[model_name] or {})}


LLM_TIMEOUT_S = float(_load_cfg().get("defaults", {}).get("timeout_s", 90.0))
LLM_MAX_RETRIES = int(_load_cfg().get("defaults", {}).get("max_retries", 2))


class _LiteLLMRecorder(CustomLogger):

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            usage = getattr(response_obj, "usage", None)
            pt = getattr(usage, "prompt_tokens", 0) if usage else 0
            ct = getattr(usage, "completion_tokens", 0) if usage else 0
            tt = getattr(usage, "total_tokens", 0) if usage else 0
            cost = kwargs.get("response_cost")
            if cost is None:
                try:
                    cost = litellm.completion_cost(completion_response=response_obj)
                except Exception:
                    cost = 0.0
            dt = (end_time - start_time).total_seconds()
            _log.debug("litellm ok model=%s dt=%.2fs tokens=%s cost=%s", kwargs.get("model"), dt, tt, cost)
            from llm.cost import record

            record(kwargs.get("model"), pt, ct, tt, cost or 0.0)
        except Exception:
            pass

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            _log.warning("litellm fail model=%s err=%s", kwargs.get("model"), kwargs.get("exception"))
        except Exception:
            pass


litellm.callbacks = [_LiteLLMRecorder()]


class _Completion:

    def __init__(self, model: str, params: dict, sim_date_template: bool = False):
        self._model = model
        self._params = params
        self._sim_date_template = sim_date_template

    @staticmethod
    def _to_messages(prompt: Any) -> List[dict]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        if isinstance(prompt, list):
            out = []
            for m in prompt:
                if isinstance(m, dict):
                    out.append(m)
                else:
                    role = getattr(m, "role", None) or getattr(m, "type", "user")
                    role = {"human": "user", "ai": "assistant"}.get(role, role)
                    out.append({"role": role, "content": getattr(m, "content", str(m))})
            return out
        return [{"role": "user", "content": str(prompt)}]

    def invoke(self, prompt: Any, sim_date: str | None = None) -> "_Result":
        params = self._params
        if self._sim_date_template and sim_date:
            params = dict(params)
            extra = dict(params.get("extra_body") or {})
            extra["chat_template_kwargs"] = {**(extra.get("chat_template_kwargs") or {}), "current_date": sim_date}
            params["extra_body"] = extra
        resp = litellm.completion(model=self._model, messages=self._to_messages(prompt), **params)
        return _Result(resp.choices[0].message.content or "")


class _Result:
    def __init__(self, content: str):
        self.content = content


def _effective_seed(entry: dict, seed: int | None) -> int:
    env_seed = os.environ.get("LLM_SEED")
    if env_seed is not None:
        return int(env_seed)
    return seed if seed is not None else int(entry.get("seed", 42))


def _require_key(env_name: str, label: str) -> str:
    key = os.getenv(env_name)
    if not key:
        print(f"API Key Error: Please make sure {env_name} is set in your .env file.")
        raise ValueError(f"{label} API key not found. Please make sure {env_name} is set in your .env file.")
    return key


def get_model(model_name: str, model_provider=None, seed: int | None = None) -> _Completion:
    entry = _model_entry(model_name)
    provider = ModelProvider(entry["provider"])
    params = {
        "temperature": entry.get("temperature", 0),
        "seed": _effective_seed(entry, seed),
        "timeout": float(entry.get("timeout_s", LLM_TIMEOUT_S)),
        "num_retries": int(entry.get("max_retries", LLM_MAX_RETRIES)),
    }
    if entry.get("extra_body"):
        params["extra_body"] = entry["extra_body"]
    if entry.get("max_tokens"):
        params["max_tokens"] = int(entry["max_tokens"])

    if provider == ModelProvider.OPENAI:
        params["api_key"] = _require_key("OPENAI_API_KEY", "OpenAI")
        model = model_name
    elif provider == ModelProvider.OPENAI_COMPATIBLE:
        params["api_key"] = _require_key(entry.get("api_key_env", "OPENAI_API_KEY"), model_name)
        params["api_base"] = entry["base_url"]
        model = f"openai/{model_name}"
    elif provider == ModelProvider.DEEPSEEK:
        params["api_key"] = _require_key(entry.get("api_key_env", "DEEPSEEK_API_KEY"), "DeepSeek")
        model = f"deepseek/{model_name}"
    elif provider == ModelProvider.ANTHROPIC:
        params["api_key"] = _require_key("ANTHROPIC_API_KEY", "Anthropic")
        model = f"anthropic/{model_name}"
    elif provider == ModelProvider.GEMINI:
        params["api_key"] = _require_key("GOOGLE_API_KEY", "Google")
        model = f"gemini/{model_name}"
    else:
        raise ValueError(f"Unsupported provider '{provider}' for model '{model_name}'.")

    return _Completion(model, params, sim_date_template=bool(entry.get("sim_date_template")))
