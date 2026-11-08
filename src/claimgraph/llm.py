"""Chat model factory and call logging. The rest of the code only sees LangChain chat models.

Default provider on this install: hf_gguf, a free quantised model file downloaded from Hugging Face
and run in this process by llama-cpp-python. Nothing else needs to be installed or running.
"""
from __future__ import annotations

import gc
import logging
import threading
import time
from typing import Any, Optional
from uuid import UUID

import httpx
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult

from .config import get_settings

log = logging.getLogger(__name__)


class LLMCallLogger(BaseCallbackHandler):
    """Writes one row to llm_calls per model call. A logging failure never stops the pipeline."""

    def __init__(self) -> None:
        self._starts: dict[UUID, tuple[float, dict]] = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs) -> None:
        self._starts[run_id] = (time.perf_counter(), metadata or {})

    def on_llm_start(self, serialized, prompts, *, run_id, metadata=None, **kwargs) -> None:
        self._starts[run_id] = (time.perf_counter(), metadata or {})

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        self._write(run_id, ok=True)

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        self._write(run_id, ok=False)

    def _write(self, run_id: UUID, ok: bool) -> None:
        started, metadata = self._starts.pop(run_id, (time.perf_counter(), {}))
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        try:
            from .db import session_scope
            from .models_db import LLMCall

            with session_scope() as session:
                session.add(
                    LLMCall(
                        chain_name=metadata.get("chain_name"),
                        prompt_version=metadata.get("prompt_version"),
                        model=metadata.get("ls_model_name") or metadata.get("model"),
                        ms=elapsed_ms,
                        ok=ok,
                    )
                )
        except Exception:  # noqa: BLE001
            log.debug("Could not record the LLM call", exc_info=True)


# One loaded model is shared by every chain: loading it twice would use the RAM twice,
# and llama.cpp must not run two calls on one model at the same moment.
_engine = None
_engine_key: tuple | None = None
_engine_lock = threading.RLock()


def gguf_path(repo_id: str, filename: str, download: bool = True) -> str | None:
    """Local path of a model file in the Hugging Face cache, downloading it if allowed."""
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    cached = try_to_load_from_cache(repo_id, filename)
    if isinstance(cached, str):
        return cached
    if not download:
        return None
    return hf_hub_download(repo_id, filename, token=get_settings().hf_token or None)


def _get_engine(repo_id: str, filename: str, n_ctx: int, n_threads: int):
    global _engine, _engine_key
    key = (repo_id, filename, n_ctx, n_threads)
    with _engine_lock:
        if _engine is None or _engine_key != key:
            from llama_cpp import Llama

            _engine = None
            gc.collect()
            _engine = Llama(
                model_path=gguf_path(repo_id, filename),
                n_ctx=n_ctx,
                n_threads=n_threads,
                chat_format="chatml",
                verbose=False,
            )
            _engine_key = key
        return _engine


def unload_llm() -> None:
    """Free the model's RAM before a memory-heavy step. Important on low-memory machines."""
    global _engine, _engine_key
    with _engine_lock:
        _engine = None
        _engine_key = None
    gc.collect()
    settings = get_settings()
    for key in ("llm", "llm_small"):
        config = settings.models[key]
        if config.get("provider") == "ollama":
            try:
                httpx.post(
                    f"{settings.ollama_base_url}/api/generate",
                    json={"model": config.get("model"), "keep_alive": 0},
                    timeout=10,
                )
            except Exception:  # noqa: BLE001
                log.debug("Could not unload %s", config.get("model"), exc_info=True)


def _role(message: BaseMessage) -> str:
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, HumanMessage):
        return "user"
    return "assistant"


def strict_schema(model) -> dict:
    """The JSON schema of a Pydantic model with every field required, at every level.

    Fields with defaults are optional in Pydantic's schema, so a grammar built from it lets the model
    answer "{}", which then parses as a valid, empty result. Unknown values are still written as null.
    """

    def require(node):
        if isinstance(node, dict):
            if "properties" in node:
                node["required"] = list(node["properties"])
            for value in node.values():
                require(value)
        elif isinstance(node, list):
            for value in node:
                require(value)
        return node

    return require(model.model_json_schema())


class ChatHFGGUF(BaseChatModel):
    """LangChain chat model over a GGUF file from Hugging Face.

    ChatLlamaCpp is not used because each instance loads its own copy of the model, and its
    structured output relies on tool calling. This class shares one engine and uses grammar-constrained
    JSON, so the output follows the schema.
    """

    repo_id: str
    filename: str
    temperature: float = 0.1
    max_tokens: int = 900
    n_ctx: int = 4096
    n_threads: int = 4

    @property
    def _llm_type(self) -> str:
        return "hf_gguf"

    def _get_ls_params(self, stop: Optional[list[str]] = None, **kwargs: Any):
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_model_name"] = self.filename
        return params

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        with _engine_lock:
            engine = _get_engine(self.repo_id, self.filename, self.n_ctx, self.n_threads)
            output = engine.create_chat_completion(
                messages=[{"role": _role(item), "content": str(item.content)} for item in messages],
                temperature=self.temperature,
                max_tokens=int(kwargs.get("max_tokens", self.max_tokens)),
                stop=stop or [],
                response_format=kwargs.get("response_format"),
            )
        choice = output["choices"][0]
        message = AIMessage(content=choice["message"].get("content") or "")
        info = {"finish_reason": choice.get("finish_reason"), "usage": output.get("usage")}
        return ChatResult(generations=[ChatGeneration(message=message, generation_info=info)])

    def with_structured_output(self, schema, **kwargs: Any):
        response_format = {"type": "json_object", "schema": strict_schema(schema)}
        return self.bind(response_format=response_format) | PydanticOutputParser(pydantic_object=schema)


def get_chat_model(kind: str = "main", **overrides: Any):
    """Return a LangChain chat model for kind 'main' or 'small', built from configs/models.yaml."""
    if kind not in ("main", "small"):
        raise ValueError(f"Unknown model kind: {kind}")
    settings = get_settings()
    config = dict(settings.models["llm" if kind == "main" else "llm_small"])
    config.update(overrides)
    provider = config.get("provider", "hf_gguf")
    callbacks = [LLMCallLogger()]

    if provider == "hf_gguf":
        return ChatHFGGUF(
            repo_id=config["hf_model_id"],
            filename=config["gguf_file"],
            temperature=config.get("temperature", 0.1),
            max_tokens=config.get("num_predict", 900),
            n_ctx=config.get("num_ctx", 4096),
            n_threads=config.get("n_threads", 4),
            callbacks=callbacks,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config["model"],
            temperature=config.get("temperature", 0.1),
            base_url=settings.ollama_base_url,
            num_ctx=config.get("num_ctx", 4096),
            num_predict=config.get("num_predict", 900),
            keep_alive=config.get("keep_alive", "2m"),
            callbacks=callbacks,
        )

    if provider == "hf_local":
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
        from transformers import BitsAndBytesConfig

        pipeline = HuggingFacePipeline.from_model_id(
            model_id=config["hf_model_id"],
            task="text-generation",
            device_map="auto",
            model_kwargs={"quantization_config": BitsAndBytesConfig(load_in_4bit=True)},
            pipeline_kwargs={
                "max_new_tokens": config.get("num_predict", 900),
                "temperature": config.get("temperature", 0.1),
                "do_sample": True,
                "return_full_text": False,
            },
        )
        return ChatHuggingFace(llm=pipeline, callbacks=callbacks)

    if provider == "hf_endpoint":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        endpoint = HuggingFaceEndpoint(
            repo_id=config["hf_model_id"],
            temperature=config.get("temperature", 0.1),
            max_new_tokens=config.get("num_predict", 900),
            huggingfacehub_api_token=settings.hf_token or None,
        )
        return ChatHuggingFace(llm=endpoint, callbacks=callbacks)

    raise ValueError(f"Unknown LLM provider: {provider}")


def structured(model, schema):
    """Bind a Pydantic schema so that the model's reply is parsed and validated."""
    if model.__class__.__name__ == "ChatOllama":
        return model.with_structured_output(schema, method="json_schema")
    return model.with_structured_output(schema)


def check_llm(timeout: float = 2.0) -> dict:
    """Report whether the configured language model can run. Never raises. Shown on the setup screen.

    fatal means analysis cannot start; a model that is merely not downloaded yet is fetched on first use.
    """
    settings = get_settings()
    config = settings.models["llm"]
    provider = config.get("provider", "hf_gguf")
    base = {"provider": provider, "model": config.get("gguf_file") or config.get("model") or config.get("hf_model_id")}

    if provider == "hf_gguf":
        try:
            import llama_cpp  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return {
                **base,
                "ok": False,
                "fatal": True,
                "detail": f"llama-cpp-python is not usable ({exc.__class__.__name__}). Run: python scripts/run.py setup",
            }
        try:
            present = gguf_path(config["hf_model_id"], config["gguf_file"], download=False) is not None
        except Exception:  # noqa: BLE001
            present = False
        if not present:
            return {
                **base,
                "ok": False,
                "fatal": False,
                "detail": f"{config['gguf_file']} is not downloaded yet. It downloads from Hugging Face on first use, or run: python scripts/run.py setup",
            }
        return {**base, "ok": True, "fatal": False, "detail": "ready"}

    if provider == "ollama":
        wanted = {config["model"], settings.models["llm_small"].get("model")}
        try:
            response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=timeout)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            return {
                **base,
                "ok": False,
                "fatal": True,
                "detail": f"Ollama is not reachable at {settings.ollama_base_url} ({exc.__class__.__name__}).",
            }
        available = {item.get("name") for item in response.json().get("models", [])}
        missing = sorted(name for name in wanted if name and name not in available)
        detail = "ready" if not missing else "Run: " + "; ".join(f"ollama pull {name}" for name in missing)
        return {**base, "ok": not missing, "fatal": bool(missing), "detail": detail}

    return {**base, "ok": True, "fatal": False, "detail": f"Using provider {provider}."}
