import os
import sys
import time
import traceback
import multiprocessing
import runpod
from utils import JobInput
from engine import FastDeployEngine
from runpod import RunPodLogger

log = RunPodLogger()
fd_engine = None


def _startup():
    global fd_engine
    if fd_engine is not None:
        return fd_engine

    try:
        fd_engine = FastDeployEngine()
        log.info("FastDeploy engine initialized successfully")
        return fd_engine
    except Exception as exc:
        log.error(f"Worker startup failed: {exc}\n{traceback.format_exc()}")
        sys.exit(1)


def _merge_batches(batches):
    merged = {"texts": [], "usage": {"input": 0, "output": 0}}
    if not batches:
        return merged

    for batch in batches:
        if isinstance(batch, dict) and "error" in batch:
            return batch

    choice_count = max((len(batch.get("choices", [])) for batch in batches), default=0)
    merged["texts"] = ["" for _ in range(choice_count)]

    for batch in batches:
        usage = batch.get("usage") or {}
        merged["usage"]["input"] = max(merged["usage"]["input"], int(usage.get("input", 0) or 0))
        merged["usage"]["output"] = max(merged["usage"]["output"], int(usage.get("output", 0) or 0))

        for index, choice in enumerate(batch.get("choices", [])):
            tokens = choice.get("tokens") or []
            merged["texts"][index] += "".join("" if token is None else str(token) for token in tokens)

    return merged


def _get_model_name():
    model = (
        os.getenv("OPENAI_SERVED_MODEL_NAME_OVERRIDE")
        or os.getenv("SERVED_MODEL_NAME")
        or os.getenv("MODEL")
        or os.getenv("MODEL_NAME")
        or "baidu/ERNIE-4.5-0.3B-Paddle"
    )
    return str(model).rstrip("/").split("/")[-1] or str(model)


def _build_usage(usage):
    prompt_tokens = int(usage.get("input", 0) or 0)
    completion_tokens = int(usage.get("output", 0) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _build_models_response():
    created = int(time.time())
    model_name = _get_model_name()
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": created,
                "owned_by": "runpod",
            }
        ],
    }


def _build_final_response(job_input, merged):
    if isinstance(merged, dict) and "error" in merged:
        return merged

    created = int(time.time())
    model_name = _get_model_name()
    texts = merged.get("texts", [])
    usage = _build_usage(merged.get("usage", {}))

    if job_input.openai_route == "/v1/chat/completions":
        return {
            "id": f"chatcmpl-{job_input.request_id[:24]}",
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": index,
                    "message": {
                        "role": "assistant",
                        "content": text,
                    },
                    "finish_reason": "stop",
                }
                for index, text in enumerate(texts)
            ],
            "usage": usage,
        }

    return {
        "id": f"cmpl-{job_input.request_id[:24]}",
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": index,
                "text": text,
                "finish_reason": "stop",
            }
            for index, text in enumerate(texts)
        ],
        "usage": usage,
        "text": texts[0] if texts else "",
    }


async def handler(job):
    try:
        engine = _startup()
        job_input = JobInput(job.get("input", {}))
        if job.get("id"):
            job_input.request_id = str(job["id"])

        if job_input.openai_route == "/v1/models":
            return _build_models_response()

        results_generator = engine.generate(job_input)
        results = []
        async for batch in results_generator:
            if isinstance(batch, dict) and "error" in batch:
                return batch
            results.append(batch)

        if job_input.stream:
            return results

        return _build_final_response(job_input, _merge_batches(results))
    except SystemExit:
        raise
    except Exception as exc:
        error_str = str(exc)
        log.error(f"Error during inference: {error_str}")
        log.error(f"Full traceback:\n{traceback.format_exc()}")

        if "CUDA" in error_str or "cuda" in error_str:
            log.error("Terminating worker due to CUDA/GPU error")
            sys.exit(1)

        return {"error": error_str}


if __name__ == "__main__" or multiprocessing.current_process().name == "MainProcess":
    _startup()
    runpod.serverless.start(
        {
            "handler": handler,
            "concurrency_modifier": lambda x: int(os.getenv("MAX_CONCURRENCY", "1")),
            "return_aggregate_stream": True,
        }
    )
