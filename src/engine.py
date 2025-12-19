# engine.py
import os
import json
import logging
import threading
import asyncio
from typing import AsyncGenerator, Dict, Any

from fastdeploy.engine.args_utils import EngineArgs
# from fastdeploy.engine.engine import LLMEngine 
from fastdeploy.engine.async_llm import AsyncLLM
from fastdeploy.entrypoints.openai.utils import make_arg_parser
from fastdeploy.utils import FlexibleArgumentParser
from utils import JobInput, create_error_response
from engine_args import get_engine_args
from runpod import RunPodLogger
log = RunPodLogger()

import time
from fastdeploy.engine.engine import LLMEngine as _FDLLM

def _llm_get_generated_tokens(self, req_id):
    while True:
        results = self._get_generated_result()
        if not results:
            time.sleep(0.001)
            continue

        for res in results:
            if getattr(res, "request_id", None) != req_id:
                continue
            yield res
            if getattr(res, "finished", False):
                return

if not hasattr(_FDLLM, "_get_generated_tokens"):
    _FDLLM._get_generated_tokens = _llm_get_generated_tokens

# def _build_argv_from_env():
#     """
#     用环境变量构造一小段“伪 argv”，喂给 FD 自己的 argparse。
#     你可以按需扩展，这里先把最关键的几个写上。
#     """
#     argv = []

#     # 模型路径 / 名称（必填）
#     model = os.getenv("MODEL","/root/PaddlePaddle/ERNIE-4.5-0.3B-Paddle")
#     if not model:
#         raise RuntimeError("ENV MODEL / MODEL_NAME 未设置，FastDeploy 无法加载模型。")
#     argv += ["--model", model]

#     # 可选：最长长度
#     max_model_len = os.getenv("MAX_MODEL_LEN")
#     if max_model_len:
#         argv += ["--max-model-len", str(max_model_len)]

#     # 可选：tensor parallel
#     tp = os.getenv("TENSOR_PARALLEL_SIZE")
#     if tp:
#         argv += ["--tensor-parallel-size", str(tp)]

#     return argv


def get_fd_engine_args() -> EngineArgs:
    parser = make_arg_parser(FlexibleArgumentParser())
    argv = _build_argv_from_env()
    args = parser.parse_args(argv)
    logging.info(f"[FD] Parsed args from env: {args.__dict__}")
    return EngineArgs.from_cli_args(args)


class FastDeployEngine:
    def __init__(self):
        log.info("Initializing FastDeploy LLM engine (RunPod adapter, Async)...")
        self.engine_args = get_engine_args()
        log.info(f"Engine args: {self.engine_args}")

        self.llm_engine = LLMEngine.from_engine_args(self.engine_args)

        ok = self.llm_engine.start()
        if not ok:
            raise RuntimeError("Failed to initialize FastDeploy LLM engine.")

        self.max_concurrency = int(os.getenv("MAX_CONCURRENCY", "512"))

    # ===== The prompt dictionary required to map JobInput to FD =====
    def _jobinput_to_prompts(self, job_input: JobInput) -> Dict[str, Any]:
        """
        适配 utils.JobInput → FastDeploy LLMEngine 所需的 prompts dict。
        当前 JobInput 结构：
            - job_input.llm_input: 真正的 prompt（str 或 list）
            - job_input.max_tokens: int
            - job_input.raw: 原始 input dict
            - job_input.openai_route / openai_input: OpenAI 兼容模式
        """
        prompts: Dict[str, Any] = {}

        # ========== 1. 优先处理 OpenAI 兼容模式 ==========
        if getattr(job_input, "openai_route", None) and getattr(job_input, "openai_input", None):
            route = job_input.openai_route
            body = job_input.openai_input or {}

            # 公共采样参数
            max_tokens = body.get("max_tokens") or body.get("max_new_tokens")
            temperature = body.get("temperature", None)

            if route == "/v1/chat/completions":
                # ---- chat 模式：messages -> 一个大 prompt 字符串 ----
                messages = body.get("messages", [])
                system_parts = []
                dialog_parts = []

                for m in messages:
                    role = m.get("role")
                    content = m.get("content", "")

                    # content 可能是 list（富文本 / 多模态），先压成纯 text
                    if isinstance(content, list):
                        content = "".join(
                            p.get("text", "") if isinstance(p, dict) else str(p)
                            for p in content
                        )

                    if role == "system":
                        system_parts.append(content)
                    elif role == "user":
                        dialog_parts.append(f"User: {content}")
                    elif role == "assistant":
                        dialog_parts.append(f"Assistant: {content}")
                    else:
                        dialog_parts.append(f"{role}: {content}")

                prefix = ""
                if system_parts:
                    prefix = "\n".join(system_parts) + "\n\n"

                dialog_text = "\n".join(dialog_parts)

                # 最简单的 prompt：system 提示 + 对话 + Assistant: 收尾
                prompt = prefix + dialog_text + "\nAssistant:"
                prompts["prompt"] = prompt

            elif route == "/v1/completions":
                # ---- text completions 模式 ----
                prompt = body.get("prompt")
                # OpenAI 里 prompt 可能是 str 或 list[str]
                if isinstance(prompt, list):
                    prompt = "\n".join(str(p) for p in prompt)
                prompts["prompt"] = prompt

            else:
                # 其他 openai_route 先简单兜底：用 body.prompt
                prompt = body.get("prompt")
                if isinstance(prompt, list):
                    prompt = "\n".join(str(p) for p in prompt)
                prompts["prompt"] = prompt

            # 补充采样参数
            if max_tokens:
                prompts["max_tokens"] = int(max_tokens)
            if temperature is not None:
                prompts["temperature"] = float(temperature)

            return prompts

        # ========== 2. 非 OpenAI 模式：走“原始 llm_input / raw” ==========
        llm_input = getattr(job_input, "llm_input", None)

        if llm_input is None:
            # 兜底：从 raw 里尝试取 prompt
            raw = getattr(job_input, "raw", {}) or {}
            llm_input = raw.get("prompt")

        if llm_input is None:
            raise ValueError("JobInput 中未找到 llm_input 或 raw['prompt'] 字段。")

        # 统一把 llm_input 转成字符串 prompt
        if isinstance(llm_input, dict):
            # 如果是 dict（比如单条 message），尝试取 content，否则直接序列化
            if "content" in llm_input:
                prompt = str(llm_input["content"])
            else:
                prompt = json.dumps(llm_input, ensure_ascii=False)
        elif isinstance(llm_input, list):
            # list[str] 或 list[dict]，简单拼接
            if llm_input and isinstance(llm_input[0], dict) and "content" in llm_input[0]:
                prompt = "\n".join(str(m.get("content", "")) for m in llm_input)
            else:
                prompt = "\n".join(str(x) for x in llm_input)
        else:
            # str / 其它类型
            prompt = str(llm_input)

        prompts["prompt"] = prompt

        # 原始模式下的 max_tokens / temperature：
        raw = getattr(job_input, "raw", {}) or {}
        # 先用 job_input.max_tokens，再兜 raw
        max_tokens = getattr(job_input, "max_tokens", None) \
                    or raw.get("max_tokens") \
                    or raw.get("max_new_tokens")
        temperature = raw.get("temperature")

        if max_tokens:
            prompts["max_tokens"] = int(max_tokens)
        if temperature is not None:
            prompts["temperature"] = float(temperature)

        return prompts

    async def generate(self, job_input: JobInput) -> AsyncGenerator[Dict[str, Any], None]:
        prompts = self._jobinput_to_prompts(job_input)
        log.debug("==============================================================================")
        log.debug(job_input.stream)
        log.debug("==============================================================================")
        try:
            async for batch in self._generate_fd(
                llm_input=prompts,
                stream=job_input.stream,
                request_id=job_input.request_id,
                validated_sampling_params=job_input.sampling_params,
            ):
                yield batch
        except Exception as e:
            yield create_error_response(str(e))


    async def _generate_fd(
        self,
        request_id: str,
        llm_input,
        validated_sampling_params,
        stream
    ) -> AsyncGenerator[Dict[str, Any], None]:
        results_generator = self.llm_engine.generate(llm_input, validated_sampling_params, request_id)
        n_responses, n_input_tokens, is_first_output = validated_sampling_params.n, 0, True
        last_output_texts, token_counters = ["" for _ in range(n_responses)], {"batch": 0, "total": 0}
        
        batch = {
            "choices": [{"tokens": []} for _ in range(n_responses)],
        }
        async for request_output in results_generator:
            if is_first_output:  # 只在第一步统计一次输入 token 数
                n_input_tokens = len(request_output.prompt_token_ids)
                is_first_output = False

            # FastDeploy 这里是一个 CompletionOutput，而不是 list
            completion = request_output.outputs
            # 保险一点，兼容没有 text 的情况
            output_index = getattr(completion, "index", 0)
            full_text    = getattr(completion, "text", "") or ""   

            token_counters["total"] += 1

            if stream:
                # 和 vLLM 一样，只拿增量
                new_output = full_text[len(last_output_texts[output_index]):]
                batch["choices"][output_index]["tokens"].append(new_output)
                token_counters["batch"] += 1
        
                if token_counters["batch"] >= batch_size.current_batch_size:
                    batch["usage"] = {
                        "input": n_input_tokens,
                        "output": token_counters["total"],
                    }
                    yield batch
        
                    # 重置本 batch
                    batch = {
                        "choices": [{"tokens": []} for _ in range(n_responses)],
                    }
                    token_counters["batch"] = 0
                    batch_size.update()
        
            # 更新 last_output_texts，方便下次算“增量”
            last_output_texts[output_index] = full_text
        
        if not stream:
            for output_index, output in enumerate(last_output_texts):
                batch["choices"][output_index]["tokens"] = [output]
            token_counters["batch"] += 1

        if token_counters["batch"] > 0:
            batch["usage"] = {"input": n_input_tokens, "output": token_counters["total"]}
            yield batch