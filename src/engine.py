# engine.py
import os
import json
import logging
import threading
import asyncio
from typing import AsyncGenerator, Dict, Any

from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.engine import LLMEngine
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
        log.info("Initializing FastDeploy LLM engine (RunPod adapter)...")
        self.engine_args = get_engine_args()
        log.info(f"Engine args: {self.engine_args}")
        self.llm_engine = LLMEngine.from_engine_args(self.engine_args)

        ok = self.llm_engine.start(api_server_pid=None)
        if not ok:
            raise RuntimeError("Failed to initialize FastDeploy LLM engine.")

        # 给 handler 用的最大并发（你也可以直接写死）
        self.max_concurrency = int(os.getenv("MAX_CONCURRENCY", "512"))

    # ===== 把 JobInput 映射成 FD 需要的 prompts dict =====
    def _jobinput_to_prompts(self, job_input) -> Dict[str, Any]:
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
        if job_input.openai_route and job_input.openai_input:
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
        # 你的 utils.JobInput 里没有 prompt 字段，只有 llm_input + raw
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


    async def _generate_fd(
        self,
        llm_input,
        request_id: str,
        job_input: JobInput,
        stream: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        prompts = self._jobinput_to_prompts(job_input)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _worker():
            try:
                log.debug("==================== gougou1 ====================")
                log.info(llm_input)
                log.info(prompts)
                log.debug("==================== gougou2 ====================")
                # FastDeploy 的 generate 是同步 iterator
                for out in self.llm_engine.generate(prompts, stream=stream):
                    # out 通常是 dict，这里顺手塞个 request_id 进去，方便调试
                    if isinstance(out, dict) and request_id is not None:
                        out.setdefault("request_id", request_id)
                    loop.call_soon_threadsafe(queue.put_nowait, out)
            except Exception as e:
                # 出错时，用 error payload 推回 async 侧
                err_payload = {
                    "error": {
                        "message": str(e),
                        "type": "FastDeployEngineError",
                    }
                }
                loop.call_soon_threadsafe(queue.put_nowait, err_payload)
            finally:
                # 用 None 作为结束标记
                loop.call_soon_threadsafe(queue.put_nowait, None)

        # 启动后台线程
        threading.Thread(target=_worker, daemon=True).start()

        # async 侧消费队列里的数据，一条条往外 yield
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item


    async def generate(self, job_input: JobInput):
        log.debug("==================== Dump JobInput ====================")
        log.debug(f"type(job_input) = {type(job_input)}")
        try:
            log.debug(f"job_input.__dict__ = {getattr(job_input, '__dict__', None)}")
        except Exception as e:
            log.debug(f"access __dict__ failed: {e}")
        log.debug("==================== Dump JobInput END ====================")
        try:
            async for batch in self._generate_fd(
                llm_input=job_input.llm_input,
                stream=job_input.stream,
                request_id=job_input.request_id,
                job_input=job_input
            ):
                yield batch
        except Exception as e:
            yield create_error_response(str(e))
