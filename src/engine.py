# engine.py
import os
import logging
import threading
import asyncio
from typing import AsyncGenerator, Dict, Any

from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.engine.engine import LLMEngine
from fastdeploy.entrypoints.openai.utils import make_arg_parser
from fastdeploy.utils import FlexibleArgumentParser
from utils import JobInput
from engine_args import get_engine_args



import time
from fastdeploy.engine.engine import LLMEngine as _FDLLM

def _llm_get_generated_tokens(self, req_id):
    """用 _get_generated_result 模拟出 _get_generated_tokens 的行为。"""
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
        logging.info("Initializing FastDeploy LLM engine (RunPod adapter)...")
        self.engine_args = get_engine_args()
        logging.info(f"Engine args: {self.engine_args}")
        self.llm_engine = LLMEngine.from_engine_args(self.engine_args)

        ok = self.llm_engine.start(api_server_pid=None)
        if not ok:
            raise RuntimeError("Failed to initialize FastDeploy LLM engine.")

        # 给 handler 用的最大并发（你也可以直接写死）
        self.max_concurrency = int(os.getenv("MAX_CONCURRENCY", "512"))

    # ===== 把 JobInput 映射成 FD 需要的 prompts dict =====
    def _jobinput_to_prompts(self, job_input) -> Dict[str, Any]:
        """
        把 JobInput 统一转换成 FastDeploy LLMEngine 所需的 prompts dict。
        支持两种情况：
        1）原始模式：input 里直接有 prompt / text / input
        2）OpenAI 模式：input 里有 openai_route + openai_input
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

        # ========== 2. 非 OpenAI 模式：走“原始 prompt” ==========
        # 这里 job_input.prompt 是 __init__ 里从 prompt/text/input 兜过来的
        prompt = job_input.prompt
        if prompt is None:
            raise ValueError("JobInput 中未找到 prompt 或 openai_input 字段。")

        # 如果 prompt 是 dict（例如用户误传了个结构体），简单序列化一下避免类型错误
        if isinstance(prompt, dict):
            prompt = json.dumps(prompt, ensure_ascii=False)

        # 如果是 list[str]，合并成一个字符串
        if isinstance(prompt, list):
            if prompt and isinstance(prompt[0], str):
                prompt = "\n".join(prompt)
            # 如果将来支持 list[int]（已经是 token id），这里就直接交给 FD，不处理

        prompts["prompt"] = prompt

        # 原始模式下的 max_tokens / temperature 从 job_input.raw 里取
        raw = job_input.raw
        max_tokens = raw.get("max_tokens") or raw.get("max_new_tokens")
        temperature = raw.get("temperature")

        if max_tokens:
            prompts["max_tokens"] = int(max_tokens)
        if temperature is not None:
            prompts["temperature"] = float(temperature)

        return prompts

    # ===== 核心：同步 generate -> 异步 async generator =====
    async def generate(self, job_input) -> AsyncGenerator[Dict[str, Any], None]:
        """
        RunPod 期望的 async generator：
        - 内部起一个线程跑 LLMEngine.generate(prompts, stream=True)
        - 通过 asyncio.Queue 把结果推出来
        """
        prompts = self._jobinput_to_prompts(job_input)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def _worker():
            try:
                for out in self.llm_engine.generate(prompts, stream=True):
                    loop.call_soon_threadsafe(queue.put_nowait, out)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, {"__error__": str(e)})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=_worker, daemon=True).start()

        while True:
            item = await queue.get()
            if item is None:
                break
            if "__error__" in item:
                raise RuntimeError(f"FD engine error: {item['__error__']}")
            yield item
