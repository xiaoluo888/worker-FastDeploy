# engine2.py
import os
import json
import uuid
import asyncio
from dataclasses import is_dataclass, replace
from typing import Any, Dict, AsyncGenerator, Optional

from fastdeploy.engine.async_llm import AsyncLLM
from utils import JobInput, create_error_response
from engine_args import get_engine_args
from runpod import RunPodLogger

log = RunPodLogger()


class BatchSize:
    def __init__(self, max_batch_size: int = 64, min_batch_size: int = 1, growth_factor: float = 1.3):
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.growth_factor = growth_factor
        self.current_batch_size = min_batch_size

    def update(self) -> None:
        nxt = int(self.current_batch_size * self.growth_factor)
        if nxt < self.min_batch_size:
            nxt = self.min_batch_size
        if nxt > self.max_batch_size:
            nxt = self.max_batch_size
        self.current_batch_size = nxt


class FastDeployEngine:
    def __init__(self):
        log.info("Initializing FastDeployEngine (FD AsyncLLM, non-OpenAI path)...")
        self.engine_args = get_engine_args()
        log.info(f"Engine args: {self.engine_args}")

        # FD AsyncLLM needs a pid (IPC namespace)
        self.pid = (
            os.getenv("FD_ASYNC_LLM_PID")
            or os.getenv("RUNPOD_POD_ID")
            or f"fd-{uuid.uuid4().hex[:8]}"
        )

        if hasattr(AsyncLLM, "from_engine_args"):
            self.llm = AsyncLLM.from_engine_args(self.engine_args, pid=self.pid)
        else:
            self.llm = AsyncLLM(self.engine_args, pid=self.pid)  # type: ignore

        # init once
        self._init_task: Optional[asyncio.Task] = None
        self._init_lock = asyncio.Lock()

        self.batch_size = BatchSize(
            max_batch_size=int(os.getenv("MAX_BATCH_SIZE", "64")),
            min_batch_size=int(os.getenv("MIN_BATCH_SIZE", "1")),
            growth_factor=float(os.getenv("BATCH_GROWTH_FACTOR", "1.3")),
        )

    async def _init_async(self) -> None:
        if hasattr(self.llm, "start"):
            await self.llm.start()
        if hasattr(self.llm, "init_connections"):
            await self.llm.init_connections()

    async def _ensure_ready(self) -> None:
        async with self._init_lock:
            if self._init_task is None:
                self._init_task = asyncio.create_task(self._init_async())
        await self._init_task

    def _build_prompt_dict(self, job_input: JobInput) -> Dict[str, Any]:
        """
        非 OpenAI 路径：
        - 兼容 raw = {"input": {...}} 的 RunPod local_test 格式
        - 确保 max_tokens 一定是 int（默认 128）
        """
        raw = getattr(job_input, "raw", None) or {}
        inp = raw.get("input", raw)  # ✅ 关键：兼容 {"input": {...}}
        llm_input = getattr(job_input, "llm_input", None)
        if llm_input is None:
            llm_input = inp.get("prompt", None)

        if llm_input is None:
            raise ValueError("Missing prompt: job_input.llm_input or raw['prompt'] or raw['input']['prompt'].")

        if isinstance(llm_input, dict):
            prompt = str(llm_input.get("content")) if "content" in llm_input else json.dumps(llm_input, ensure_ascii=False)
        elif isinstance(llm_input, list):
            prompt = "\n".join(str(x.get("content", "")) if isinstance(x, dict) else str(x) for x in llm_input)
        else:
            prompt = str(llm_input)

        out: Dict[str, Any] = {"prompt": prompt}

        mt = getattr(job_input, "max_tokens", None)
        if mt is None:
            mt = inp.get("max_tokens", None)
        if mt is None:
            mt = inp.get("max_new_tokens", None)
        if mt is None:
            mt = 128
        out["max_tokens"] = int(mt)
        out["n"] = int(1) 


        # temperature（可选）
        temp = inp.get("temperature", None)
        if temp is not None:
            out["temperature"] = float(temp)

        return out

    def _patch_sampling_params(self, sp: Any, max_tokens: int) -> Any:
        """
        只在你必须传 sampling_params 的时候用（比如 n>1）。
        保证 sp.max_tokens 不为 None，避免 FD 内部 min(int, None)。
        """
        if sp is None:
            return None

        if is_dataclass(sp):
            fields = getattr(sp, "__dataclass_fields__", {})
            updates = {}

            if "max_tokens" in fields and getattr(sp, "max_tokens", None) is None:
                updates["max_tokens"] = int(max_tokens)

            if "max_new_tokens" in fields and getattr(sp, "max_new_tokens", None) is None:
                updates["max_new_tokens"] = int(max_tokens)

            if "n" in fields:
                n = getattr(sp, "n", None)
                try:
                    n_int = int(n) if n is not None else 1
                except Exception:
                    n_int = 1
                if n_int <= 0:
                    n_int = 1
                if getattr(sp, "n", None) is None:
                    updates["n"] = n_int

            return replace(sp, **updates) if updates else sp

        return None

    async def _maybe_abort(self, request_id: str) -> None:
        for name in ("abort", "abort_request", "abort_requests"):
            fn = getattr(self.llm, name, None)
            if fn is None:
                continue
            try:
                ret = fn(request_id)
                if asyncio.iscoroutine(ret):
                    await ret
            except Exception:
                pass
            break

    async def generate(self, job_input: JobInput) -> AsyncGenerator[Dict[str, Any], None]:
        try:
            await self._ensure_ready()

            prompt_dict = self._build_prompt_dict(job_input)
            request_id = getattr(job_input, "request_id", None) or f"cmpl-{uuid.uuid4().hex[:12]}"
            stream = getattr(job_input, "stream", True)
            sp = getattr(job_input, "sampling_params", None)

            n = getattr(sp, "n", None)
            try:
                n_int = int(n) if n is not None else 1
            except Exception:
                n_int = 1

            sampling_params_to_pass = None
            if n_int > 1:
                sampling_params_to_pass = self._patch_sampling_params(sp, prompt_dict["max_tokens"])

            async for batch in self._generate_fd(
                request_id=request_id,
                prompt_dict=prompt_dict,
                sampling_params=sampling_params_to_pass,
                stream=stream,
            ):
                yield batch



        except asyncio.CancelledError:
            rid = getattr(job_input, "request_id", None) or "unknown"
            await self._maybe_abort(rid)
            raise
        except Exception as e:
            yield create_error_response(str(e))

    async def _generate_fd(
        self,
        request_id: str,
        prompt_dict: Dict[str, Any],
        sampling_params: Any,
        stream: bool,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        # FD AsyncLLm
        results = self.llm.generate(prompt_dict, sampling_params, request_id)

        # n_responses：如果 sampling_params=None，就当 1
        n_responses = 1
        if sampling_params is not None:
            try:
                n_responses = int(getattr(sampling_params, "n", 1) or 1)
            except Exception:
                n_responses = 1

        last_output_texts = ["" for _ in range(n_responses)]
        token_counters = {"batch": 0, "total": 0}
        n_input_tokens = 0
        first = True

        batch: Dict[str, Any] = {"choices": [{"tokens": []} for _ in range(n_responses)]}

        async for request_output in results:
            if first:
                prompt_ids = getattr(request_output, "prompt_token_ids", None)
                n_input_tokens = len(prompt_ids) if prompt_ids is not None else 0
                first = False

            outputs = getattr(request_output, "outputs", None)
            if outputs is None:
                out_list = []
            elif isinstance(outputs, list):
                out_list = outputs
            else:
                out_list = [outputs]
            for out in out_list:
                idx = int(getattr(out, "index", 0) or 0)
                if idx >= n_responses:
                    continue

                full_text = getattr(out, "text", "") or ""
                prev_text = last_output_texts[idx]


                if stream:
                    new_piece = full_text[len(prev_text):] if len(full_text) >= len(prev_text) else full_text
                    if new_piece:
                        batch["choices"][idx]["tokens"].append(new_piece)
                        token_counters["batch"] += 1
                        token_counters["total"] += 1

                    if token_counters["batch"] >= self.batch_size.current_batch_size:
                        batch["usage"] = {"input": n_input_tokens, "output": token_counters["total"]}
                        yield batch
                        batch = {"choices": [{"tokens": []} for _ in range(n_responses)]}
                        token_counters["batch"] = 0
                        self.batch_size.update()

                last_output_texts[idx] = full_text
        


        if not stream:
            log.info("===================================================================11")
            log.info(f"last_output_texts type={type(last_output_texts)} len={len(last_output_texts)} id={id(last_output_texts)}")
            log.info(f"last_output_texts repr={last_output_texts!r}")
            token_counters["total"] += sum(len(x) for x in last_output_texts if isinstance(x, str))
            log.info(f"token_counters={token_counters}")
            log.info("===================================================================22")
            for output_index, output in enumerate(last_output_texts):
                batch["choices"][output_index]["tokens"] = [output]
            token_counters["batch"] += 1

        if token_counters["batch"] > 0:
            batch["usage"] = {"input": n_input_tokens, "output": token_counters["total"]}
            yield batch


