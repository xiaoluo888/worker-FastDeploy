import os
import uuid
import logging
from http import HTTPStatus
from fastdeploy.engine.sampling_params import SamplingParams 
from functools import wraps
from time import time

logging.basicConfig(level=logging.INFO)


def coerce_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def normalize_openai_route(route):
    if not route:
        return None

    normalized = str(route).strip()
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    if normalized.startswith("/openai"):
        normalized = normalized[len("/openai"):]
    return normalized


def convert_limit_mm_per_prompt(input_string: str):
    result = {}
    if not input_string:
        return result
    pairs = input_string.split(',')
    for pair in pairs:
        if not pair:
            continue
        key, value = pair.split('=')
        result[key] = int(value)
    return result

def count_physical_cores():
    with open('/proc/cpuinfo') as f:
        content = f.readlines()

    cores = set()
    current_physical_id = None
    current_core_id = None

    for line in content:
        if 'physical id' in line:
            current_physical_id = line.strip().split(': ')[1]
        elif 'core id' in line:
            current_core_id = line.strip().split(': ')[1]
            cores.add((current_physical_id, current_core_id))

    return len(cores)


class JobInput:
    def __init__(self, job: dict):
        self.raw = job or {}
        self.openai_route = normalize_openai_route(self.raw.get("openai_route"))
        self.openai_input = self.raw.get("openai_input") or {}

        # 通用文本输入：适配多种字段名
        self.llm_input = self.raw.get("prompt")
        if self.llm_input is None:
            self.llm_input = self.raw.get("text")
        if self.llm_input is None and not self.openai_route:
            self.llm_input = self.raw.get("input")
        if self.llm_input is None and self.openai_route == "/v1/completions":
            self.llm_input = self.openai_input.get("prompt")

        self.stream = coerce_bool(self.raw.get("stream"), False)
        self.max_tokens = self.raw.get("max_tokens")
        if self.max_tokens is None:
            self.max_tokens = self.raw.get("max_new_tokens")
        if self.max_tokens is None:
            self.max_tokens = self.openai_input.get("max_tokens")
        if self.max_tokens is None:
            self.max_tokens = 200

        # 是否走 OpenAI 兼容路由（比如 /v1/chat/completions /v1/images/generations）
        samp_param = dict(self.raw.get("sampling_params", {}))
        if "max_tokens" not in samp_param:
            samp_param["max_tokens"] = int(self.max_tokens)
        if "n" not in samp_param and self.openai_input.get("n") is not None:
            samp_param["n"] = int(self.openai_input["n"])
        self.sampling_params = SamplingParams(**samp_param)
        self.request_id = str(uuid.uuid4())

    def __repr__(self):
        return f"JobInput({self.__dict__})"

# ----------------------------------------------------------------------
# Dummy request/state，用于模拟 web 框架上下文（如果后面某些逻辑需要）
# 目前 FD 版可能用不到，但保留兼容性
# ----------------------------------------------------------------------
class DummyState:
    def __init__(self):
        self.request_metadata = None


class DummyRequest:
    def __init__(self):
        self.headers = {}
        self.state = DummyState()

    async def is_disconnected(self):
        return False



def create_error_response(
    message: str,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
):
    return {
        "error": {
            "message": message,
            "type": err_type,
            "code": status_code.value,
        }
    }


def get_int_bool_env(env_var: str, default: bool) -> bool:
    return int(os.getenv(env_var, int(default))) == 1



def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        logging.info(f"{func.__name__} completed in {end - start:.2f} seconds")
        return result

    return wrapper
