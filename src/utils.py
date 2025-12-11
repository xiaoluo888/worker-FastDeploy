import os
import uuid
import logging
from http import HTTPStatus
from fastdeploy.engine.sampling_params import SamplingParams 
from functools import wraps
from time import time

logging.basicConfig(level=logging.INFO)


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
        self.raw = job

        # 通用文本输入：适配多种字段名
        self.llm_input = job.get("prompt") or job.get("text") or job.get("input")
        self.stream = job.get("stream", False)
        self.max_tokens = job.get("max_tokens", 200)

        # 是否走 OpenAI 兼容路由（比如 /v1/chat/completions /v1/images/generations）
        self.openai_route = job.get("openai_route", False)
        self.openai_input = job.get("openai_input")
        samp_param = job.get("sampling_params", {})
        if "max_tokens" not in samp_param:
            samp_param["max_tokens"] = 100
        self.sampling_params = SamplingParams(**samp_param)
        self.request_id = str(uuid.uuid4())
        self.openai_route = job.get("openai_route")
        self.openai_input = job.get("openai_input")

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
