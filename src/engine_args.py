# engine_args.py
import os
import logging

from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.entrypoints.openai.utils import make_arg_parser
from fastdeploy.utils import FlexibleArgumentParser


def _env_bool(key: str):
    v = os.getenv(key)
    if v is None:
        return None
    v = v.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return None


def _append_if_set(argv, flag: str, env_key: str):
    v = os.getenv(env_key)
    if v not in (None, ""):
        argv += [flag, str(v)]


def _append_bool_flag(argv, flag: str, env_key: str):
    b = _env_bool(env_key)
    if b:
        argv.append(flag)


def _build_argv_from_env():
    argv = []

    # model = os.getenv("MODEL","/root/PaddlePaddle/ERNIE-4.5-0.3B-Paddle")
    model = os.getenv("MODEL","baidu/ERNIE-4.5-0.3B-Paddle")
    if not model:
        raise RuntimeError("MODEL not set")
    argv += ["--model", model]

    _append_if_set(argv, "--max-model-len", "MAX_MODEL_LEN")
    _append_if_set(argv, "--tensor-parallel-size", "TENSOR_PARALLEL_SIZE")
    _append_if_set(argv, "--data-parallel-size", "DATA_PARALLEL_SIZE")
    _append_if_set(argv, "--block-size", "BLOCK_SIZE")
    _append_if_set(argv, "--max-num-seqs", "MAX_NUM_SEQS")
    _append_if_set(argv, "--tokenizer", "TOKENIZER")

    _append_if_set(argv, "--gpu-memory-utilization", "GPU_MEMORY_UTILIZATION")
    _append_if_set(argv, "--kv-cache-ratio", "KV_CACHE_RATIO")
    _append_bool_flag(argv, "--enable-prefix-caching", "ENABLE_PREFIX_CACHING")
    _append_if_set(argv, "--swap-space", "SWAP_SPACE")

    _append_bool_flag(argv, "--enable-chunked-prefill", "ENABLE_CHUNKED_PREFILL")
    _append_if_set(argv, "--max-num-partial-prefills", "MAX_NUM_PARTIAL_PREFILLS")
    _append_if_set(argv, "--max-long-partial-prefills", "MAX_LONG_PARTIAL_PREFILLS")
    _append_if_set(argv, "--long-prefill-token-threshold", "LONG_PREFILL_TOKEN_THRESHOLD")
    _append_if_set(argv, "--static-decode-blocks", "STATIC_DECODE_BLOCKS")

    _append_if_set(argv, "--reasoning-parser", "REASONING_PARSER")

    _append_bool_flag(argv, "--disable-custom-all-reduce", "DISABLE_CUSTOM_ALL_REDUCE")
    _append_bool_flag(argv, "--use-internode-ll-two-stage", "USE_INTERNODE_LL_TWO_STAGE")
    _append_bool_flag(argv, "--disable-sequence-parallel-moe", "DISABLE_SEQUENCE_PARALLEL_MOE")

    _append_if_set(argv, "--splitwise-role", "SPLITWISE_ROLE")
    _append_if_set(argv, "--innode-prefill-ports", "INNODE_PREFILL_PORTS")

    _append_if_set(argv, "--guided-decoding-backend", "GUIDED_DECODING_BACKEND")
    _append_bool_flag(argv, "--disable-any-whitespace", "GUIDED_DECODING_DISABLE_ANY_WHITESPACE")

    spec_conf = os.getenv("SPECULATIVE_CONFIG")
    if spec_conf:
        argv += ["--speculative-config", spec_conf]

    _append_if_set(argv, "--dynamic-load-weight", "DYNAMIC_LOAD_WEIGHT")
    _append_bool_flag(argv, "--enable-expert-parallel", "ENABLE_EXPERT_PARALLEL")

    _append_bool_flag(argv, "--enable-logprob", "ENABLE_LOGPROB")
    _append_if_set(argv, "--logprobs-mode", "LOGPROBS_MODE")
    _append_if_set(argv, "--max-logprobs", "MAX_LOGPROBS")

    _append_if_set(argv, "--served-model-name", "SERVED_MODEL_NAME")
    _append_if_set(argv, "--revision", "REVISION")

    _append_if_set(argv, "--chat-template", "CHAT_TEMPLATE")
    _append_if_set(argv, "--tool-call-parser", "TOOL_CALL_PARSER")
    _append_if_set(argv, "--tool-parser-plugin", "TOOL_PARSER_PLUGIN")

    _append_if_set(argv, "--load-choices", "LOAD_CHOICES")
    _append_if_set(argv, "--max-encoder-cache", "MAX_ENCODER_CACHE")

    quant = os.getenv("QUANTIZATION")
    if quant and quant.upper() != "BF16":
        argv += ["--quantization", quant]

    logging.info(f"[FD] argv from env: {argv}")
    return argv


def get_engine_args() -> EngineArgs:
    parser = make_arg_parser(FlexibleArgumentParser())
    argv = _build_argv_from_env()
    args = parser.parse_args(argv)
    logging.info(f"[FD] Parsed args: {args.__dict__}")
    return EngineArgs.from_cli_args(args)
