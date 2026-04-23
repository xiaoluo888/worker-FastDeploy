# engine_args.py
import os
import logging
import json
from pathlib import Path

from fastdeploy.engine.args_utils import EngineArgs
from fastdeploy.entrypoints.openai.utils import make_arg_parser
from fastdeploy.utils import FlexibleArgumentParser


DEFAULT_MODEL_ID = "baidu/ERNIE-4.5-0.3B-Paddle"
DEFAULT_LOCAL_MODEL_ROOT = "/root/PaddlePaddle"
LOCAL_MODEL_ARGS_PATH = "/local_model_args.json"

ENV_ALIASES = {
    "MODEL_NAME": "MODEL",
    "MODEL_REPO": "MODEL",
    "MODEL_REVISION": "REVISION",
    "TOKENIZER_NAME": "TOKENIZER",
}


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
    v = _get_env(env_key)
    if v not in (None, ""):
        argv += [flag, str(v)]


def _append_bool_flag(argv, flag: str, env_key: str):
    b = _env_bool(env_key)
    if b:
        argv.append(flag)


def _get_env(key: str, default=None):
    keys = [key]
    keys.extend(alias for alias, target in ENV_ALIASES.items() if target == key)
    for env_key in keys:
        value = os.getenv(env_key)
        if value not in (None, ""):
            return value
    return default


def _load_local_model_args():
    if not os.path.exists(LOCAL_MODEL_ARGS_PATH):
        return {}

    with open(LOCAL_MODEL_ARGS_PATH, "r", encoding="utf-8") as f:
        local_args = json.load(f)

    logging.info("Using baked in model args from %s: %s", LOCAL_MODEL_ARGS_PATH, local_args)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    for key, value in local_args.items():
        target_key = ENV_ALIASES.get(key, key)
        if value not in (None, "", "None") and os.getenv(target_key) in (None, ""):
            os.environ[target_key] = str(value)

    return local_args


def _is_remote_model_id(model: str) -> bool:
    model = model.strip()
    if not model or "://" in model:
        return False
    if model.startswith(("/", "./", "../", "~")):
        return False
    return "/" in model and not Path(model).expanduser().exists()


def _resolve_model(model: str) -> str:
    expanded = Path(model).expanduser()
    if expanded.exists():
        return str(expanded)

    if not _is_remote_model_id(model):
        return model

    local_root = Path(os.getenv("LOCAL_MODEL_ROOT", DEFAULT_LOCAL_MODEL_ROOT)).expanduser()
    local_candidate = local_root / model.rstrip("/").split("/")[-1]
    if local_candidate.exists():
        logging.info("[FD] Resolved MODEL %s to local model path: %s", model, local_candidate)
        return str(local_candidate)

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            f"MODEL={model!r} looks like a Hugging Face model id, but huggingface_hub "
            "is unavailable and no local model directory was found."
        ) from exc

    offline = bool(_env_bool("HF_HUB_OFFLINE")) or bool(_env_bool("TRANSFORMERS_OFFLINE"))
    revision = _get_env("REVISION") or None
    cache_dir = os.getenv("HF_HOME") or os.getenv("HUGGINGFACE_HUB_CACHE") or None
    logging.info("[FD] Downloading MODEL %s from Hugging Face Hub", model)
    return snapshot_download(
        repo_id=model,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=offline,
    )


def _build_argv_from_env():
    _load_local_model_args()
    argv = []

    model = _get_env("MODEL", DEFAULT_MODEL_ID)
    if not model:
        raise RuntimeError("MODEL not set")
    model = _resolve_model(model)
    argv += ["--model", model]

    _append_if_set(argv, "--max-model-len", "MAX_MODEL_LEN")
    _append_if_set(argv, "--tensor-parallel-size", "TENSOR_PARALLEL_SIZE")
    _append_if_set(argv, "--data-parallel-size", "DATA_PARALLEL_SIZE")
    _append_if_set(argv, "--block-size", "BLOCK_SIZE")
    _append_if_set(argv, "--max-num-seqs", "MAX_NUM_SEQS")
    max_num_batched_tokens = os.getenv("MAX_NUM_BATCHED_TOKENS")
    if max_num_batched_tokens in (None, ""):
        # Keep profile-time prefill sizing aligned with the configured context limit.
        # FastDeploy's V1 scheduler otherwise defaults to 8192 on GPU, which can
        # over-profile small RunPod smoke-test configs and fail startup.
        max_num_batched_tokens = os.getenv("MAX_MODEL_LEN")
    if max_num_batched_tokens not in (None, ""):
        argv += ["--max-num-batched-tokens", str(max_num_batched_tokens)]
    tokenizer = _get_env("TOKENIZER")
    if tokenizer not in (None, ""):
        argv += ["--tokenizer", _resolve_model(str(tokenizer))]

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
