#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/src"
TEST_SPEC="$ROOT_DIR/.runpod/tests.json"

RUNPOD_ENV_PREFIX="${RUNPOD_ENV_PREFIX:-/root/miniconda3/envs/runpod}"
PYTHON_BIN="${PYTHON_BIN:-$RUNPOD_ENV_PREFIX/bin/python}"
TEST_NAME_FILTER="${1:-${SMOKE_TEST_NAME:-}}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  echo "Set RUNPOD_ENV_PREFIX or PYTHON_BIN before running this script." >&2
  exit 1
fi

if [[ ! -f "$TEST_SPEC" ]]; then
  echo "Test spec not found: $TEST_SPEC" >&2
  exit 1
fi

SYSTEM_LIB_DIRS="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"

export CONDA_PREFIX="${CONDA_PREFIX:-$RUNPOD_ENV_PREFIX}"
export PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$RUNPOD_ENV_PREFIX/lib:$SYSTEM_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

export MODEL="${MODEL:-/root/PaddlePaddle/ERNIE-4.5-0.3B-Paddle}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$MAX_MODEL_LEN}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
export ENABLE_V1_KVCACHE_SCHEDULER="${ENABLE_V1_KVCACHE_SCHEDULER:-1}"

export SMOKE_TEST_SPEC="$TEST_SPEC"
export SMOKE_TEST_NAME="$TEST_NAME_FILTER"

cd "$ROOT_DIR"

exec "$PYTHON_BIN" - <<'PY'
import asyncio
import json
import os
import sys
import traceback

import runpod.serverless

# Import handler without booting the long-running RunPod worker loop.
runpod.serverless.start = lambda config: None

if os.getenv("SMOKE_BYPASS_PORT_SCAN", "").strip().lower() in {"1", "true", "yes", "on"}:
    import fastdeploy.engine.args_utils as fd_args_utils

    base_port = int(os.getenv("SMOKE_PORT_BASE", "43000"))

    def fake_find_free_ports(num_ports=1, host="0.0.0.0", port_range=(10000, 65535)):
        return list(range(base_port, base_port + num_ports))

    fd_args_utils.find_free_ports = fake_find_free_ports
    fd_args_utils.is_port_available = lambda host, port: True

test_spec = os.environ["SMOKE_TEST_SPEC"]
name_filter = os.getenv("SMOKE_TEST_NAME", "").strip()

try:
    import handler as handler_module
    print("IMPORT_OK")
except Exception:
    print("IMPORT_FAILED")
    traceback.print_exc()
    raise

with open(test_spec, "r", encoding="utf-8") as f:
    spec = json.load(f)

tests = spec.get("tests", [])
if name_filter:
    tests = [test for test in tests if test.get("name") == name_filter]
    if not tests:
        print(f"NO_TEST_MATCHED name={name_filter}")
        sys.exit(1)

failures = []


def summarize(result):
    text = json.dumps(result, ensure_ascii=False)
    return text if len(text) <= 1500 else text[:1500] + "...(truncated)"


async def run_test(test):
    name = test["name"]
    payload = {"id": f"local-{name}", "input": test["input"]}
    timeout_s = min(int(test.get("timeout", 100000)) / 1000.0, 100.0)
    print(f"TEST_START {name} timeout={timeout_s}")

    try:
        result = await asyncio.wait_for(handler_module.handler(payload), timeout=timeout_s)
    except Exception as exc:
        failures.append(name)
        print(f"TEST_EXCEPTION {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return

    if isinstance(result, dict) and "error" in result:
        failures.append(name)
        print(f"TEST_ERROR {name}")
        print(summarize(result))
        return

    print(f"TEST_OK {name}")
    print(summarize(result))


async def main():
    for test in tests:
        await run_test(test)


asyncio.run(main())

if failures:
    print(f"SMOKE_TESTS_FAILED count={len(failures)} names={','.join(failures)}")
    sys.exit(1)

print(f"SMOKE_TESTS_PASSED count={len(tests)}")
PY
