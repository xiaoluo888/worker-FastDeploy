# Worker FastDeploy - Development Conventions & Architecture Guide

## Project Overview

**worker-FastDeploy** is a RunPod serverless worker that provides OpenAI-compatible endpoints for Large Language Model (LLM) inference, powered by the FastDeploy engine (PaddlePaddle-based). It enables blazing-fast LLM deployment on RunPod's serverless infrastructure with minimal configuration.

### Core Purpose

- **Primary Function**: Deploy any FastDeploy-compatible LLM as an OpenAI-compatible API endpoint
- **Platform**: RunPod Serverless infrastructure
- **Engine**: FastDeploy (high-performance LLM inference engine built on PaddlePaddle and custom kernels)
- **Compatibility**: Drop-in replacement for OpenAI API (Chat Completions, Models)

## High-Level Architecture

### 1. **Entry Point & Request Flow**

```
RunPod Request → handler.py → JobInput → engine.py (AsyncLLM) → Streaming/Buffered Response
```

**Key Components:**

- `src/handler.py`: Main entry point using RunPod serverless framework
- `src/utils.py`: Request parsing and utility classes such as `JobInput`
- `src/engine.py`: Canonical async FastDeploy engine wrapper built on `AsyncLLM`

### 2. **Engine Architecture**

#### Core Classes:

- **`FastDeployEngine`**: Async engine wrapper handling FastDeploy initialization, request normalization, generation, and batching
- **OpenAI Compatibility**: Implemented inside `FastDeployEngine` by translating `job_input.openai_route` and `openai_input` into FastDeploy prompts

#### Key Design Patterns:

- **Dual Input Support**: Same engine handles both OpenAI-compatible requests and raw prompt-style requests
- **Streaming by Default**: Token-level streaming with configurable batching
- **Dynamic Batching**: Adaptive batch sizes that grow from min → max for efficiency

### 3. **Configuration System**

#### Environment-Based Configuration:

- **Single Source of Truth**: All configuration via environment variables
- **Hierarchical Loading**: `DEFAULT_ARGS` → `os.environ` → `local_model_args.json` (for baked models)
- **FastDeploy Argument Mapping**: Automatic translation of env vars to FastDeploy engine arguments

#### Key Configuration Files:

- `src/engine_args.py`: Centralized configuration management
- `src/engine.py`: Async engine lifecycle, request normalization, and streaming/batching behavior
- `src/utils.py`: Request parsing and sampling parameter setup
- `.runpod/hub.json`: Hub UI configuration (CRITICAL: always update when changing defaults)
- `.runpod/tests.json`: RunPod smoke-test payloads and default test env

## Core Development Concepts

### 1. **Deployment Models**

#### Pre-built Docker Installation (Recommended)
For production or quick setup, the recommended approach is to use a pre-built Docker image with FastDeploy GPU pre-installed.

Notice: The pre-built image only supports SM80/SM90 GPUs (e.g. H800 / A800).
If you are deploying on SM86/SM89 GPUs (e.g. L40 / 4090 / L20), you should reinstall fastdeploy-gpu inside the container after creation.


- **Image**: `docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-cuda-12.9:2.5.0`
- **Worker Layer**: This repository's `Dockerfile` adds the RunPod runtime and current `/src` worker code on top of the official FastDeploy base image
- **Override Path**: You can override the default base image at build time with `--build-arg FASTDEPLOY_BASE_IMAGE=...`
- **Configuration**: Entirely via environment variables
- **Model Loading**: Downloads or mounts model at runtime (e.g., from local volume or remote storage)
- **Use Case**: Quick deployment, model experimentation


### 2. **Request Processing Patterns**

#### Input Handling:

```python
class JobInput:
    - llm_input: str | List[Dict] (prompt or messages)
    - sampling_params: SamplingParams (generation settings)
    - stream: bool (streaming vs batch response)
    - openai_route: bool (API compatibility mode)
    - batch_size configs: Dynamic batching parameters
```

#### Response Streaming:

- **Batched Streaming**: Tokens grouped into configurable batch sizes
- **Usage Tracking**: Input/output token counting for billing and analytics

### 3. **Model & Tokenizer Management**

#### Tokenizer Handling:

- **Engine-Managed Tokenization**: Tokenization and chat-template handling are delegated to FastDeploy and the model-specific processors
- **Special Cases**: Certain models may use their own native tokenizer or special chat formatting
- **Chat Templates**: OpenAI-style message input is normalized in `engine.py` before reaching FastDeploy

#### Model Loading:

- **Multi-GPU Support**: Automatic tensor parallelism detection or configuration
- **Quantization**: Support for quantized FastDeploy models (e.g., block_wise_fp8 / W4A8)
- **Caching**: Hugging Face cache management

## Development Patterns & Best Practices

### 1. **Code Organization**

#### File Structure:

```
src/
├── handler.py         # RunPod entry point
├── engine.py          # Canonical async FastDeploy engine wrapper
├── engine_args.py     # Environment → FastDeploy argument mapping
├── utils.py           # Request parsing and shared helpers
├── test_input.json    # Local smoke-test payload
└── __init__.py
```

#### Separation of Concerns:

- **Engine Logic**: Isolated in `engine.py`
- **Configuration**: Centralized in `engine_args.py`
- **Request Handling**: Abstracted via `JobInput` class
- **Platform Integration**: Contained in `handler.py`

### 2. **Error Handling & Logging**

#### Logging Strategy:

- **Structured Logging**: Consistent format across components
- **Error Context**: Detailed error messages with configuration context

#### Error Responses:

- **OpenAI Compatibility**: Standard OpenAI error format
- **Graceful Degradation**: Fallback behaviors for edge cases

### 3. **Environment Variable Conventions**

#### Naming Patterns:

- **FastDeploy Settings**: Match engine parameter names (uppercase)
- **RunPod Settings**: `MAX_CONCURRENCY`, `DEFAULT_BATCH_SIZE`
- **Model Serving Settings**: `MODEL`, `SERVED_MODEL_NAME`, `MAX_MODEL_LEN`, etc.
- **Feature Flags**: `ENABLE_*`, `DISABLE_*` pattern

#### Type Conventions:

- **Booleans**: String 'true'/'false' or int 0/1
- **Lists**: Comma-separated strings
- **Objects**: JSON strings for complex configurations

### 4. **Docker & Deployment**

#### Multi-Stage Builds:

- **Base**: Official FastDeploy runtime image (`fastdeploy-cuda-12.9:2.5.0`)
- **Dependencies**: Add only worker-specific runtime packages such as `runpod`
- **Runtime**: Overlay the current repository's `/src` files and start `handler.py`

#### Build Arguments:

- **MODEL**: Primary model identifier or path
- **QUANTIZATION**: Optimization settings
- **WORKER_CUDA_VERSION**: CUDA compatibility
