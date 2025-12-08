![FastDeploy worker banner](https://github.com/user-attachments/assets/42b0039f-39e3-4279-afda-6d1865dfbffb)

Run LLMs using [Fastdeploy](https://github.com/PaddlePaddle/FastDeploy) with an OpenAI-compatible API

---

[![Runpod](https://api.runpod.io/badge/xiaoluo888/runpod)](https://www.console.runpod.io/hub/xiaoluo888/runpod)

---



## Endpoint Configuration

All behaviour is controlled through environment variables:

| Environment Variable                | Description                                       | Default                       | Options                                                            |
| ----------------------------------- | ------------------------------------------------- | ----------------------------- | ------------------------------------------------------------------ |
| `MODEL`                             | Path of the model weights                         | "baidu/ERNIE-4.5-0.3B-Paddle" | Local folder or Hugging Face repo ID                               |
| `MAX_MODEL_LEN`                     | Model's maximum context length                    |                               | Integer (e.g., 4096)                                               |
| `QUANTIZATION`                      | Quantization method                               |                               | Model quantization strategy, when loading BF16 CKPT, specifying wint4 or wint8 supports lossless online 4bit/8bit quantization |
| `TENSOR_PARALLEL_SIZE`              | Number of GPUs                                    | 1                             | Integer                                                            |
| `GPU_MEMORY_UTILIZATION`            | Fraction of GPU memory to use                     | 0.9                           | Float between 0.0 and 1.0                                          |
| `MAX_NUM_SEQS`                      | Maximum number of sequences per iteration         | 8                             | Integer                                                            |


For complete configuration options, see the [full configuration documentation](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/parameters.md).

## API Usage

This worker supports two API formats: **RunPod native** and **OpenAI-compatible**.

### RunPod Native API

For testing directly in the RunPod UI, use these examples in your endpoint's request tab.

#### Chat Completions

```json
{
  "input": {
    "messages": [
      { "role": "system", "content": "You are a helpful assistant." },
      { "role": "user", "content": "What is the capital of France?" }
    ],
    "sampling_params": {
      "max_tokens": 100,
      "temperature": 0.7
    }
  }
}
```

#### Chat Completions (Streaming)

```json
{
  "input": {
    "messages": [
      { "role": "user", "content": "Write a short story about a robot." }
    ],
    "sampling_params": {
      "max_tokens": 500,
      "temperature": 0.8
    },
    "stream": true
  }
}
```

#### Text Generation

For direct text generation without chat format:

```json
{
  "input": {
    "prompt": "The capital of France is",
    "sampling_params": {
      "max_tokens": 64,
      "temperature": 0.0
    }
  }
}
```

#### List Models

```json
{
  "input": {
    "openai_route": "/v1/models"
  }
}
```

---

### OpenAI-Compatible API

For external clients and SDKs, use the `/openai/v1` path prefix with your RunPod API key.

#### Chat Completions

**Path:** `/openai/v1/chat/completions`

```json
{
  "model": "baidu/ERNIE-4.5-0.3B-Paddle",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "What is the capital of France?" }
  ],
  "max_tokens": 100,
  "temperature": 0.7
}
```

#### Chat Completions (Streaming)

```json
{
  "model": "baidu/ERNIE-4.5-0.3B-Paddle",
  "messages": [
    { "role": "user", "content": "Write a short story about a robot." }
  ],
  "max_tokens": 500,
  "temperature": 0.8,
  "stream": true
}
```

#### Text Completions

**Path:** `/openai/v1/completions`

```json
{
  "model": "baidu/ERNIE-4.5-0.3B-Paddle",
  "prompt": "The capital of France is",
  "max_tokens": 100,
  "temperature": 0.7
}
```

#### List Models

**Path:** `/openai/v1/models`

```json
{}
```

#### Response Format

Both APIs return the same response format:

```json
{
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "Paris." },
      "finish_reason": "stop"
    }
  ],
  "usage": { "prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10 }
}
```

---

## Usage

Below are minimal `python` snippets so you can copy-paste to get started quickly.

> Replace `<ENDPOINT_ID>` with your endpoint ID and `<API_KEY>` with a [RunPod API key](https://docs.runpod.io/get-started/api-keys).

### OpenAI compatible API

Minimal Python example using the official `openai` SDK:

```python
from openai import OpenAI
import os

# Initialize the OpenAI Client with your RunPod API Key and Endpoint URL
client = OpenAI(
    api_key=os.getenv("RUNPOD_API_KEY"),
    base_url=f"https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1",
)
```

`Chat Completions (Non-Streaming)`

```python
response = client.chat.completions.create(
    model="baidu/ERNIE-4.5-0.3B-Paddle",
    messages=[{"role": "user", "content": "Explain quantum computing in simple terms"}],
    temperature=0,
    max_tokens=100,
)
print(f"Response: {response.choices[0].message.content}")
```

`Chat Completions (Streaming)`

```python
response_stream = client.chat.completions.create(
    model="baidu/ERNIE-4.5-0.3B-Paddle",
    messages=[{"role": "user", "content": "Explain quantum computing in simple terms"}],
    temperature=0,
    max_tokens=100,
    stream=True
)
for response in response_stream:
    print(response.choices[0].delta.content or "", end="", flush=True)
```

### RunPod Native API

```python
import requests

response = requests.post(
    "https://api.runpod.ai/v2/<ENDPOINT_ID>/run",
    headers={"Authorization": "Bearer <API_KEY>"},
    json={
        "input": {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain quantum computing in simple terms"}
            ],
            "sampling_params": {
                "temperature": 0.7,
                "max_tokens": 150
            }
        }
    }
)

result = response.json()
print(result["output"])
```

## Compatibility

For supported models, see the [FastDeploy supported models documentation](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/supported_models.md).

