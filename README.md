# cudabroker

`cudabroker` is a small local CUDA/GPU lease broker for Python services that load heavy models into VRAM.

It does **not** run inference and does **not** load models for you. Your services still own their models. The broker only answers one question: *is it safe for this process to load or keep this model on GPU right now?*

This is useful when one machine runs several independent GPU projects: Whisper/ASR, image recognition, local LLMs, OCR, video jobs, etc. Without coordination every process sees the same free VRAM, loads optimistically, and the unlucky one gets CUDA OOM. With cudabroker each process asks for a lease, heartbeats while the model is loaded or active, and releases or acknowledges eviction when it unloads.

## What it gives you

- Local FastAPI broker with in-memory state.
- Concurrent GPU leases when VRAM fits.
- Optional strict mode: one GPU lease at a time.
- Idle TTL and heartbeat timeout cleanup.
- Safe eviction: an evicting model still reserves VRAM until `release` / `evict_ack` / heartbeat timeout.
- Idempotent acquire via `worker_id` + `request_id`, so retries do not duplicate queue entries.
- Sync client API: `gpu_lease`, `ManagedModel`.
- Async client API: `gpu_lease_async`, `AsyncManagedModel`.
- Local fallback mode when broker is unavailable, or required mode when you prefer fail-fast.
- GPU memory sampling with `torch`, `nvidia-smi`, or disabled sampling.

## Mental model

```text
service A        service B        service C
   |                |                |
   +---- ask for CUDA lease ----------+
                    |
                    v
              cudabroker server
                    |
          tracks leases, queue,
          VRAM estimates, heartbeats,
          idle state and eviction
```

A typical lifecycle:

1. Service registers/acquires `model_id=whisper-large-v3`, `vram_mb=5200`, `gpu_priority=8`.
2. Broker grants immediately, queues it, or asks a lower-priority idle CPU-capable model to evict.
3. Service loads its model after the lease is granted.
4. Service sends heartbeats: `loading`, `loaded_idle`, `inference_active`, `unloading`.
5. On eviction, service unloads CUDA memory, optionally falls back to CPU, and sends `evict_ack`.
6. On shutdown or TTL unload, service sends `release`.

## Install

From a checkout:

```bash
git clone https://github.com/megamen32/cudabroker.git
cd cudabroker
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

For tests:

```bash
pip install -e '.[test]'
pytest -q
```

## Run the broker

```bash
CUDABROKER_HOST=127.0.0.1 CUDABROKER_PORT=17777 cudabroker
```

Health and state:

```bash
curl -fsS http://127.0.0.1:17777/v1/health
curl -fsS http://127.0.0.1:17777/v1/status
```

The broker is intended for localhost or a trusted LAN. It has no auth layer by itself.

## Sync usage: `ManagedModel`

`ManagedModel` wraps the common pattern: acquire lease, load CUDA model, heartbeat in a background thread, evict to CPU if requested, release on unload.

```python
from faster_whisper import WhisperModel
from cudabroker_client import ManagedModel

whisper_model = ManagedModel(
    model_id="whisper-large-v3",
    loader=lambda: WhisperModel("large-v3", device="cuda", compute_type="float16"),
    vram_mb=5200,
    gpu_priority=8,
    cpu_capable=False,
    ttl_seconds=900,
    client_id="whisper",
)

with whisper_model as model:
    whisper_model.touch(active=True)
    segments, info = model.transcribe("audio.mp3")
    for segment in segments:
        whisper_model.touch(active=True)
        print(segment.text)
```

For small models that can survive on CPU:

```python
small = ManagedModel(
    model_id="whisper-small",
    loader=lambda: WhisperModel("small", device="cuda", compute_type="float16"),
    cpu_fallback=lambda: WhisperModel("small", device="cpu", compute_type="int8"),
    vram_mb=1600,
    gpu_priority=3,
    cpu_capable=True,
)
```

If the broker asks this model to evict while idle, the wrapper unloads CUDA memory, sends `evict_ack`, and calls `cpu_fallback` if provided.

## Async usage: `AsyncManagedModel`

Async mirror exists for asyncio/FastAPI services. The loader and CPU fallback may be either sync functions or async functions.

```python
from cudabroker_client import AsyncManagedModel

async_model = AsyncManagedModel(
    model_id="vision-encoder",
    loader=load_cuda_encoder,          # sync or async callable
    cpu_fallback=load_cpu_encoder,     # optional, sync or async callable
    vram_mb=2400,
    gpu_priority=5,
    cpu_capable=True,
    client_id="vision-api",
)

async with async_model as model:
    await async_model.touch(active=True)
    result = await run_inference(model, image)
```

Lower-level async lease:

```python
from cudabroker_client import gpu_lease_async

async with gpu_lease_async("my-model", vram_mb=2048, gpu_priority=4) as lease:
    await lease.touch(active=False, state="loading")
    model = await load_model()
    await lease.touch(active=False, state="loaded_idle")
    await lease.touch(active=True, state="inference_active")
    output = await infer(model)
```

## Lower-level sync lease

```python
from cudabroker_client import gpu_lease

with gpu_lease("my-model", vram_mb=2048, gpu_priority=4) as lease:
    lease.touch(active=False, state="loading")
    model = load_model()
    lease.touch(active=False, state="loaded_idle")
    lease.touch(active=True, state="inference_active")
    output = infer(model)
```

## Configuration

Client-side variables:

| Variable | Default | Meaning |
|---|---:|---|
| `CUDABROKER_URL` | `http://127.0.0.1:7777` | broker URL used by clients |
| `CUDABROKER_CLIENT_ID` | `default` | logical service name |
| `CUDABROKER_WORKER_ID` | `hostname:pid` | process/worker identity |
| `CUDABROKER_TIMEOUT` | `5` | HTTP timeout |
| `CUDABROKER_HEARTBEAT_INTERVAL` | `5` | client heartbeat interval |
| `CUDABROKER_ACTIVE_GRACE_SECONDS` | `30` | how long `touch(active=True)` keeps the model protected from idle eviction |
| `CUDABROKER_REQUIRED` | `0` | `1` means fail if broker is unavailable; `0` means local fallback |

Common server-side variables:

| Variable | Example | Meaning |
|---|---:|---|
| `CUDABROKER_HOST` | `127.0.0.1` | bind host |
| `CUDABROKER_PORT` | `17777` | bind port |
| `CUDABROKER_HEADROOM_MB` | `512` | safety margin before granting leases |
| `CUDABROKER_STRICT` | `0` | `1` means only one active GPU lease |
| `CUDABROKER_STAT_MODE` | `p98` | footprint statistic mode |
| `CUDABROKER_LEASE_TIMEOUT` | `30` | heartbeat timeout before a lease is dead |
| `CUDABROKER_GPU_SAMPLER` | `auto` | `auto`, `torch`, `nvidia-smi`, or `none` |

## GPU sampler

`CUDABROKER_GPU_SAMPLER=auto` tries torch first and falls back to `nvidia-smi`.

Use `nvidia-smi` when the broker venv has no torch or when workers are not PyTorch-based. Torch is good when the broker sees the same CUDA runtime view as workers, but CTranslate2, llama.cpp, TensorRT and other native runtimes may not report useful memory through torch.

## Failure behavior

By default clients use local fallback:

```bash
CUDABROKER_REQUIRED=0
```

That keeps services alive if the broker is down, but it also disables coordination and can bring back OOM risk.

For production jobs where coordination is mandatory:

```bash
CUDABROKER_REQUIRED=1
```

Then acquire fails if the broker is unavailable.

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/register` | register/update model metadata |
| `POST` | `/v1/acquire` | request a GPU lease |
| `POST` | `/v1/heartbeat` | update lease state and receive eviction status |
| `POST` | `/v1/release` | release a lease |
| `POST` | `/v1/evict_ack` | acknowledge eviction after unloading CUDA memory |
| `GET` | `/v1/status` | broker state, active leases, queue, registered models |
| `GET` | `/v1/health` | health check |

More detail: [docs/architecture.md](docs/architecture.md). Whisper integration example: [docs/whisper-example.md](docs/whisper-example.md).

## What it is not

- Not a scheduler for inference requests.
- Not a model server.
- Not a security boundary.
- Not persistent: leases are in memory and reset when the broker restarts.
- Not a perfect VRAM oracle. You should still provide realistic `vram_mb` values and tune them from real measurements.
