# cudabroker

Small local GPU lease broker for Python/CUDA services.

It helps several independent processes share one GPU without loading too many models at once. Each service asks for a CUDA lease before loading a model, sends heartbeats while it is active, and releases the lease when the model is unloaded.

## Features

- FastAPI broker with in-memory state.
- Concurrent loading when VRAM fits.
- Strict mode: one GPU model at a time.
- TTL-based idle eviction.
- Safe eviction: evicting leases still reserve VRAM until ack/release.
- Idempotent acquire with worker_id and request_id.
- Client context manager and ManagedModel wrapper.
- Local-mode fallback when broker is unavailable.
- GPU memory sampling through torch, with nvidia-smi fallback.

## Quick start

Run broker:

```bash
CUDABROKER_PORT=17777 cudabroker
```

Health check:

```bash
curl http://127.0.0.1:17777/v1/health
curl http://127.0.0.1:17777/v1/status
```

Use in code:

```python
from cudabroker_client import ManagedModel

model = ManagedModel(
    "whisper-large-v3",
    loader=lambda: WhisperModel("large-v3", device="cuda", compute_type="float16"),
    vram_mb=5200,
    gpu_priority=8,
    cpu_capable=False,
    ttl_seconds=900,
)

whisper = model.acquire()
model.touch(active=True)
segments, info = whisper.transcribe(path)
```

## Docs

See docs/architecture.md and docs/whisper-example.md.

