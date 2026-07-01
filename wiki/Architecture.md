# Architecture

cudabroker is a local coordinator for CUDA model processes. It does not load models itself. Models stay inside your services. The broker only tracks metadata, leases, GPU memory, admission decisions, heartbeat state, and eviction.

## Components

```text
service process  ->  cudabroker-client  ->  cudabroker FastAPI server
                                                |
                                                +-- GPU memory sampler
```

The broker is intentionally simple:

- no Redis;
- no external database;
- in-memory leases;
- optional systemd deployment;
- designed for localhost or trusted LAN use.

## Model

A logical model registration:

```text
model_id
client_id
declared_vram_mb
gpu_priority
cpu_capable
ttl_seconds
```

## Lease

A concrete permission to load or keep a model on GPU:

```text
lease_id
model_id
client_id
worker_id
request_id
status
state
last_heartbeat
last_active
last_vram_sample
```

request_id makes acquire idempotent. Retrying the same acquire request does not create duplicate queue entries.

## Lease statuses

```text
queued     waiting for VRAM
active     allowed to use GPU
evicting   broker asked client to unload; VRAM is still reserved
released   client released or acknowledged eviction
dead       heartbeat timeout
```

Important: evicting leases still reserve VRAM. This prevents the classic bug where the broker asks A to evict, immediately grants B, A has not unloaded yet, and B hits CUDA OOM.

## Runtime states

```text
loading
loaded_idle
inference_active
unloading
cpu
released
```

The broker evicts only idle loaded models. It does not evict a model while it is loading or actively running inference.

## Admission policy

Default mode is concurrent-when-fits:

```text
F = footprint(requested_model)
reserved = sum(footprint(active or evicting leases))
live_room = current GPU free memory
available = min(live_room, total - reserved)

grant if available >= F + headroom
otherwise queue or request eviction
```

Strict mode:

```bash
CUDABROKER_STRICT=1
```

In strict mode, only one GPU lease can be active at a time.

## Eviction policy

When there is not enough space, the broker prefers:

1. idle cpu_capable models with lowest priority;
2. other idle cpu_capable models with lowest priority;
3. queue if nothing safe can be evicted.

Models in inference_active, loading, or unloading are not eviction candidates.

## GPU memory estimation

The broker combines declared model footprint from the client and live GPU memory sampling from torch or nvidia-smi.

Client-side PyTorch memory deltas are useful but not always enough. Native CUDA libraries may not be fully visible through torch.cuda.memory_allocated(). For those, prefer explicit vram_mb values and tune them from real observations.

## Failure modes

By default, the client falls back to local mode:

```bash
CUDABROKER_REQUIRED=0
```

This keeps services alive but removes coordination. For safer production behavior:

```bash
CUDABROKER_REQUIRED=1
```

Then a missing broker fails lease acquisition instead of silently loading on GPU.

If a worker stops sending heartbeats for longer than CUDABROKER_LEASE_TIMEOUT, the broker marks the lease dead and can promote queued work.

## HTTP API

Register:

```http
POST /v1/register
```

Acquire:

```http
POST /v1/acquire
```

Heartbeat:

```http
POST /v1/heartbeat
```

Release and eviction ack:

```http
POST /v1/release
POST /v1/evict_ack
```

Status:

```http
GET /v1/status
```

## systemd example

```ini
[Unit]
Description=cudabroker GPU lease broker
After=network.target

[Service]
Type=simple
User=roomhacker
WorkingDirectory=/opt/cudabroker
Environment=CUDABROKER_HOST=127.0.0.1
Environment=CUDABROKER_PORT=17777
Environment=CUDABROKER_HEADROOM_MB=512
Environment=CUDABROKER_STAT_MODE=p98
Environment=CUDABROKER_STRICT=0
ExecStart=/opt/cudabroker/.venv/bin/python -m cudabroker.server
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```
