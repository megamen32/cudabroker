# Managed model idle and release

Canonical SDK source reconciled with the deployed WhisperServer client:
managed.py, async_managed.py, lease.py. Preserve the already-deployed inference
context, lost-lease handling and managed CPU fallback; no server code change.

Confirmed by five failing fast unit checks: a heartbeat renewed its own activity
grace forever, end_inference retained activity grace, and release acknowledged
VRAM availability before running CUDA object finalizers. Fix both sync/async
clients; preserve active_count for nested inference and existing GPU admission.

Budget on server-88 for focused SDK checks: RAM soft/hard1/2 GiB, swap0,
CPU100%, tasks128, RuntimeMaxSec30, temp/disk128 MiB, no GPU/model load.
Observed initial five-check reproduction0.16s, whole unit0.988s. Purpose,
category and timing are recorded in tests/test_managed_idle.py. Installation
into the WhisperServer venv changes only its consumer dependency, not the
running broker, foreign models, or other project environments.

Post-install consumer: WhisperServer deployed2e64328 with the pinned0.2.1
client.13 SDK checks passed in0.14s; five previously failing lifecycle checks
now pass. Fresh public API observer captured Community-1 inference_active and
its release; two actual CUDA load/unload cycles left zero model bytes above
an unchanged8.125MiB process workspace. Full63-check Whisper release plus
restart and public API canary took57.89s. Private user recordings were also
processed by the same worker on CUDA. Evidence belongs to WhisperServer:
.tmp/diarization-practice-20261013/gpu-api-accepted.json and gpu-canary.json.
No broker server deployment or foreign model restart was needed for this fix.

## Lease handoff after loading — 2026-10-18

WhisperServer logs showed a granted Large V3 lease being marked evicted between
`ManagedModel.acquire()` and `inference()`. The client cleared its model during
that handoff, then returned `None`, which surfaced later as “model is not
loaded”. Both sync and async clients now protect the handoff with an active
lease and raise immediately if the broker has already revoked it. Focused
regression and eviction lifecycle checks passed (4 checks, 0.10s). The broker
server and unrelated GPU clients were not restarted or changed.
