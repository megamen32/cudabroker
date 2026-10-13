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
