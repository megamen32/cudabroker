# Whisper integration example

Typical faster-whisper integration:

```python
from faster_whisper import WhisperModel
from cudabroker_client import ManagedModel

mm = ManagedModel(
    model_id="whisper-large-v3",
    loader=lambda: WhisperModel("large-v3", device="cuda", compute_type="float16"),
    vram_mb=5200,
    gpu_priority=8,
    cpu_capable=False,
    ttl_seconds=900,
    client_id="whisper",
)

model = mm.acquire()
mm.touch(active=True)
segments, info = model.transcribe("audio.mp3")

for segment in segments:
    mm.touch(active=True)
    print(segment.text)
```

For tiny/base/small you can add CPU fallback:

```python
mm = ManagedModel(
    model_id="whisper-small",
    loader=lambda: WhisperModel("small", device="cuda", compute_type="float16"),
    cpu_fallback=lambda: WhisperModel("small", device="cpu", compute_type="int8"),
    vram_mb=1600,
    gpu_priority=3,
    cpu_capable=True,
)
```
