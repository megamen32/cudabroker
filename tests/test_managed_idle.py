"""Fast unit; expected1s/max10s. Detect immortal active heartbeat and early
lease release before model finalizers free GPU memory. No model or GPU load."""
import time
import pytest
from cudabroker_client.managed import ManagedModel
from cudabroker_client.async_managed import AsyncManagedModel

class Lease:
    def __init__(self):self.calls=[]
    def touch(self,**kwargs):self.calls.append(kwargs);return 'active'
    def release(self):self.calls.append({'released':True})

class AsyncLease(Lease):
    async def touch(self,**kwargs):return super().touch(**kwargs)
    async def release(self):return super().release()


def test_idle_after_end_has_no_activity_grace():
    model=ManagedModel('test',lambda:object())
    model.model=object();model.current_device='cuda';model.lease=Lease()
    model.touch(active=True)
    model.begin_inference();model.end_inference()
    assert model._active_count==0
    assert model._active_until<=time.time()
    assert model.lease.calls[-1]['active'] is False


def test_heartbeat_does_not_extend_its_own_activity(monkeypatch):
    import cudabroker_client.managed as module
    now=[0.0];monkeypatch.setattr(module.time,'time',lambda:now[0])
    model=ManagedModel('test',lambda:object());model.lease=Lease()
    model.touch(active=True)
    class Stop:
        def wait(self,interval):
            now[0]+=10
            return now[0]>60
    model._stop=Stop()
    model._heartbeat_loop()
    assert any(not c['active'] for c in model.lease.calls if 'active' in c)
    assert model._active_until==30


@pytest.mark.asyncio
async def test_async_idle_after_end_has_no_activity_grace():
    model=AsyncManagedModel('test',lambda:object())
    model.model=object();model.current_device='cuda';model.lease=AsyncLease()
    await model.touch(active=True)
    await model.begin_inference();await model.end_inference()
    assert model._active_count==0
    assert model._active_until<=time.time()
    assert model.lease.calls[-1]['active'] is False


def test_release_frees_memory_before_broker_can_promote(monkeypatch):
    model=ManagedModel('test',lambda:object());model.model=object()
    lease=model.lease=Lease()
    monkeypatch.setattr(model,'_empty_cuda_cache',lambda:lease.calls.append({'memory_freed':True}))
    model.release()
    assert lease.calls.index({'memory_freed':True})<lease.calls.index({'released':True})


@pytest.mark.asyncio
async def test_async_release_frees_memory_before_broker_can_promote(monkeypatch):
    model=AsyncManagedModel('test',lambda:object());model.model=object()
    lease=model.lease=AsyncLease()
    monkeypatch.setattr(model,'_empty_cuda_cache',lambda:lease.calls.append({'memory_freed':True}))
    await model.release()
    assert lease.calls.index({'memory_freed':True})<lease.calls.index({'released':True})
