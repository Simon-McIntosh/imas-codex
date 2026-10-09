"""A multi-worker embed server gives its workers time to load the model."""

import sys
import types

from imas_codex.cli import embed


def test_multi_worker_server_waits_for_slow_model_loads(monkeypatch):
    calls = []
    fake = types.ModuleType("uvicorn")
    fake.run = lambda *args, **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, "uvicorn", fake)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("IMAS_CODEX_GPU_POOL", "")

    embed._start_foreground(
        host="127.0.0.1",
        port=18765,
        log_level="INFO",
        gpu=None,
        idle_timeout=0,
        deploy_label=None,
        workers=8,
        gpus="0,1,2,3,4,5,6,7",
    )

    assert calls[0]["workers"] == 8
    assert calls[0]["timeout_worker_healthcheck"] >= 600
