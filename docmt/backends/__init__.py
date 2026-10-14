from __future__ import annotations

from .base import BACKENDS, Backend

__all__ = ["BACKENDS", "Backend", "make_backend"]

def make_backend(name: str, cfg=None, tok=None, **kw) -> Backend:
    """Factory used by the CLI/runtime. Extra keyword arguments go to the backend class."""
    if name == "hf":
        from .hf import HFBackend
        return HFBackend(getattr(cfg, "model", ""), cfg, tok, **kw)
    if name == "mock":
        from .mock import MockBackend
        opts = dict((getattr(cfg, "extra", None) or {}).get("mock", {}))
        opts.update(kw)
        return MockBackend(tok=tok, **opts)
    raise ValueError(f"unknown backend {name!r}; expected one of {BACKENDS}")
