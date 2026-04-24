import importlib

__all__ = [
    "RQVAEModel",
    "RQVAEV2Model",
    "RQKMeansModel",
    "RQVAEV3Model",
    "RVQModel",
]

_MODULE_MAP = {
    "RQVAEModel": ".rq_vae",
    "RQVAEV2Model": ".rq_vae_v2",
    "RQKMeansModel": ".rq_kmeans",
    "RQVAEV3Model": ".rq_vae_v3",
    "RVQModel": ".rvq",
}


def __getattr__(name):
    if name in _MODULE_MAP:
        module = importlib.import_module(_MODULE_MAP[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
