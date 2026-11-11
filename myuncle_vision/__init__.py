"""MyUncleVision: a reusable lattice-based framework for unsupervised anomaly discovery in images (computer-vision sister of the MyUncle maintenance-lattice core)."""
__version__ = "0.9.0"


def discover_images(*args, **kwargs):
    """see myuncle_vision.api.discover_images (a function name 'discover' would shadow the submodule myuncle_vision.discover)"""
    from .api import discover_images as f
    return f(*args, **kwargs)


def discover_arrays(*args, **kwargs):
    """see myuncle_vision.api.discover_arrays"""
    from .api import discover_arrays as f
    return f(*args, **kwargs)


def CallableEncoder(*args, **kwargs):
    """see myuncle_vision.encoder.CallableEncoder"""
    from .encoder import CallableEncoder as c
    return c(*args, **kwargs)
