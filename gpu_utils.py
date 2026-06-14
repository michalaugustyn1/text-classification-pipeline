import os
import numpy as np

def _try_cupy():
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        return cp
    except Exception:
        return None

_cp = _try_cupy()
_FORCE_CPU = os.environ.get("FORCE_CPU", "0") == "1"

if _cp is not None and not _FORCE_CPU:
    xp      = _cp
    USE_GPU = True
else:
    xp      = np
    USE_GPU = False


def to_numpy(arr) -> np.ndarray:
    if USE_GPU and isinstance(arr, _cp.ndarray):
        return _cp.asnumpy(arr)
    return np.asarray(arr)


def to_xp(arr, dtype=None):
    if dtype is not None:
        return xp.array(np.asarray(arr), dtype=dtype)
    return xp.array(np.asarray(arr))


def seed(s: int):
    np.random.seed(s)
    if USE_GPU:
        _cp.random.seed(s)
