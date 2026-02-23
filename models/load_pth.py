"""
Load PyTorch .pth / .pth.tar checkpoints without requiring torch.
Supports both the new zip-based format and the old legacy format.
Returns a dict with numpy arrays instead of torch.Tensors.
"""
import pickle
import io
import struct
import zipfile
import collections
import numpy as np

_DTYPE_MAP = {
    'FloatStorage':  np.float32,
    'HalfStorage':   np.float16,
    'DoubleStorage': np.float64,
    'LongStorage':   np.int64,
    'IntStorage':    np.int32,
    'ShortStorage':  np.int16,
    'ByteStorage':   np.uint8,
    'BoolStorage':   np.bool_,
}


def _rebuild_tensor(storage, offset, shape, stride, requires_grad, *args):
    arr = storage.data
    if len(shape) == 0:
        return arr[offset:offset + 1].reshape(())
    size = 1
    for s in shape:
        size *= s
    chunk = arr[offset:offset + size]
    try:
        result = np.lib.stride_tricks.as_strided(
            chunk, shape=shape,
            strides=tuple(st * chunk.itemsize for st in stride)
        ).copy()
    except Exception:
        result = chunk.reshape(shape)
    return result


class _TorchStorage:
    def __init__(self, dtype, key):
        self.dtype = dtype
        self.key = key
        self.data = None


def _resolve(obj):
    """Recursively replace ('tensor', ...) tuples with numpy arrays."""
    if isinstance(obj, tuple) and len(obj) == 5 and obj[0] == 'tensor':
        return _rebuild_tensor(*obj[1:])
    if isinstance(obj, dict):
        return {k: _resolve(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_resolve(v) for v in obj)
    return obj


# ── new zip format ────────────────────────────────────────────────────────────

def _load_zip(path):
    storages = {}

    class _Unpickler(pickle.Unpickler):
        def __init__(self, f, zf):
            super().__init__(f)
            self._zf = zf

        def find_class(self, module, name):
            if module == 'torch._utils' and name in ('_rebuild_tensor_v2', '_rebuild_tensor'):
                return _rebuild_tensor
            if module == 'collections' and name == 'OrderedDict':
                return collections.OrderedDict
            if module == '_codecs' and name == 'encode':
                return lambda x, enc: x.encode(enc)
            if module == 'torch' and name in _DTYPE_MAP:
                return name  # return dtype name string
            return super().find_class(module, name)

        def persistent_load(self, pid):
            # pid = ('storage', dtype_name_str, key, location, size)
            storage_cls, key = pid[1], pid[2]
            dtype = _DTYPE_MAP.get(storage_cls if isinstance(storage_cls, str) else 'FloatStorage', np.float32)
            with self._zf.open('archive/data/' + key) as f:
                data = np.frombuffer(f.read(), dtype=dtype).copy()
            s = _TorchStorage(dtype, key)
            s.data = data
            storages[key] = s
            return s

    with zipfile.ZipFile(path) as zf:
        with zf.open('archive/data.pkl') as f:
            obj = _Unpickler(f, zf).load()

    return _resolve(obj)


# ── old legacy format ─────────────────────────────────────────────────────────

def _load_legacy(path):
    with open(path, 'rb') as f:
        content = f.read()

    storages_ref = {}

    # In legacy format, _rebuild_tensor is called during pickle parsing
    # before storage data is read. Return a lazy placeholder instead.
    class _LazyTensor:
        def __init__(self, storage, offset, shape, stride):
            self.storage = storage
            self.offset = offset
            self.shape = shape
            self.stride = stride

    def _lazy_rebuild(storage, offset, shape, stride, requires_grad, *args):
        return _LazyTensor(storage, offset, shape, stride)

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch._utils' and name in ('_rebuild_tensor_v2', '_rebuild_tensor'):
                return _lazy_rebuild
            if module == 'collections' and name == 'OrderedDict':
                return collections.OrderedDict
            if module == '_codecs' and name == 'encode':
                return lambda x, enc: x.encode(enc)
            if module == 'torch' and name in _DTYPE_MAP:
                return name
            return super().find_class(module, name)

        def persistent_load(self, pid):
            if isinstance(pid, tuple) and len(pid) >= 5:
                storage_cls, root_key, location, size = pid[1], pid[2], pid[3], pid[4]
                dtype = _DTYPE_MAP.get(storage_cls if isinstance(storage_cls, str) else 'FloatStorage', np.float32)
                s = _TorchStorage(dtype, root_key)
                storages_ref[root_key] = s
                return s
            return pid

    buf = io.BytesIO(content)
    up = _Unpickler(buf)
    up.load()   # magic
    up.load()   # protocol
    up.load()   # sys_info
    data = up.load()
    pos_after_main = buf.tell()

    remaining = io.BytesIO(content[pos_after_main:])
    storage_keys = _Unpickler(remaining).load()
    pos_after_keys = pos_after_main + remaining.tell()

    # Now read the raw storage data
    raw_pos = pos_after_keys
    for key in storage_keys:
        size = struct.unpack_from('<q', content, raw_pos)[0]
        raw_pos += 8
        s = storages_ref.get(key)
        if s is not None:
            nbytes = size * np.dtype(s.dtype).itemsize
            s.data = np.frombuffer(content[raw_pos:raw_pos + nbytes], dtype=s.dtype).copy()
            raw_pos += nbytes
        else:
            raw_pos += size * 4  # fallback float32

    # Resolve lazy tensors now that storage data is populated
    def _resolve_legacy(obj):
        if isinstance(obj, _LazyTensor):
            return _rebuild_tensor(obj.storage, obj.offset, obj.shape, obj.stride, False)
        if isinstance(obj, dict):
            return {k: _resolve_legacy(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_resolve_legacy(v) for v in obj)
        return obj

    return _resolve_legacy(data)


# ── public API ────────────────────────────────────────────────────────────────

def load_pth(path):
    """
    Load a PyTorch checkpoint without torch.
    Returns a dict where tensor values are numpy arrays.
    """
    if zipfile.is_zipfile(path):
        return _load_zip(path)
    else:
        return _load_legacy(path)
