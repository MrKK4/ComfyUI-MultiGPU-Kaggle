"""Offline equivalence test for the row-chunked attention path.

This is the test the branch was missing: the chunked path fed `qkv[a:b]` to `_attn`, which splits
its argument into q, k and v, so each band attended only to its own rows -- wrong output, and about
a quarter of the attention work (i.e. a false speedup). The check runs the real `_rope_qk` /
`_attend` / `_attn` methods out of h3_tensor_parallel.py against a plain fp32 reference over full
keys and values, with four bands, and asserts:

  1. the chunked path equals the sequential path (both should be reference attention),
  2. the old slicing (`_attn(qkv[a:b], ...)`) does NOT equal it -- so a regression to that shape
     fails this test instead of shipping,
  3. rope is applied exactly once per row even though attention runs band by band.

No torch needed: the few tensor methods these paths use are emulated, and the reference attention is
plain Python over nested lists.

    python check_tp_attn.py        # exits 1 on any mismatch
"""
import ast
import math
import sys

import numpy as np

SRC = "h3_tensor_parallel.py"
S = 64.0
HEAD = 128


# --------------------------------------------------------------------------- tiny tensors

class T:                                    # noqa: N801 - a tensor, not a test case
    def __init__(self, array, scale=1.0):
        self.a = np.asarray(array, dtype="float32")

    # -- shape / view ops used by the attention path
    @property
    def shape(self):
        return self.a.shape

    def split(self, size, dim=-1):
        """Plain slices: torch's split returns views, and _rope_qk relies on that (rope is applied
        to views of q/k and must land in the qkv storage the bands are read from)."""
        assert dim == -1
        return tuple(T(self.a[..., i:i + size]) for i in range(0, self.a.shape[-1], size))

    def view(self, *shape):
        return T(self.a.reshape(shape))

    def __getitem__(self, idx):
        return T(self.a[idx])

    def transpose(self, d0, d1):
        return T(np.swapaxes(self.a, d0, d1))

    def unsqueeze(self, dim):
        return T(np.expand_dims(self.a, dim))

    def squeeze(self, dim):
        return T(np.squeeze(self.a, axis=dim))

    def reshape(self, *shape):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        return T(self.a.reshape(shape))

    def contiguous(self):
        return T(np.ascontiguousarray(self.a))

    def transpose_(self, d0, d1):
        self.a = np.swapaxes(self.a, d0, d1)
        return self

    def float(self):
        return self

    def clone(self):
        return T(self.a.copy())

    def dim(self):
        return self.a.ndim

    def numel(self):
        return self.a.size

    def mul_(self, x):
        self.a = self.a * x
        return self

    def __add__(self, other):
        return T(self.a + getattr(other, "a", other))

    __radd__ = __add__

    @property
    def T_(self):                            # pragma: no cover - not used
        return self


def _reference_attention(q, k, v, heads, scale):
    """Plain fp32 softmax attention, q/k/v as nested (1, heads, s, HEAD) arrays."""
    out = np.zeros_like(q.a)
    for h in range(heads):
        qh, kh, vh = q.a[0, h].astype("float64"), k.a[0, h].astype("float64"), v.a[0, h].astype("float64")
        scores = (qh @ kh.T) * scale
        scores -= scores.max(axis=-1, keepdims=True)
        p = np.exp(scores)
        p /= p.sum(axis=-1, keepdims=True)
        out[0, h] = (p @ vh).astype("float32")
    return T(out)


# --------------------------------------------------------------------------- the harness

class _Ck:
    """Stands in for comfy.quant_ops.ck; rope here is a scale, so double-roping is detectable."""

    calls = 0

    @staticmethod
    def rms_rope_split_half_(q, k, rope, qn, kn, epsilon=None, rot_dim=None):
        _Ck.calls += 1
        rope = np.asarray(getattr(rope, "a", rope), dtype="float32")
        for t in (q, k):
            t.a *= rope            # in place on a view: multiplicative, so rope^2 != rope, and it
        return q, k                # must land in the storage the bands read (checked in main())


class _AttnContainer:
    def __init__(self, t):
        self.t = t


def _optimized_attention(q, k, v, heads, mask=None, skip_reshape=True, transformer_options=None):
    q = q.t if isinstance(q, _AttnContainer) else q            # containers exist to keep the
    k = k.t if isinstance(k, _AttnContainer) else k            # (1, heads, s, HEAD) layout; unwrap
    v = v.t if isinstance(v, _AttnContainer) else v
    scale = 1.0 / math.sqrt(q.shape[-1])
    out = _reference_attention(q, k, v, heads, scale)
    # comfy's contract: (B, heads, S, HEAD) in, (B, S, heads * HEAD) out
    return T(out.a.transpose(0, 2, 1, 3).reshape(out.a.shape[0], out.a.shape[2], heads * out.a.shape[3]))


class _Comfy(types.ModuleType if False else object):        # placeholder replaced below
    pass


def load_methods():
    """Extract and compile _rope_qk / _attend / _attn from the real class body."""
    import types
    src = open(SRC).read()
    tree = ast.parse(src)
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "H3TensorParallel")
    keep = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name in ("_rope_qk", "_attend", "_attn")]
    assert len(keep) == 3, f"expected 3 methods, found {[n.name for n in keep]}"
    mod = ast.Module(body=[ast.ClassDef(name="Extracted", bases=[], keywords=[], decorator_list=[], body=keep)],
                     type_ignores=[])
    # inner class so `cls` + `super()` semantics stay irrelevant; compile it standalone
    code = compile(ast.fix_missing_locations(mod), SRC, "exec")
    ns = {"HEAD": HEAD, "comfy": types.SimpleNamespace(quant_ops=types.SimpleNamespace(ck=_Ck)),
          "AttentionTensorContainer": _AttnContainer, "optimized_attention": _optimized_attention,
          "_attention_backend": lambda: None,      # exercise the default ComfyUI path
          "np": np, "torch": types.SimpleNamespace()}
    exec(code, ns)
    cls_obj = ns["Extracted"]
    obj = cls_obj()
    obj.qk_eps = 1e-6
    return obj


def main():
    rng = np.random.default_rng(3)
    n_tokens, heads, inner = 64, 4, 4 * HEAD          # 4 heads per rank, full sequence 64 tokens
    qkv = T(rng.normal(size=(n_tokens, 3 * inner)) * 0.5)
    # a per-token rope that is not the identity, so "roped twice" cannot look like "roped once"
    rope = T(0.5 + rng.random((n_tokens, 1, 1)).astype("float32"))

    obj = load_methods()
    scale = HEAD ** -0.5

    # sequential path: rope once, attend all rows
    plain = qkv.clone()
    seq = obj._attn(plain, {"heads": heads, "inner": inner, "qn": None, "kn": None}, rope, None)
    assert _Ck.calls == 1, f"sequential path roped {_Ck.calls} times"
    roped_qk = np.abs(plain.a[:, :2 * inner] - qkv.a[:, :2 * inner]).max()
    assert roped_qk > 0, "harness is not aliasing: rope did not modify the qkv storage"

    # chunked path, as the fixed block() does it: rope once, then attend query bands
    banded = qkv.clone()
    obj._rope_qk(banded, {"heads": heads, "inner": inner, "qn": None, "kn": None}, rope)
    assert _Ck.calls == 2, f"rope applied {_Ck.calls} times in total"
    bands = []
    for a in range(0, n_tokens, 16):
        bands.append(obj._attend(banded, {"heads": heads, "inner": inner}, None, (a, a + 16)))
    chunked = T(np.concatenate([b.a for b in bands], axis=0))

    # the buggy shape: hand each band its own rows (q, k and v all cut to the band)
    buggy = []
    for a in range(0, n_tokens, 16):
        buggy.append(obj._attn(qkv[a:a + 16].clone(), {"heads": heads, "inner": inner, "qn": None, "kn": None},
                               rope[a:a + 16], None))
    buggy = T(np.concatenate([b.a for b in buggy], axis=0))

    checks = []
    d_seq_chunk = float(np.abs(seq.a - chunked.a).max())
    checks.append(("chunked == sequential (max abs %.2e)" % d_seq_chunk, d_seq_chunk < 1e-5))
    d_seq_bug = float(np.abs(seq.a - buggy.a).max())
    rel = d_seq_bug / max(float(np.abs(seq.a).max()), 1e-9)
    checks.append(("old per-band slicing differs, as it must (max abs %.3f, %.0f%% of scale)"
                   % (d_seq_bug, rel * 100), rel > 0.05))
    # and rope must be applied once per row: roping twice would scale the keys by rope**2
    twice = qkv.clone()
    obj._rope_qk(twice, {"heads": heads, "inner": inner, "qn": None, "kn": None}, rope)
    obj._rope_qk(twice, {"heads": heads, "inner": inner, "qn": None, "kn": None}, rope)
    d_double = float(np.abs(obj._attend(twice, {"heads": heads, "inner": inner}, None, None).a
                            - seq.a).max())
    checks.append(("double rope is detectable (max abs %.2e)" % d_double, d_double > 1e-3))
    # structural guard: the chunked path must rope once for the whole sequence, not per band
    calls_before = _Ck.calls
    banded2 = qkv.clone()
    obj._rope_qk(banded2, {"heads": heads, "inner": inner, "qn": None, "kn": None}, rope)
    for a0 in range(0, n_tokens, 16):
        obj._attend(banded2, {"heads": heads, "inner": inner}, None, (a0, a0 + 16))
    checks.append(("chunked path ropes exactly once for %d bands" % (n_tokens // 16),
                   _Ck.calls - calls_before == 1))

    for name, ok in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    print("\nreference: bands of 16 over %d tokens, %d heads, head_dim %d" % (n_tokens, heads, HEAD))
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
