"""Regression test for the exchange's event reuse (the second blocking bug in review).

`_ChunkedExchange` reuses event objects instead of creating one per record. That is only safe where
the wait consuming an event is enqueued before the event is recorded again. `sent` and `read` are
recorded per chunk and waited on *later* (the block sends every chunk, then receives every chunk), so
sharing one object across chunks binds all the receives to the last chunk's record: results stay
correct, but the chunk-by-chunk overlap the exchange exists for is gone.

Because a stream wait binds to the event's most recent record *at enqueue time* (that is the CUDA
semantic this code depends on), the bug is invisible to a check that only compares the op sequence --
which is exactly why the earlier verification missed it. This test models the binding, runs the real
`_record` + `_ChunkedExchange` source twice (once with the per-chunk keys, once with the shared keys
the review was written against), and asserts:

  1. fixed: receive of chunk k depends on send of chunk k only,
  2. old: every receive depends on the last chunk's send (the regression, reproduced),
  3. pooling still works: after the first block, no new event objects are created,
  4. the cross-block host-buffer guard still waits on the last chunk's read.

No torch: streams, events and pinned buffers are emulated, with events carrying identity and a record
history.

    python check_tp_events.py        # exits 1 if any of the four properties fails
"""
import ast
import sys

SRC = "h3_tensor_parallel.py"
BLOCK = 2 * 1024          # rows in one phase's buffer
CHUNKS = 4
SIZE = BLOCK // CHUNKS
PHASE = "attn"
SHAPE = (BLOCK, 8)


# --------------------------------------------------------------------------- fake torch

class FakeTensor:
    def __init__(self, shape, dtype=None, device=None):
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def __getitem__(self, idx):
        a, b = (0, self.shape[0]) if idx == slice(None) else (idx.start, idx.stop or self.shape[0])
        return FakeTensor((b - a,) + self.shape[1:], self.dtype, self.device)

    def copy_(self, src, non_blocking=False):
        LOG.append(("copy", self.device, src.device, self.shape[0]))
        return self


class FakeRecord:
    """A record of an event on a stream. `tag` is set by the test to name the logical step."""

    def __init__(self, seq, stream, tag):
        self.seq = seq
        self.stream = stream
        self.tag = tag

    def __repr__(self):
        return f"<rec{self.seq}:{self.tag}>"


class FakeEvent:
    def __init__(self):
        EVENTS_CREATED.append(1)
        self.records = []

    def record(self, stream=None):
        rec = FakeRecord(len(LOG), stream, CONTEXT[0])
        self.records.append(rec)
        LOG.append(("record", id(self), rec.seq, rec.tag))
        return rec

    @property
    def latest(self):
        return self.records[-1] if self.records else None


class FakeStream:
    def __init__(self, name):
        self.name = name
        self.waits = []

    def wait_event(self, ev):
        bound = ev.latest
        if bound is None:
            LOG.append(("wait-unrecorded", self.name, id(ev)))
            return None
        self.waits.append(bound)
        LOG.append(("wait", self.name, bound.seq, bound.tag))
        return bound


LOG = []
CONTEXT = ["init"]                 # tag applied to the next record()
EVENTS_CREATED = []
DEVICES = ["cuda:0", "cuda:1"]
_DEFAULT_STREAMS = {}


def _stream(dev):
    return _DEFAULT_STREAMS.setdefault(dev, FakeStream(f"compute:{dev}"))


class _Ctx:
    def __init__(self, obj):
        self.obj = obj

    def __enter__(self):
        return self.obj

    def __exit__(self, *exc):
        return False


class FakeCuda:
    @staticmethod
    def Stream(dev):
        return FakeStream(f"copy:{dev}")

    @staticmethod
    def current_stream(dev):
        return _stream(dev)

    @staticmethod
    def device(dev):
        return _Ctx(dev)

    @staticmethod
    def stream(cs):
        return _Ctx(cs)

    @staticmethod
    def Event():
        return FakeEvent()


def fake_torch():
    import types
    torch = types.ModuleType("torch")
    torch.cuda = FakeCuda()
    torch.float16 = "float16"
    torch.empty = lambda shape, **kw: FakeTensor(tuple(shape), kw.get("dtype"), kw.get("device"))
    return torch


# --------------------------------------------------------------------------- harness

def load(shared_keys=False):
    """Compile `_record` + `_ChunkedExchange` from the module, optionally reverted to shared keys."""
    src = open(SRC).read()
    if shared_keys:
        src = (src.replace('key=("sent", phase, r, a)', 'key=("sent", phase, r)')
                  .replace('key=("read", phase, r, a)', 'key=("read", phase, r)'))
    tree = ast.parse(src)
    want = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_record"]
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_ChunkedExchange"]
    assert want and cls, "could not find _record / _ChunkedExchange"
    mod = ast.Module(body=want + cls, type_ignores=[])
    import types
    ns = {"torch": fake_torch(), "os": __import__("os"), "DEVICES": DEVICES,
          "_EVENTS": {}, "logging": types.SimpleNamespace(getLogger=lambda *a: types.SimpleNamespace(info=lambda *a, **k: None)),
          "defaultdict": __import__("collections").defaultdict, "_tl": None}
    exec(compile(ast.fix_missing_locations(mod), SRC, "exec"), ns)
    return ns


def drive(ns, blocks=1):
    """One phase of each block: send every chunk, then receive every chunk (the block() order)."""
    xchg = ns["_ChunkedExchange"]()
    parts = [FakeTensor((SIZE, 8), "float16", DEVICES[r]) for r in range(2)]
    per_block = []
    for block in range(blocks):
        LOG.append(("block", block))
        CONTEXT[0] = f"block{block}/begin"
        xchg.begin(PHASE, SHAPE)
        sent = []
        for j in range(CHUNKS):
            a, b = j * SIZE, (j + 1) * SIZE
            sent.append(xchg.send(PHASE, SHAPE, (a, b), parts))
        recv_binds = []
        for j in range(CHUNKS):
            a, b = j * SIZE, (j + 1) * SIZE
            before = len(LOG)
            xchg.recv(PHASE, SHAPE, (a, b), sent[j])
            recv_binds.append([entry for entry in LOG[before:] if entry[0] == "wait"])
        per_block.append(recv_binds)
    return xchg, per_block


def summarise(recv_binds):
    """Which send-record seq did each receive's copy-stream wait bind to?"""
    out = []
    for waits in recv_binds:
        copy_waits = [w for w in waits if w[1].startswith("copy:")]
        out.append([w[3] for w in copy_waits])
    return out


def main():
    checks = []

    # --- fixed code: per-chunk keys -----------------------------------------------------
    ns = load(shared_keys=False)
    _tag_records(ns)
    _, per_block = drive(ns, blocks=2)
    flat = [b[0] if b else None for b in summarise(per_block[0])]
    checks.append(("fixed: recv k waits on send k (got %s)" % flat,
                   flat == [f"sent1#{j + 1}" for j in range(CHUNKS)]))

    # --- the reviewed code: shared keys -------------------------------------------------
    ns_old = load(shared_keys=True)
    _tag_records(ns_old)
    _, per_block_old = drive(ns_old, blocks=2)
    flat_old = [b[0] if b else None for b in summarise(per_block_old[0])]
    checks.append(("old: every recv bound to the last send (got %s)" % flat_old,
                   all(f == f"sent1#{CHUNKS}" for f in flat_old)))

    # --- pooling: no new event objects after the first block ----------------------------
    created_after_first = len(EVENTS_CREATED)
    drive(ns, blocks=1)
    checks.append(("pooling: %d events created for two blocks, %d for a third"
                   % (created_after_first, len(EVENTS_CREATED) - created_after_first),
                   len(EVENTS_CREATED) - created_after_first == 0))

    # --- the cross-block guard still waits on the last chunk's read of the previous block --
    guard_hits = [entry for entry in LOG if entry[0] == "wait" and entry[1].startswith("copy:")
                  and str(entry[3]).startswith("read") and entry[3].endswith(f"#{CHUNKS}")]
    checks.append(("cross-block guard: host buffer waits on a previous block's last-chunk read (%d)"
                   % len(guard_hits), len(guard_hits) >= 1))

    for name, ok in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    return 0 if all(ok for _, ok in checks) else 1


def _tag_records(ns):
    """Tag each record with the ordinal of its logical call site (sent0#1, read1#3, ...).

    A stream wait binds to an event's most recent record, so the tag on the *bound* record is what
    shows which send a receive actually waited for.
    """
    real = ns["_record"]
    ordinal = {}

    def tagged(dev, stream=None, key=None):
        if key is None:
            CONTEXT[0] = "fresh"
        else:
            kind, phase, r = key[0], key[1], key[2]
            n = ordinal.get((kind, r), 0) + 1
            ordinal[(kind, r)] = n
            CONTEXT[0] = f"{kind}{r}#{n}"
        return real(dev, stream=stream, key=key)

    # _ChunkedExchange resolves _record from its module globals at call time, so rebinding it in
    # the namespace the class was compiled into is enough
    ns["_record"] = tagged


if __name__ == "__main__":
    sys.exit(main())
