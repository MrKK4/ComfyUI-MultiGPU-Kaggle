"""Feasibility probe for a dual-GPU tensor-parallel port.

Layer 1 (stdlib only, works on a local file or a Hugging Face URL, no GPU): reads the safetensors header and checks the
feasibility rules of docs/PORTING.md section 4 that can be decided from tensor names, shapes and dtypes.
Layer 2 (--deep --comfy <ComfyUI dir>, needs torch + ComfyUI, meant for Kaggle): builds the model on the meta device,
reports the module tree, attention attributes, dumps the source of the block and forward functions, and benchmarks the
block linears and attention on the GPU to predict the compute per block.

usage: python probe_model.py --model <path or URL> --out <dir> [--deep --comfy <ComfyUI dir>] [--tokens 18900]
Writes <dir>/probe.json, <dir>/PROBE_REPORT.md and, with --deep, <dir>/block_source.txt.
"""
import argparse
import collections
import json
import os
import re
import struct
import sys
import urllib.request

BLOCK_LISTS = ("blocks", "transformer_blocks", "layers", "double_blocks", "single_blocks", "joint_blocks", "joint_transformer_blocks", "h")
DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


class Source:
    def __init__(self, where):
        self.where = where
        self.url = where.startswith(("http://", "https://"))

    def read(self, a, b):
        """bytes [a, b)"""
        if self.url:
            req = urllib.request.Request(self.where, headers={"Range": "bytes=%d-%d" % (a, b - 1), "User-Agent": "tp-port-probe"})
            return urllib.request.urlopen(req, timeout=120).read()
        with open(self.where, "rb") as f:
            f.seek(a)
            return f.read(b - a)

    def size(self):
        if self.url:
            req = urllib.request.Request(self.where, headers={"Range": "bytes=0-0", "User-Agent": "tp-port-probe"})
            r = urllib.request.urlopen(req, timeout=120)
            return int(r.headers.get("Content-Range", "0/0").split("/")[-1])
        return os.path.getsize(self.where)


def read_header(src):
    n = struct.unpack("<Q", src.read(0, 8))[0]
    return json.loads(src.read(8, 8 + n)), 8 + n


def nbytes(t):
    p = 1
    for d in t["shape"]:
        p *= d
    return p * DTYPE_BYTES.get(t["dtype"], 2)


def strip_prefix(keys):
    for p in ("model.diffusion_model.", "diffusion_model.", "model."):
        if all(k.startswith(p) for k in keys if not k.startswith("__")):
            return p
    return ""


def layer1(src):
    header, base = read_header(src)
    meta = header.pop("__metadata__", None) or {}
    prefix = strip_prefix(list(header))
    tensors = {k[len(prefix):]: v for k, v in header.items()}
    rx = re.compile(r"^(?P<list>[A-Za-z_]+)\.(?P<i>\d+)\.(?P<rest>.+)$")
    lists = collections.defaultdict(lambda: collections.defaultdict(dict))
    for k, t in tensors.items():
        m = rx.match(k)
        if m and m["list"] in BLOCK_LISTS:
            lists[m["list"]][int(m["i"])][m["rest"]] = t
    out = {"file": src.where.split("/")[-1], "bytes": src.size(), "key_prefix": prefix, "metadata_keys": sorted(meta)[:12],
           "tensor_count": len(tensors), "dtype_counts": dict(collections.Counter(t["dtype"] for t in tensors.values()))}
    if not lists:
        out["blocks"] = None
        return out, tensors, base
    name = max(lists, key=lambda l: len(lists[l]))
    blocks = lists[name]
    sig = {i: tuple(sorted((r, t["dtype"], tuple(t["shape"])) for r, t in b.items())) for i, b in blocks.items()}
    groups = collections.defaultdict(list)
    for i, s in sig.items():
        groups[s].append(i)
    first = sorted(max(groups.values(), key=len))[0]   # a block of the majority signature
    out["blocks"] = {"list_name": name, "count": len(blocks), "indices": [min(blocks), max(blocks)], "distinct_signatures": len(groups),
                     "signature_groups": [{"indices": sorted(v)[:12] + (["..."] if len(v) > 12 else []), "count": len(v)} for v in sorted(groups.values(), key=len, reverse=True)],
                     "block_bytes": {str(i): sum(nbytes(t) for t in blocks[i].values()) for i in (first, max(blocks))},
                     "total_block_bytes": sum(nbytes(t) for b in blocks.values() for t in b.values())}
    out["block0_tensors"] = [{"name": r, "dtype": t["dtype"], "shape": t["shape"], "MB": round(nbytes(t) / 2**20, 2)} for r, t in sorted(blocks[first].items())]
    quant = {}
    for r, t in blocks[first].items():
        if r.endswith(".comfy_quant"):
            a, b = t["data_offsets"]
            try:
                quant[r[:-len(".comfy_quant")]] = json.loads(src.read(base + a, base + b).decode())
            except Exception as e:
                quant[r[:-len(".comfy_quant")]] = "unreadable: %s" % e
    out["block0_quant_config"] = quant
    other = collections.defaultdict(int)
    for k, t in tensors.items():
        m = rx.match(k)
        if not (m and m["list"] == name):
            other[k.split(".")[0]] += nbytes(t)
    out["outside_blocks_MB"] = {k: round(v / 2**20, 1) for k, v in sorted(other.items(), key=lambda kv: -kv[1])[:12]}
    return out, tensors, base


def rules(info):
    r = {}
    b = info.get("blocks")
    if not b:
        return {"R1 uniform blocks": "FAIL: no repeated block list found in the key names"}
    r["R1 uniform blocks"] = "PASS: %d blocks, one signature" % b["count"] if b["distinct_signatures"] == 1 else \
        "FAIL: %d distinct block signatures (dtype or shape differs between blocks): %s" % (b["distinct_signatures"], json.dumps(b["signature_groups"]))
    lin = [t for t in info["block0_tensors"] if len(t["shape"]) == 2 and t["name"].endswith(".weight") and t["shape"][0] > 64 and t["shape"][1] > 64]
    bad = []
    for t in lin:   # a layer with more inputs than outputs is treated as row-parallel (input split), otherwise column-parallel (output split)
        n, k = t["shape"]
        if (k > n and (k % 2 or (k // 2) % 256)) or (k <= n and (n % 2 or k % 256)):
            bad.append((t["name"], t["shape"], "row-parallel" if k > n else "column-parallel"))
    r["R2 shapes (row-parallel: half the input % 256; column-parallel: output even, input % 256)"] = \
        "PASS" if not bad else "CHECK: %s (matters only for int8 convrot layers; heads/ffn parity needs --deep)" % bad
    q = info["block0_quant_config"]
    dts = {t["dtype"] for t in lin}
    if q and dts <= {"I8"} and all(isinstance(v, dict) and v.get("format") == "int8_tensorwise" and v.get("convrot") for v in q.values()):
        r["R6 quantization"] = "PASS: native int8 convrot (group %s)" % {v.get("convrot_groupsize") for v in q.values()}
    elif dts <= {"BF16", "F16", "F32"}:
        r["R6 quantization"] = "REQUANT: dense %s weights; the shard builder requantizes to int8 convrot (about 1-3 s per billion parameters)" % sorted(dts)
    else:
        r["R6 quantization"] = "CHECK: %s with quant config %s" % (sorted(dts), json.dumps(q)[:300])
    shard_gib = b["total_block_bytes"] / 2 / 2**30
    if all(t["dtype"] == "I8" for t in lin):
        r["R7 memory"] = "PASS: ~%.2f GiB of shards per GPU" % shard_gib if shard_gib <= 10.5 else "FAIL: ~%.2f GiB of shards per GPU (limit ~10.5)" % shard_gib
    else:
        int8_gib = sum(nbytes(t) // (2 if t["dtype"] in ("BF16", "F16") else 1) for t in lin) * b["count"] / 2 / 2**30
        r["R7 memory"] = "CHECK: file shards %.2f GiB per GPU as stored; after int8 requant about %.2f GiB per GPU" % (shard_gib, int8_gib)
    for k in ("R3 attention through one function with transformer_options", "R4 block replaceable", "R5 no token mixing besides attention", "R8 extra streams"):
        r[k] = "UNKNOWN: needs --deep (source text) or a reader of block_source.txt"
    return r


def layer2(comfy, model_path, info, tensors, tokens, out_dir):
    import importlib
    import inspect
    sys.path.insert(0, comfy)
    os.chdir(comfy)
    import torch
    md = importlib.import_module("comfy.model_detection")
    dt = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "I8": torch.int8, "U8": torch.uint8, "I32": torch.int32, "I64": torch.int64,
          "F8_E4M3": getattr(torch, "float8_e4m3fn", torch.float16)}
    prefix = info["key_prefix"]
    sd = {prefix + k: torch.empty(t["shape"], dtype=dt.get(t["dtype"], torch.float16), device="meta") for k, t in tensors.items()}
    deep = {}
    mc = md.model_config_from_unet(sd, prefix)
    if mc is None:
        deep["error"] = "ComfyUI does not recognise this checkpoint (model_config_from_unet returned None)"
        return deep, ""
    deep["model_config"] = type(mc).__name__
    deep["unet_config"] = {k: (v if isinstance(v, (int, float, str, bool, list, tuple, type(None))) else str(v)) for k, v in mc.unet_config.items()}
    with torch.device("meta"):
        model = mc.get_model(sd, prefix, device=None)
    dm = model.diffusion_model
    deep["diffusion_model_class"] = "%s.%s" % (type(dm).__module__, type(dm).__name__)
    lst = info["blocks"]["list_name"] if info.get("blocks") else None
    blocks = getattr(dm, lst, None) if lst else None
    srcs = []
    if blocks is not None and len(blocks):
        blk = blocks[0]
        deep["block_class"] = "%s.%s" % (type(blk).__module__, type(blk).__name__)
        deep["block_children"] = [(n, type(m).__name__, tuple(m.weight.shape) if hasattr(m, "weight") and m.weight is not None and m.weight.ndim else None) for n, m in blk.named_modules() if n][:80]
        attrs = {}
        for n, m in blk.named_modules():
            for a in ("heads", "num_heads", "n_heads", "head_dim", "dim_head", "hidden_size", "dim", "inner_dim", "scale"):
                if hasattr(m, a) and isinstance(getattr(m, a), (int, float)):
                    attrs["%s.%s" % (n or "block", a)] = getattr(m, a)
        deep["block_attributes"] = attrs
        for label, obj in (("block class", type(blk)), ("diffusion model forward", type(dm).forward)):
            try:
                srcs.append((label, inspect.getsource(obj)))
            except Exception as e:
                srcs.append((label, "unavailable: %s" % e))
        for n, m in blk.named_children():
            if "attn" in n.lower() or "attention" in n.lower():
                try:
                    srcs.append(("%s class %s" % (n, type(m).__name__), inspect.getsource(type(m))))
                except Exception:
                    pass
        if hasattr(dm, "_forward"):
            try:
                srcs.append(("diffusion model _forward", inspect.getsource(type(dm)._forward)))
            except Exception:
                pass
    deep["top_level_modules"] = [(n, type(m).__name__) for n, m in dm.named_children()]
    text = "\n\n".join("# ===== %s =====\n%s" % (a, b[:18000]) for a, b in srcs)
    allsrc = "\n".join(b for _, b in srcs)
    deep["static_flags"] = {"uses optimized_attention": "optimized_attention" in allsrc, "mentions patches_replace": "patches_replace" in allsrc,
                            "mentions blocks_replace": "blocks_replace" in allsrc, "mentions double_block": "double_block" in allsrc,
                            "passes transformer_options": "transformer_options" in allsrc, "uses gradient checkpointing": "checkpoint" in allsrc}
    # GPU microbenchmark of the block linears and attention (only when a GPU exists)
    if torch.cuda.is_available() and info.get("blocks"):
        import time
        dev = torch.device("cuda", 0)
        M = tokens
        rows = []
        for t in info["block0_tensors"]:
            if len(t["shape"]) == 2 and t["name"].endswith(".weight") and t["shape"][0] > 256 and t["shape"][1] > 256:
                n, k = t["shape"]
                a = torch.randint(-127, 127, (M, k), device=dev, dtype=torch.int8)
                w = torch.randint(-127, 127, (n, k), device=dev, dtype=torch.int8)
                torch._int_mm(a, w.t()); torch.cuda.synchronize(); t0 = time.time()
                for _ in range(4):
                    torch._int_mm(a, w.t())
                torch.cuda.synchronize(); dtm = (time.time() - t0) / 4
                rows.append({"linear": t["name"], "shape": [n, k], "int8_ms_full": round(dtm * 1e3, 1), "TOPS": round(2 * M * n * k / dtm / 1e12, 1)})
                del a, w
        deep["linear_bench_tokens_%d" % M] = rows
        deep["linear_ms_per_block_full"] = round(sum(r["int8_ms_full"] for r in rows), 1)
        heads = next((v for k, v in attrs.items() if k.endswith(("heads", "num_heads", "n_heads"))), None)
        hd = next((v for k, v in attrs.items() if k.endswith(("head_dim", "dim_head"))), None)
        if heads and hd:
            import torch.nn.functional as F
            from torch.nn.attention import SDPBackend, sdpa_kernel
            h = int(heads) // 2
            q, kk, v = (torch.randn(1, h, M, int(hd), device=dev, dtype=torch.half) for _ in range(3))
            try:
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    F.scaled_dot_product_attention(q, kk, v); torch.cuda.synchronize(); t0 = time.time()
                    for _ in range(2):
                        F.scaled_dot_product_attention(q, kk, v)
                    torch.cuda.synchronize()
                ms = (time.time() - t0) / 2 * 1e3
                deep["attention_ms_per_block_per_rank"] = round(ms, 1)
                tp_ms = deep["linear_ms_per_block_full"] / 2 + ms
                deep["predicted_tp_ms_per_block"] = round(tp_ms, 1)
                deep["predicted_tp_step_seconds"] = round(tp_ms * info["blocks"]["count"] / 1000, 1)
            except Exception as e:
                deep["attention_bench"] = "failed: %s" % e
    return deep, text


def report(info, rl, deep, tokens):
    L = ["# Probe report: %s" % info["file"], "", "## Rules (docs/PORTING.md section 4)"]
    L += ["- **%s**: %s" % (k, v) for k, v in rl.items()]
    L += ["", "## File", "- bytes %d, tensors %d, key prefix '%s', dtype counts %s" % (info["bytes"], info["tensor_count"], info["key_prefix"], info["dtype_counts"])]
    b = info.get("blocks")
    if b:
        L += ["- block list `%s`: %d blocks, %d distinct signature(s), block bytes %s, total %.2f GiB" % (b["list_name"], b["count"], b["distinct_signatures"], b["block_bytes"], b["total_block_bytes"] / 2**30),
              "- outside the blocks (MB): %s" % info["outside_blocks_MB"], "", "### Block 0 tensors"]
        L += ["- `%s` %s %s (%s MB)" % (t["name"], t["dtype"], t["shape"], t["MB"]) for t in info["block0_tensors"]]
        L += ["", "### Block 0 quantization config", "```json", json.dumps(info["block0_quant_config"], indent=1)[:2500], "```"]
    if deep:
        L += ["", "## Deep probe (ComfyUI, meta device)", "```json", json.dumps(deep, indent=1, default=str)[:12000], "```"]
    L += ["", "Tokens used for the benchmark: %d. Everything marked UNKNOWN must be decided by reading block_source.txt." % tokens]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--comfy")
    ap.add_argument("--tokens", type=int, default=18900)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    src = Source(a.model)
    info, tensors, base = layer1(src)
    rl = rules(info)
    deep, text = {}, ""
    if a.deep:
        try:
            deep, text = layer2(a.comfy, a.model, info, tensors, a.tokens, a.out)
        except Exception as e:
            import traceback
            deep = {"error": "%s: %s" % (type(e).__name__, e), "trace": traceback.format_exc()[-1800:]}
    res = {"info": info, "rules": rl, "deep": deep}
    with open(os.path.join(a.out, "probe.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    with open(os.path.join(a.out, "PROBE_REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report(info, rl, deep, a.tokens))
    if text:
        with open(os.path.join(a.out, "block_source.txt"), "w", encoding="utf-8") as f:
            f.write(text)
    print(open(os.path.join(a.out, "PROBE_REPORT.md"), encoding="utf-8").read()[:6000])


if __name__ == "__main__":
    main()
