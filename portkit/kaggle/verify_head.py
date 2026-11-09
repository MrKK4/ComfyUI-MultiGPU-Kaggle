# Kit verify, part 1 (runs before ComfyUI starts). Expects KIT_GRAPH (API-format prompt of the target model) from the builder.
import glob, sysconfig

EXTRA_ENV['MMH3_TP_CHECK'] = '1'   # H3 module: compare the first tensor-parallel block with the original block (contract: every module logs a [TP-CHECK] line)
EXTRA_ENV['LD_PRELOAD'] = ':'.join(sorted(glob.glob(sysconfig.get_paths()['purelib'] + '/nvidia/cudnn/lib/libcudnn*.so.9*')))
EXTRA_NODES['zz_tp_kit'] = '''import gc, json, sys, torch

RES = "/kaggle/working/h3_up/tpprobe.json"


def _tp_modules():
    return [m for k, m in list(sys.modules.items()) if k.endswith("_tensor_parallel") and hasattr(m, "_INSTANCES") and bool(m._INSTANCES)]


def _mem():
    return {"cuda:%d" % i: {"allocated_gib": round(torch.cuda.memory_allocated(i) / 2**30, 2), "reserved_gib": round(torch.cuda.memory_reserved(i) / 2**30, 2),
                            "free_gib": round(torch.cuda.mem_get_info(i)[0] / 2**30, 2)} for i in range(torch.cuda.device_count())}


class TPSwitch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"exchange_off": ("BOOLEAN", {"default": False})}}
    RETURN_TYPES = ()
    FUNCTION = "go"
    OUTPUT_NODE = True
    CATEGORY = "test"

    def go(self, exchange_off):
        names = []
        for m in _tp_modules():
            if hasattr(m, "XCHG_OFF"):
                m.XCHG_OFF = bool(exchange_off); names.append(m.__name__)
        print("[TPSwitch] exchange_off =", exchange_off, "modules", names, flush=True)
        return ()


class TPProbe2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"release": ("BOOLEAN", {"default": True})}}
    RETURN_TYPES = ()
    FUNCTION = "go"
    OUTPUT_NODE = True
    CATEGORY = "test"

    def go(self, release):
        out = {"before": _mem()}
        if release:
            for m in _tp_modules():
                if hasattr(m, "release_gpu"):
                    m.release_gpu(drop_shards=True)
            gc.collect(); torch.cuda.empty_cache()
            out["after"] = _mem()
        json.dump(out, open(RES, "w"))
        print("[TPProbe2]", json.dumps(out), flush=True)
        return ()


class FiniteCheck:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"latent": ("LATENT",)}}
    RETURN_TYPES = ()
    FUNCTION = "go"
    OUTPUT_NODE = True
    CATEGORY = "test"

    def go(self, latent):
        x = latent["samples"]
        parts = x.unbind() if getattr(x, "is_nested", False) else [x]
        bad = sum(int((~torch.isfinite(p)).sum()) for p in parts)
        print("[FiniteCheck] non-finite values: %d in %d part(s)" % (bad, len(parts)), flush=True)
        return ()


NODE_CLASS_MAPPINGS = {"TPSwitch": TPSwitch, "TPProbe2": TPProbe2, "FiniteCheck": FiniteCheck}
'''
NEEDED = sorted({n['class_type'] for n in KIT_GRAPH.values()}) + ['TPSwitch', 'TPProbe2', 'FiniteCheck']
