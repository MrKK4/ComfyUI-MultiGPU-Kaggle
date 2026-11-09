# VERIFY_REPORT: FAIL

Direct denoiser outputs, synthetic inputs; image encoders, decoder, sampler hooks and production workflows are not exercised.

- **supports target**: PASS: LatentSeqMMFlowModel
- **rejects other models**: PASS: None
- **automatic attach**: PASS: None
- **full forward seed 1**: FAIL: {'finite': True, 'cosine': -0.4477924704551697, 'rel_err': 2.001352548599243}
- **full forward seed 2**: FAIL: {'finite': True, 'cosine': -0.4474090337753296, 'rel_err': 1.9870169162750244}
- **full forward seed 3**: FAIL: {'finite': True, 'cosine': -0.4454461932182312, 'rel_err': 1.9864752292633057}
- **memory margin**: PASS: [1.1824707984924316, 0.31996774673461914]
- **deliberately broken port is rejected**: PASS: {'cosine': -0.4477924704551697, 'verifier_verdict': 'FAIL', 'seconds': 0.8162092960000109}
- **release under 3 GiB**: PASS: [0.7332925796508789, 0.00390625]
- **reload reproduces output**: FAIL: 0.8527003379999769

```json
{
  "single_gpu_seconds": [
    1.2384972810000363,
    1.0054016330000195,
    1.0101949439999771
  ],
  "tp_seconds": [
    1.017575813999997,
    0.8163449730000139,
    0.8158660310000414
  ],
  "peak_allocated_gib": [
    1.1824707984924316,
    0.31996774673461914
  ],
  "deliberate_exchange_off": {
    "cosine": -0.4477924704551697,
    "verifier_verdict": "FAIL",
    "seconds": 0.8162092960000109
  },
  "after_release_allocated_gib": [
    0.7332925796508789,
    0.00390625
  ],
  "speed_note": "Compare warm timings only. A slower port requires an explicit fallback decision; no speed acceptance is inferred."
}
```
