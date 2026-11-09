# VERIFY_REPORT: PASS

Direct denoiser outputs, synthetic inputs; image encoders, decoder, sampler hooks and production workflows are not exercised.

- **supports target**: PASS: LatentSeqMMFlowModel
- **rejects other models**: PASS: None
- **automatic attach**: PASS: None
- **full forward seed 1**: PASS: {'finite': True, 'cosine': 0.9998368620872498, 'rel_err': 0.017757881432771683}
- **full forward seed 2**: PASS: {'finite': True, 'cosine': 0.999839723110199, 'rel_err': 0.017637694254517555}
- **full forward seed 3**: PASS: {'finite': True, 'cosine': 0.9998395442962646, 'rel_err': 0.017622485756874084}
- **memory margin**: PASS: [1.1981887817382812, 0.33568572998046875]
- **deliberately broken port is rejected**: PASS: {'cosine': -0.4477924704551697, 'verifier_verdict': 'FAIL', 'seconds': 0.8531400740000095}
- **release under 3 GiB**: PASS: [0.7332925796508789, 0.00390625]
- **reload reproduces output**: PASS: 1.0673670870000365

```json
{
  "single_gpu_seconds": [
    1.329297605000022,
    0.9984143760000279,
    1.0009700310000085
  ],
  "tp_seconds": [
    1.3657839350000245,
    1.0276531439999985,
    1.0266956279999704
  ],
  "peak_allocated_gib": [
    1.1981887817382812,
    0.33568572998046875
  ],
  "deliberate_exchange_off": {
    "cosine": -0.4477924704551697,
    "verifier_verdict": "FAIL",
    "seconds": 0.8531400740000095
  },
  "after_release_allocated_gib": [
    0.7332925796508789,
    0.00390625
  ],
  "speed_note": "Compare warm timings only. A slower port requires an explicit fallback decision; no speed acceptance is inferred."
}
```
