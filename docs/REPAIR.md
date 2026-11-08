# When the first port is wrong or slow

Each question to the Kaggle Claude is a new conversation. It remembers nothing, so a follow-up must carry the whole context. `portkit/make_repair.py` does that. This page says when to use it and how to read the numbers.

## Procedure

1. Run `verify`. Read `VERIFY_REPORT.md`: it ends with PASS or FAIL per criterion (docs/PORTING.md section 7).
2. Pick ONE problem class: `wrong` (output differs), `slow` (correct but step time too high), `oom`, `crash`. If several fail, fix `wrong` first, then `crash`/`oom`, then `slow`.
3. Write the symptom as a fact with a number: "step 41 s, exchange-off step 38 s, GPU0 peak 14.2 GiB at 24.8k tokens", not "it is slow".
4. `python portkit/tp_port.py repair --name <n> --reference <family> --code <current module> --verify-report <file> --problem <class> --symptom "<fact>" --attempt <k>`.
5. Run the returned task file on Kaggle (kaggle_benchmarks), copy the CODE block into the module, run `verify` again. Keep every attempt: `portkit_runs/<name>/attempt_<k>/` holds the prompt, the answer, the verify report.
6. After 3 attempts without progress on the same criterion, stop and change the question: ask for a different approach to that one function (see "Reframing"), or accept the single-GPU fallback for that model.

## Decision tree for `slow`

Measure these on the real run (the verify kernel prints them):

- Exchange-off step time (the patch that skips the host copies; output is garbage, timing is valid). If the difference is under 5% of a step, the exchange is not the problem: do not ask for exchange changes.
- Per-kernel share of a block (qkv, attention, out, fc1, fc2, norms): compare each with the probe's microbenchmark prediction (`predicted_tp_ms_per_block`). A kernel more than 1.5x its prediction is the target.
- GPU clock and power during the step (`dmon`): on these T4s clocks sit near 700-800 MHz under the 70 W cap, so absolute TFLOPS are low for every design; compare against the probe's measured int8 TOPS, not the datasheet.
- Memory: allocator stalls show as long first steps and high `reserved` vs `allocated`.

Reference numbers (H3, 18.9k tokens, int8 convrot, both T4s): step 23 s warm; exchange on vs off 23.4 vs 22.7 s; int8 GEMM 30-40 TOPS; attention (memory efficient) 12.3 TFLOPS; first step of a cold run 115-137 s because it builds the shards.

## Reframing (when the same function fails twice)

- Give the model the failing function and its inputs and ask for only that function, with the invariant it must keep.
- Ask for a numerical test first ("write a CPU test that checks the split against the unsplit layer with random weights"), then the implementation.
- Show a working example of the same operation from the reference module and ask for the minimal difference.
- Ask for two alternative implementations with their expected trade-offs, then pick one by measurement.

## Making the follow-up read as a continuation

The generated prompt always has: the task as originally asked, the module the model wrote (verbatim), the measurements, the problem class with a short diagnostic checklist, and the instruction to change the minimum. Do not paste chat fragments or earlier answers: the code and the numbers are the memory.

## Budget

The $10 Kaggle credit buys a handful of 100k-token prompts. The prompt builder reports the token estimate; a first port is typically 45-70k tokens, a repair 40-60k. Trim by passing `--budget 80000`; the builder trims the model source, then the reference, then the guide, in that order.
