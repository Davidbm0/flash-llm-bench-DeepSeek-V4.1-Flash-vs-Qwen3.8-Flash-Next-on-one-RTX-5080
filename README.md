# flash-llm-bench — DeepSeek V4.1-Flash vs Qwen3.8-Flash-Next on one RTX 5080

Benchmark suite + results from fully-local inference experiments (November 2026):
speed tuning of a huge MoE that does not fit in 16 GB VRAM, a head-to-head accuracy
run between the two models, and a partial Humanity's Last Exam evaluation.

**Hardware:** AMD Ryzen 7 7800X3D · RTX 5080 16 GB (+ RTX 3080 10 GB, used only for
split-GPU tests) · 96 GB DDR5 · NVMe Gen5
**Serving:** Strata local inference server (OpenAI-compatible API, ports 8080/8081)

All eval scripts talk to any **OpenAI-compatible** `/v1/chat/completions` endpoint
(llama.cpp `llama-server`, vLLM, Strata, ...), so you can run the whole suite against
your own model in a few minutes. See [Run it yourself](#run-it-yourself).

## Models tested

| Model | Quantization | VRAM footprint | Native decode speed |
|---|---|---|---|
| DeepSeek V4.1-Flash (MoE, 15360 routed experts) | 2.25 bpw pack | ~11 GB in VRAM, rest offloaded to RAM/NVMe | ~9–15 tok/s |
| Qwen3.8-Flash-Next | IQ3_XXS GGUF | ~14.5 GB (fits fully in VRAM) | ~55–96 tok/s |

DeepSeek V4.1-Flash is a huge MoE: only ~6679 of the routed experts fit in the
96 GB RAM store, the rest is streamed from NVMe on every token ("CPU lane").
Most of the benchmark work below is about tuning that offload.

### The RTX 5080 constraint

The whole benchmark is shaped by one number: **16 GB VRAM**.

- DeepSeek V4.1-Flash's routed experts total **~149 GB** — 9x more than the 5080
  holds. Only the dense layers + attention + ~250 of 15360 experts fit on the GPU
  (~11 GB); everything else is offloaded: 76 GB hot-expert store in DDR5 RAM, the
  remaining ~147 GB read in place from the Gen5 NVMe file.
- This is why decode is ~9–15 tok/s instead of the 55–96 tok/s Qwen gets: every
  token touches ~5 experts, and ~1/3 of them miss the RAM store and come off disk.
- The 16 GB cap also forces **5 experts/word** ("faster mode") instead of the full
  6 — the engine picks this automatically on ≤16 GiB VRAM. Arm J shows forcing 6
  costs −13% decode with no accuracy gain, so the cap costs nothing in practice.
- The RTX 3080 (10 GB) cannot meaningfully join the same model: expert layers split
  per-GPU, and prefill collapses ~5x (arms D/I/K). It only helps raw generation
  speed (K: 18 tok/s), not real workloads.
- Practical ceiling on this box: a MoE needs roughly (active-weights × ~1.3) in VRAM
  for comfortable speed. DeepSeek V4.1-Flash at 2.25 bpw is right at the edge;
  Qwen3.8-Flash-Next IQ3_XXS (~14.5 GB) fits fully and is ~6x faster per token.

---

## 1. Speed benchmark — 15 configs ("arms")

Harness: `scripts/bench_arms.py`. Each arm starts the server with a different
config, warms up, then measures:
- **decode** — 3×800 tokens from a 25-token prompt (cold → warm)
- **think** — 4000 reasoning tokens on a reasoning prompt
- **prefill** — 8340-token prompt read speed

| Arm | Config | decode warm (tok/s) | think (tok/s) | prefill (tok/s) |
|---|---|---|---|---|
| A | baseline | 11.55 | 8.20 | 641 |
| B | + `--cpu-threads 16` | 8.78 | 6.76 | 676 |
| C | B + budget 12000 | 9.46 | 7.38 | 684 |
| D | split 5080+3080 | 14.47 | 8.41 | **139** |
| E | ctx 16K + budget 12000 | 13.27 | 9.33 | 686 |
| F | `--ssd-policy lfu` | 13.97 | 8.58 | 711 |
| G | budget 12000 alone | 12.91 | 9.33 | 696 |
| H | LFU + budget 12000 | 13.80 | 8.31 | 690 |
| I | split + LFU + budget | 13.56 | 7.06 | 138 |
| R | control (identical to A, later round) | 12.41 | 9.00 | 673 |
| J | H + `--experts 6` | 12.02 | 7.97 | 661 |
| K | split + `--experts 5` | 18.05 | 9.58 | 148 |
| L | H + `--ssd-cache 72` | 14.49 | 8.90 | 694 |
| **M** | **H + `--ssd-cache 76`** | **15.13** | **9.39** | **728** |
| N | H + `--ssd-cache 77` | 14.50 | 9.17 | 705 |

Notes on methodology:
- Cross-round numbers are NOT comparable: the engine's expert-hotness profile
  warms up during the day (+9% drift measured with control arm R vs A).
  Compare arms only within a round.
- `prefill` for L/M/N in the raw engine output showed 0.08 tok/s — a parse bug in
  the bench (prompt reused from cache); real values from engine logs: 694–728 tok/s.
- Flash-Next running on the same box steals ~45 GB RAM → DeepSeek drops to
  4–5 tok/s. All numbers above were measured with it stopped.

**Conclusions:**
1. `--ssd-policy lfu` is the only real speed lever (+8–10% decode, +7% prefill):
   keeping *frequently* routed experts in RAM beats keeping *recently* routed ones.
2. `--cpu-threads 16` is harmful (−25–30%): the engine clamps to 8 physical cores
   and 8 compute + 8 read threads thrash the 7800X3D.
3. Split 5080+3080 gives the fastest raw generation (K: 18 tok/s) but prefill
   collapses to ~140 tok/s (5× slower) — useless for real prompts.
4. `reasoning_budget_tokens 12000` doesn't change tok/s, it halves worst-case
   thinking time on hard problems (default budget 22323 ≈ 37 min of thinking).
5. Bigger RAM store pays up to ~76 GB (`--ssd-cache 76`, arm M): +391 hot experts
   in RAM, CPU lane drops from ~40% to ~34% during thinking. Beyond that (N) it
   stops helping — Windows headroom starts to matter.
6. 5 experts/word ("faster mode", default on ≤16 GiB VRAM) is enough: forcing 6
   (arm J) costs −13% decode for no measurable gain on our accuracy suite.

**Final config = arm M** → `configs/cfg_M.json` (paths sanitized):
LFU + `--ssd-cache 76` + budget 12000, ctx 32K, single 5080, 5 experts/word.

---

## 2. Accuracy — 10 hard engineering problems

Suite (`scripts/coding_eval.py` + `scripts/new_problems.py`): Dijkstra with
parity state, microstrip impedance inverse solve, transaction-lock engine with
deadlock detection, PDN impedance anti-resonance scan, AST affine optimizer,
coupled microstrip MTL, 2-D FDM Laplace solver, SPICE-style diode RC transient,
SerDes FFE/DFE eye analytic, RF conjugate-match synthesis. Each problem is
graded by running the model's code against hand-verified asserts; the E1–E5
ground truth is re-derivable by running `scripts/ref_extreme5.py` (5/5 self-checked).

### DeepSeek V4.1-Flash, cfg_M, single shot (pass@1, no repairs)

**10/10 pass@1, 6681 s total (~111 min).** All 10 solved on the first attempt,
all `finish=stop` (no truncation). Per-problem: 192–1186 s.

### Qwen3.8-Flash-Next, time-boxed against DeepSeek

Rule (`scripts/flash_timebox.py`): Flash gets exactly DeepSeek M5's wall-clock
budget per problem, but may iterate — failing code + exact traceback is fed
back until the budget runs out.

| Problem | DeepSeek budget (s) | Flash solved in (s) | Attempts |
|---|---|---|---|
| p1 dijkstra parity | 192 | 135 | 1 |
| p2 microstrip | 635 | 270 | 2 |
| p3 tx engine | 1186 | 758 | 4 |
| p4 pdn | 402 | 244 | 1 |
| p5 ast | 550 | 493 | 3 |
| e1 coupled MTL | 283 | 210 | 1 |
| e2 FDM Laplace | 633 | 369 | 2 |
| e3 diode RC | 1172 | 483 | 3 |
| e4 SerDes eye | 971 | 636 | 4 |
| e5 RF match | 657 | 306 | 3 |
| **total** | **6682** | **3907 (58%)** | **24** |

**Flash: 10/10 solved, in 58% of DeepSeek's time, with 24 attempts.**
But pure single-shot pass@1: **Flash 3/10 vs DeepSeek 10/10.**

Interpretation: DeepSeek's ~12k-token thinking gets the code right the first
time. Flash thinks ~10–20k tokens but produces broken code (truncated at
16384, prose leaking into code); its win comes entirely from the fast repair
loop (55–96 tok/s makes iteration cheap). For interactive work Flash is more
productive; for one-shot hard problems DeepSeek is far stronger.

---

## 3. HLE (Humanity's Last Exam) — partial run, stopped early

Setup (`scripts/hle_eval.py`): `cais/hle` dataset, 100-question stratified
sample (seed 42), image-dependent questions excluded (2385 text-only usable),
300 s budget per question, exact/numeric-tolerant grading.

Qwen3.8-Flash-Next, first 27 questions: **1/27 correct.**

Failure mode: the model burns the whole budget on circular "thinking" (recall
loops, re-deriving facts) and often hits the token limit before emitting a
final answer. Stopped the run at 27/100 — the trend made the remaining 73
questions pointless. (The single "correct" was also a grading false positive:
gold `ABC` matched a prediction of `A, B, C, D, E` via substring — clean score 0/27.)

---

## Run it yourself

Everything below needs only **Python 3.10+** and any model server exposing
`/v1/chat/completions`. Example with llama.cpp:

```bash
llama-server -m YourModel-Q4_K_M.gguf --port 8080 --ctx-size 16384
```

### 2-minute check: the accuracy suite

```bash
cd scripts
python coding_eval.py --port 8080 --model your-model --setup MYRUN --mode think --max-repairs 0
```

That runs all 10 problems (P1–P5 + E1–E5), single shot = **pass@1**. Add
`--max-repairs 4` for **pass@5** with self-repair (the exact traceback is fed
back to the model). Other useful flags: `--suite p` / `--suite e` (only one
half), `--problems p2,e4` (subset), `--mode nothink`.

Verify the ground truth yourself first (recommended):

```bash
python ref_extreme5.py     # must print: VERDICT: 5/5 ground-truth VERIFICAT
```

### Time-boxed fast-vs-slow comparison

1. Run the slow model single-shot (its per-problem times become the budgets):
   ```bash
   python coding_eval.py --port 8081 --model slow-model --setup SLOW --mode think --max-repairs 0
   ```
2. Give the fast model exactly those budgets, with self-repair allowed:
   ```bash
   python flash_timebox.py --port 8080 --model fast-model --budgets results/SLOW_think.json
   ```
   Score = how many problems the fast model solves inside the slow model's time.

### HLE

```bash
pip install pandas pyarrow huggingface_hub
hf download cais/hle --repo-type dataset   # gated: accept the dataset terms on HF first
# move the downloaded parquet(s) into scripts/data/
HLE_PORT=8080 HLE_MODEL=your-model python hle_eval.py
```

Env knobs: `HLE_BUDGET` (s per question, default 300), `HLE_MAXTOK`,
`HLE_THINK_BUDGET` (cap thinking for slow models), `HLE_OUT`/`HLE_PROG`.
Full 2500-question run is ~60 h at Flash speed — the default sample is 100
questions, stratified, seed 42.

### Speed bench

`scripts/bench_arms.py` is a template: set `SERVER_CMD` to your server launch
command and adapt the arms to your engine's flags. The measurement logic
(warm-up → decode → prefill → think) is server-agnostic.

## Files

```
configs/cfg_M.json          final recommended DeepSeek config (sanitized)
scripts/coding_eval.py      10-problem accuracy harness (pass@1 / pass@5)
scripts/new_problems.py     problems E1-E5 (prompts + asserts)
scripts/ref_extreme5.py     hand-verified references for E1-E5 (ground truth)
scripts/flash_timebox.py    time-boxed fast-vs-slow eval (self-repair loop)
scripts/hle_eval.py         HLE sample eval harness
scripts/run_hle_ds.py       example: run the same HLE sample on a 2nd model
scripts/bench_arms.py       speed benchmark template (arms A-N)
scripts/accuracy_5v6.py     notes on the 5-vs-6-experts accuracy comparison
```

Raw per-attempt JSONs are not included — the summary tables are in this README.
