"""bench_arms.py — speed benchmark for a MoE offloaded to RAM/NVMe (arms A-N).

Generic harness: it starts YOUR server (any command you configure), waits for an
OpenAI-compatible endpoint, then measures decode / think / prefill per config arm.
Adapt SERVER_CMD, ARMS and the readiness check to your server (llama.cpp, Strata,
vLLM, ...). The measurement logic itself is server-agnostic.

What each arm measures (after a warm-up that is NOT counted):
- decode  : 3x800 tokens from a short prompt, no thinking (cold -> warm)
- prefill : one 8340-token prompt, 1 output token (prompt-read speed)
- think   : 4000 reasoning tokens on a hard reasoning prompt

Arms A-N from our run (DeepSeek V4.1-Flash on Strata, single RTX 5080):
  A baseline | B --cpu-threads 16 | C B+budget | D split 2 GPUs | E ctx 16K+budget
  F --ssd-policy lfu | G budget 12000 | H F+G | I split+F+G | R control (=A, later)
  J H+--experts 6 | K split+--experts 5 | L/M/N H+--ssd-cache 78/81/82 (adaptive)
See README for the results table and conclusions.

Usage: python bench_arms.py [ARM ...]   (no args = all arms, skips already-done ones)
"""
