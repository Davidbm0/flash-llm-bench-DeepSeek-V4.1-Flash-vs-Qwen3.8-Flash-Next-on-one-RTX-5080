# accuracy_5v6.py — accuracy comparison driver (5 experts vs 6 experts)
#
# Original ran two server configs (cfg_M = 5 experts/word, cfg_J6 = forced 6) against
# the 10-problem suite, single step (pass@1), think mode. Engine-specific (Strata
# configs); the transferable part is the eval call:
#
#   python coding_eval.py --port 8081 --model deepseek-v4.1-flash \
#       --setup DSV41-M5 --mode think --max-repairs 0 --max-tokens 16384 --http-timeout 3600
#
# and the comparison rule: same suite, same hyperparameters, same machine state
# (no other model server running — a second server steals ~45 GB RAM and halves
# the offloaded model's speed).
