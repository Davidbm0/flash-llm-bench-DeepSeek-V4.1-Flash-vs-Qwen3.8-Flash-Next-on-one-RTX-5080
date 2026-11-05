# acc_eval.py — wrapper: coding_eval_extreme.py + suita noua din new_problems.py
# (nu modificam harness-ul original; doar suprascriem build_problems daca
#  SUITE_NEW e definit). Argumentele se pas mai departe identic.
import sys, os

QWEN = r"C:\Users\black\Desktop\Cronos Work\qwen35-comparatie"
TUNE = r"C:\Users\black\Desktop\Cronos Work\dsv41-tuning"
sys.path.insert(0, QWEN)
sys.path.insert(0, TUNE)

import coding_eval_extreme as ce
_orig_build = ce.build_problems
try:
    import new_problems
    if new_problems.SUITE_NEW:
        # 10 probleme: P1-P5 (original) + E1-E5 (suite nou, ground-truth verificat)
        ce.build_problems = lambda v: _orig_build(v) + new_problems.build_new_problems()
        print(f"[wrapper] suita = P1-P5 + {len(new_problems.SUITE_NEW)} probleme noi "
              f"= {len(ce.build_problems('ref14'))} total", flush=True)
except ImportError:
    pass

ce.main()
