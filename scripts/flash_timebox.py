# flash_timebox.py — time-boxed comparison: a FAST model vs a SLOW model's wall-clock
#
# Rule: the fast model gets exactly the slow model's per-problem wall-clock budget
# (read from the slow model's pass@1 result JSON), but may iterate — failing code +
# the exact traceback is fed back until the budget runs out.
# Metric: X/10 solved inside the slow model's time. Progress saved after each problem.
#
# Usage:
#   python coding_eval.py --port 8081 --model slow-model --setup SLOW --mode think --max-repairs 0
#   python flash_timebox.py --port 8080 --model fast-model --budgets results/SLOW_think.json
import json, os, sys, time, argparse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import coding_eval as ce
import new_problems

def log(msg, prog=None):
    print(msg, flush=True)
    if prog:
        with open(prog, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--budgets", required=True,
                    help="JSON from coding_eval.py (slow model, --max-repairs 0): per-problem t_s")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--out", default=os.path.join(HERE, "timebox.json"))
    ap.add_argument("--progress", default=os.path.join(HERE, "timebox_progress.log"))
    a = ap.parse_args()

    base = f"http://127.0.0.1:{a.port}/v1"
    try:
        urllib.request.urlopen(base + "/models", timeout=5).read()
    except Exception:
        log("model server not reachable on port " + str(a.port) + " — start it first.", a.progress)
        return

    budgets = {k: v["attempts"][0]["t_s"]
               for k, v in json.load(open(a.budgets, encoding="utf-8"))["detail"].items()}
    suite = ce.build_problems("ref14") + new_problems.build_new_problems()

    out = {"setup": "TIMEBOX", "model": a.model, "mode": "think",
           "rule": "per-problem budget = slow model's pass@1 single-step t_s; "
                   "iterative self-repair inside the budget (exact traceback fed back)",
           "date": time.strftime("%Y-%m-%d %H:%M"),
           "budgets": budgets, "solved_in_budget": 0, "detail": {}}

    total0 = time.time()
    for name, prompt, asserts in suite:
        budget = budgets[name]
        log(f"--- {name}: budget {budget:.0f}s (what the slow model needed)", a.progress)
        msgs = [{"role": "system", "content": "You are a Python expert. Complete the function. Reply ONLY with the code inside one ```python block. No explanations."},
                {"role": "user", "content": prompt}]
        attempts = []
        t0 = time.time()
        ok_final = False
        while True:
            remaining = budget - (time.time() - t0)
            if remaining < 60:
                break
            try:
                content, reasoning, dt, ctok, finish = ce.ask(
                    base, a.model, msgs, "think", a.max_tokens, http_timeout=int(remaining))
            except Exception as e:
                attempts.append({"error": str(e)[:200], "cum_s": round(time.time() - t0, 1)})
                log(f"  attempt {len(attempts)+1}: EXCEPTION {str(e)[:120]}", a.progress)
                break
            code = ce.extract_code(content) or ce.extract_code(reasoning)
            ok, err = ce.run_tests(code, asserts)
            rec = ce.attempt_metrics(content, reasoning, dt, ctok)
            rec.update({"attempt": len(attempts) + 1, "ok": ok,
                        "cum_s": round(time.time() - t0, 1), "finish": finish,
                        "code": code[:300]})
            if not ok:
                rec["err"] = err[-600:]
            attempts.append(rec)
            log(f"  attempt {len(attempts)}: {'PASS' if ok else 'fail'} "
                f"(tg={rec['tg_total']} t/s, {dt:.0f}s, cumulative {rec['cum_s']:.0f}s/{budget:.0f}s)", a.progress)
            if ok:
                ok_final = True
                break
            if finish == "length":
                repair = ("Your previous response was cut off because you spent too many "
                          "tokens inside thinking. Output the complete Python function "
                          "immediately with minimal thinking (under 1,000 tokens). "
                          "Reply ONLY with the complete code in one ```python block.")
            else:
                extra = ""
                if name == "p3_tx_engine":
                    extra = (" Analyze your logical mistake, especially regarding deadlock "
                             "cycle detection and rollback FIFO queues, and rewrite the function.")
                repair = ("Your code failed with this error:\n```\n" + err + "\n```" + extra +
                          " Reply ONLY with the corrected FULL code in one ```python block.")
            msgs = msgs + [{"role": "assistant", "content": content or code},
                           {"role": "user", "content": repair}]
        out["detail"][name] = {
            "budget_s": round(budget, 1),
            "used_s": round(time.time() - t0, 1),
            "solved_in_budget": ok_final,
            "n_attempts": len(attempts),
            "pass_at": next((i + 1 for i, v in enumerate(attempts) if v.get("ok")), None),
            "attempts": attempts,
        }
        out["solved_in_budget"] += int(ok_final)
        out["n_done"] = len(out["detail"])
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        log(f"  => {name}: {'SOLVED' if ok_final else 'NOT SOLVED'} in budget "
            f"({out['detail'][name]['used_s']:.0f}s / {budget:.0f}s, {len(attempts)} attempts)", a.progress)

    out["seconds"] = round(time.time() - total0, 1)
    out["score"] = f"{out['solved_in_budget']}/{len(suite)} inside the slow model's budget"
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    log(f"FINAL SCORE: {out['score']} in {out['seconds']}s -> {a.out}", a.progress)

if __name__ == "__main__":
    main()
