# flash_timebox.py — Qwen3.8-Flash-Next (port 8080) pe suita de 10 probleme (P1-P5 + E1-E5)
# REGULA: buget de timp per problema = timpul exact in care DeepSeek M5 a rezolvat-o (pass@1,
# din DSV41-M5_think.json). Flash primeste acelasi wall-clock budget dar cu self-repair
# iterativ: atatea incercari cate incap in buget (eroarea exacta e trimisa inapoi in chat).
# Metric final: X/10 rezolvate in bugetul DeepSeek. Progres salvat dupa FIECARE problema.
import json, os, sys, time, urllib.request

HERE = r"C:\Users\black\Desktop\Cronos Work\dsv41-tuning"
QWEN = r"C:\Users\black\Desktop\Cronos Work\qwen35-comparatie"
sys.path.insert(0, QWEN)
sys.path.insert(0, HERE)

import coding_eval_extreme as ce
import new_problems

BUDGETS = os.path.join(QWEN, "results", "DSV41-M5_think.json")
OUT  = os.path.join(HERE, "flash_timebox.json")
PROG = os.path.join(HERE, "flash_timebox_progress.log")

PORT, MODEL, MAXTOK = 8080, "qwen3.8-flash-next-iq3_xxs", 16384
BASE = f"http://127.0.0.1:{PORT}/v1"

def log(msg):
    print(msg, flush=True)
    with open(PROG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def main():
    try:
        urllib.request.urlopen(BASE + "/models", timeout=5).read()
    except Exception:
        log("FLASH-Next nu e pornit pe 8080 — porneste-l (run-iq3_xxs.bat) si ruleaza din nou.")
        return

    budgets = {k: v["attempts"][0]["t_s"]
               for k, v in json.load(open(BUDGETS, encoding="utf-8"))["detail"].items()}
    suite = ce.build_problems("ref14") + new_problems.build_new_problems()

    out = {"setup": "FLASH-TIMEBOX", "model": MODEL, "mode": "think",
           "rule": "buget per problema = t_s DeepSeek M5 (pass@1 single step); "
                   "self-repair iterativ in buget (traceback exact trimis inapoi)",
           "date": time.strftime("%Y-%m-%d %H:%M"),
           "budgets": budgets, "pass1_budget": 0, "detail": {}}

    total0 = time.time()
    for name, prompt, asserts in suite:
        budget = budgets[name]
        log(f"--- {name}: buget {budget:.0f}s (cat a facut DeepSeek M5)")
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
                    BASE, MODEL, msgs, "think", MAXTOK, http_timeout=int(remaining))
            except Exception as e:
                attempts.append({"error": str(e)[:200], "cum_s": round(time.time() - t0, 1)})
                log(f"  incercarea {len(attempts)+1}: EXCEPTIE {str(e)[:120]}")
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
            log(f"  incercarea {len(attempts)}: {'PASS' if ok else 'fail'} "
                f"(tg={rec['tg_total']} t/s, {dt:.0f}s, cumulativ {rec['cum_s']:.0f}s/{budget:.0f}s)")
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
        out["pass1_budget"] += int(ok_final)
        out["n_done"] = len(out["detail"])
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        log(f"  => {name}: {'REZOLVAT' if ok_final else 'NESOLVAT'} in buget "
            f"({out['detail'][name]['used_s']:.0f}s / {budget:.0f}s, {len(attempts)} incercari)")

    out["seconds"] = round(time.time() - total0, 1)
    out["score"] = f"{out['pass1_budget']}/{len(suite)} in bugetul DeepSeek"
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    log(f"SCOR FINAL: {out['score']} in {out['seconds']}s -> {OUT}")

if __name__ == "__main__":
    main()
