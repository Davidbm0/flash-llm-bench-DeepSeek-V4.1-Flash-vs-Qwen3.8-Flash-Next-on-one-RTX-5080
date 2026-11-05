# hle_eval.py — Humanity's Last Exam (cais/hle) pe Flash-Next (8080), subset filtrat pe categorii
# Grading: answer extraction din codul/raspunsul modelului vs answer gold (exact/numeric tolerant).
# Buget de timp per intrebare ca la timebox. Progres salvat dupa fiecare intrebare.
import json, os, re, sys, time, urllib.request

HERE = r"C:\Users\black\Desktop\Cronos Work\hle-eval"
OUT  = os.path.join(HERE, "hle_results.json")
PROG = os.path.join(HERE, "hle_progress.log")
PORT   = int(os.environ.get("HLE_PORT", "8080"))
MODEL  = os.environ.get("HLE_MODEL", "qwen3.8-flash-next-iq3_xxs")
OUT    = os.environ.get("HLE_OUT", os.path.join(HERE, "hle_results.json"))
PROG   = os.environ.get("HLE_PROG", os.path.join(HERE, "hle_progress.log"))
SETUP  = os.environ.get("HLE_SETUP", "HLE-FLASH")
MAXTOK = int(os.environ.get("HLE_MAXTOK", "16384"))
TIME_BUDGET = float(os.environ.get("HLE_BUDGET", "300"))   # s per intrebare
BASE = f"http://127.0.0.1:{PORT}/v1"

# categorii HLE; None = amestec stratificat pe toate
CATEGORIES = None            # ex: ["Physics", "Engineering", "Math"]
LIMIT        = 100           # esantion stratificat (HLE are 2500; full = ~60 h la viteza Flash)
SEED         = 42

def log(msg):
    print(msg, flush=True)
    with open(PROG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def norm(s):
    s = str(s).strip().lower()
    s = re.sub(r"[\s,]+", " ", s)
    s = s.replace("$", "").replace("\\", "").replace(" ", "")
    return s

def grade(pred, gold):
    """Exact match normalizat + tolerant numeric (rel 1e-3)."""
    if pred is None: return False
    p, g = norm(pred), norm(gold)
    if p == g: return True
    try:
        pf, gf = float(p), float(g)
        return abs(pf - gf) <= max(1e-9, abs(gf) * 1e-3)
    except ValueError:
        pass
    # gold substring in pred (pentru answers scurte intr-o fraza)
    if len(g) >= 2 and g in p: return True
    return False

def extract_answer(text):
    """Ultimul \boxed{...}, 'Answer: X', sau ultimul rand non-gol."""
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m: return m[-1].strip()
    m = re.findall(r"(?i)final answer[:\s]*([^\n]+)", text)
    if m: return m[-1].strip().strip("*").strip()
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else None

def ask(prompt, timeout):
    payload = {"model": MODEL, "max_tokens": MAXTOK, "temperature": 0.6, "top_p": 0.95,
               "chat_template_kwargs": {"thinking": True},
               "messages": [
                   {"role": "system", "content": "Answer the question with the exact final value. "
                    "Put the final answer inside \\boxed{}."},
                   {"role": "user", "content": prompt}]}
    tb = os.environ.get("HLE_THINK_BUDGET")
    if tb:
        payload["reasoning_budget_tokens"] = int(tb)
    req = urllib.request.Request(BASE + "/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        j = json.load(r)
    dt = time.time() - t0
    msg = j["choices"][0].get("message", {})
    full = (msg.get("reasoning_content") or "") + "\n" + (msg.get("content") or "")
    return full, dt, j["choices"][0].get("finish_reason", "?")

def main():
    try:
        urllib.request.urlopen(BASE + "/models", timeout=5).read()
    except Exception:
        log("Flash-Next nu e pe 8080 — porneste-l.")
        return
    import pandas as pd
    import glob
    files = glob.glob(os.path.join(HERE, "data", "**", "*.parquet"), recursive=True)
    if not files:
        log("Nu gasesc parquet HLE in data/ — download primul.")
        return
    df = pd.concat([pd.read_parquet(f) for f in files])
    log(f"HLE incarcat: {len(df)} intrebari, coloane: {list(df.columns)}")

    if CATEGORIES:
        df = df[df["category"].isin(CATEGORIES)] if "category" in df.columns else df
    # excluzem intrebarile care au imagine reala SI se bazeaza pe ea
    import numpy as np, re as _re
    lens = df["image"].apply(lambda x: len(x) if isinstance(x, str) else 0)
    pat = _re.compile(r"(?i)(the image|the figure|shown in|pictured|the diagram|the graph below|in the picture|the photo|the map|the chart|the plot)")
    has_img = np.array(lens >= 100)
    refs_img = np.array(df["question"].str.contains(pat))
    df = df[~(has_img & refs_img)]
    # esantion stratificat proportional pe categorie, seed fix
    import random
    random.seed(SEED)
    frac = LIMIT / len(df)
    parts = []
    for cat, grp in df.groupby("category"):
        k = max(1, round(len(grp) * frac))
        parts.append(grp.sample(min(k, len(grp)), random_state=SEED))
    df = pd.concat(parts).sample(frac=1, random_state=SEED).head(LIMIT)
    log(f"esantion: {len(df)} intrebari stratificate (seed {SEED})")

    out = {"setup": SETUP, "model": MODEL, "limit": LIMIT,
           "budget_s": TIME_BUDGET, "date": time.strftime("%Y-%m-%d %H:%M"),
           "n": 0, "correct": 0, "detail": []}
    t_all = time.time()
    for i, row in enumerate(df.itertuples(index=False), 1):
        q = row.question
        gold = getattr(row, "answer", None)
        t0 = time.time()
        try:
            full, dt, finish = ask(q, timeout=TIME_BUDGET)
            if finish == "length" and (time.time() - t0) < TIME_BUDGET - 60:
                # gandirea a mancat tot bugetul -> o incercare cu gandire minimala
                msgs2 = [{"role": "system", "content": "Answer directly and briefly. Do NOT explore. "
                          "Put the final answer inside \\boxed{}."},
                         {"role": "user", "content": q}]
                payload = {"model": MODEL, "max_tokens": 2048, "temperature": 0.1,
                           "chat_template_kwargs": {"thinking": False}, "messages": msgs2}
                req = urllib.request.Request(BASE + "/chat/completions",
                                             data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=int(TIME_BUDGET - (time.time() - t0))) as r:
                    j = json.load(r)
                full = j["choices"][0]["message"].get("content") or ""
                dt = time.time() - t0
            ans = extract_answer(full)
            ok = grade(ans, gold)
        except Exception as e:
            ans, ok, dt = None, False, round(time.time() - t0, 1)
            log(f"  [{i}] EXCEPTIE {str(e)[:100]}")
        out["n"] += 1
        out["correct"] += int(ok)
        out["detail"].append({"id": getattr(row, "id", str(i)),
                              "category": getattr(row, "category", "?"),
                              "gold": str(gold)[:120], "pred": str(ans)[:120],
                              "ok": ok, "t_s": round(dt, 1), "finish": finish})
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        log(f"  [{i}/{len(df)}] {'CORECT' if ok else 'gresit'} ({dt:.0f}s) "
            f"gold={str(gold)[:40]} pred={str(ans)[:40]}")
    out["seconds"] = round(time.time() - t_all, 1)
    out["score"] = f"{out['correct']}/{out['n']}"
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    log(f"SCOR: {out['score']} in {out['seconds']}s -> {OUT}")

if __name__ == "__main__":
    main()
