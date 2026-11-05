# accuracy_5v6.py — comparatie acuratete pass@1: 5 experti (cfg_M) vs 6 experti (cfg_J6)
# Ambele pe cfg de viteza M (LFU + --ssd-cache 76 + budget 12000), singura diferenta
# = --experts 6. Suite: P1-P5 + E1-E5 = 10 probleme, think, single step (pass@1).
# Ruleaza ~50-100 min per arm. Total ~2-3.5 h. Nu atinge Flash-Next (8080).
# Daca intrerupi: rezultatele per-problema se vad oricum in acc_<nume>_stdout.log.
import json, os, subprocess, sys, time, urllib.request

TUNE   = r"C:\Users\black\Desktop\Cronos Work\dsv41-tuning"
QWEN   = r"C:\Users\black\Desktop\Cronos Work\qwen35-comparatie"
PY     = r"C:\Users\black\Desktop\Strata-dsv4-dl\Strata-main\.venv\Scripts\python.exe"
SERVER = r"C:\Users\black\Desktop\Strata-dsv4-dl\Strata-main\serve\server.py"
EVAL   = os.path.join(TUNE, "acc_eval.py")   # wrapper peste coding_eval_extreme.py
CWD    = r"C:\Users\black\Desktop\Strata-dsv4-dl\Strata-main"
PORT   = 8081

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def free_ram_gb():
    import ctypes
    class MSG(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MSG(); m.dwLength = ctypes.sizeof(MSG)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 1024**3

def api(path, body=None, timeout=60):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def wait_ready(proc, timeout=1500):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            d = api("/v1/models", timeout=8)
            for m in d.get("data", []):
                if "deepseek" in str(m.get("id", "")).lower():
                    st = (m.get("status") or {}).get("value")
                    if st in (None, "loaded"):
                        return True
        except Exception:
            pass
        time.sleep(10)
    return False

def make_cfg_j6():
    """cfg_J (experts 6) + --ssd-cache 76 ca sa fie comparabil cu M."""
    with open(os.path.join(TUNE, "cfg_J.json"), encoding="utf-8") as f:
        c = json.load(f)
    if "--ssd-cache" not in c["args"]:
        c["args"] += ["--ssd-cache", "76"]
    c["log"] = os.path.join(TUNE, "acc_J6_engine.log")
    p = os.path.join(TUNE, "cfg_J6.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(c, f, indent=1)
    return p

def run_eval(name, cfgpath, results):
    free = free_ram_gb()
    if free < 82:
        log(f"{name}: REFUZAT — RAM liber {free:.1f} GB (< 82). Ruleaza STOP-FLASH.bat.")
        return
    log(f"{name}: pornire server :{PORT} (RAM liber {free:.1f} GB)")
    logf = open(os.path.join(TUNE, f"acc_{name}_stdout.log"), "w", encoding="utf-8")
    proc = subprocess.Popen([PY, SERVER, "--engine", "strata", "--config", cfgpath, "--port", str(PORT)],
                            cwd=CWD, stdout=logf, stderr=subprocess.STDOUT)
    ok = wait_ready(proc)
    log(f"{name}: ready={ok}")
    if ok:
        # verifica din log engine ca numarul de experti e cel dorit
        try:
            with open(json.load(open(cfgpath, encoding="utf-8"))["log"], encoding="utf-8", errors="replace") as f:
                txt = f.read()
            import re
            m = re.search(r"experts per word: (\d+)", txt)
            log(f"{name}: engine zice experts/word = {m.group(1) if m else 'N/A (split/full)'}")
        except Exception:
            pass
        ev = subprocess.run(
            [PY, EVAL, "--port", str(PORT), "--model", "deepseek-v4.1-flash",
             "--setup", f"DSV41-{name}", "--mode", "think", "--max-repairs", "0",
             "--max-tokens", "16384", "--http-timeout", "3600"],
            cwd=QWEN, stdout=logf, stderr=subprocess.STDOUT)
        log(f"{name}: eval terminat (exit {ev.returncode})")
    try:
        proc.terminate()
        proc.wait(timeout=60)
    except Exception:
        pass
    # citeste rezultatul scris de harness
    rp = os.path.join(QWEN, "results", f"DSV41-{name}_think.json")
    try:
        with open(rp, encoding="utf-8") as f:
            r = json.load(f)
        results[name] = {"score": r.get("score"), "seconds": r.get("seconds"),
                         "detail": {k: v.get("pass_at") for k, v in r.get("detail", {}).items()}}
        log(f"{name}: SCOR {r.get('score')} in {r.get('seconds')}s")
    except OSError:
        log(f"{name}: FARA rezultat ({rp} absent)")

def main():
    results = {}
    run_eval("M5", os.path.join(TUNE, "cfg_M.json"), results)
    time.sleep(15)
    run_eval("J6", make_cfg_j6(), results)
    out = os.path.join(TUNE, "accuracy_5v6.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    log("FINAL:")
    for k, v in results.items():
        log(f"  {k}: {v['score']}  pass_at per problema: {v['detail']}")
    log(f"salvat: {out}")

if __name__ == "__main__":
    main()
