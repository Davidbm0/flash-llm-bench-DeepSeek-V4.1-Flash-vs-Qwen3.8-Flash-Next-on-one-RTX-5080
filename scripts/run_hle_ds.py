# run_hle_ds.py — ruleaza ACELASI esantion HLE (100, seed 42) pe DeepSeek V4.1-Flash (cfg_M, port 8081)
# Flow: STOP-FLASH -> porneste server DeepSeek 8081 -> hle_eval.py (buget identic 300s/intrebare) -> opreste serverul.
# Buget identic = comparatie time-boxed corecta (timpul e metricul, nu tokenii).
import json, os, subprocess, sys, time, urllib.request

TUNE   = r"C:\Users\black\Desktop\Cronos Work\dsv41-tuning"
HERE   = r"C:\Users\black\Desktop\Cronos Work\hle-eval"
CWD    = r"C:\Users\black\Desktop\Strata-dsv4-dl\Strata-main"
PY     = os.path.join(CWD, ".venv", "Scripts", "python.exe")          # server (Strata venv)
PY_EVAL = r"C:\Users\black\Desktop\Cronos Work\hle-eval\.venv\Scripts\python.exe"  # eval (are pandas)
SERVER = os.path.join(CWD, "serve", "server.py")
PORT   = 8081

def free_ram_gb():
    import ctypes
    class M(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong)]
    m = M(); m.dwLength = ctypes.sizeof(M)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2**30

def log(msg):
    print(msg, flush=True)

def main():
    free = free_ram_gb()
    if free < 82:
        log(f"RAM liber {free:.1f} GB (< 82). Ruleaza intai STOP-FLASH.bat — DeepSeek are nevoie de store ~76 GB.")
        return
    log(f"pornire server DeepSeek :{PORT} (RAM liber {free:.1f} GB)")
    logf = open(os.path.join(HERE, "hle_ds_server.log"), "w", encoding="utf-8")
    proc = subprocess.Popen([PY, SERVER, "--engine", "strata",
                             "--config", os.path.join(TUNE, "cfg_M.json"), "--port", str(PORT)],
                            cwd=CWD, stdout=logf, stderr=subprocess.STDOUT)
    # asteapta ready (max 6 min)
    ready = False
    for _ in range(72):
        time.sleep(5)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=3).read()
            ready = True
            break
        except Exception:
            if proc.poll() is not None:
                break
    log(f"server ready: {ready}")
    if not ready:
        log("server nu a pornit — vezi hle_ds_server.log")
        try: proc.terminate()
        except Exception: pass
        return
    # la ~9-14 tok/s, in 300s incap ~2700-4000 tokeni total -> gandire plafonata ca sa
    # NU depaseasca bugetul (altfel clientul da timeout si engine-ul continua in follow)
    env = dict(os.environ,
               HLE_PORT=str(PORT), HLE_MODEL="deepseek-v4.1-flash",
               HLE_OUT=os.path.join(HERE, "hle_results_ds.json"),
               HLE_PROG=os.path.join(HERE, "hle_progress_ds.log"),
               HLE_SETUP="HLE-DS-M5",
               HLE_MAXTOK="2800", HLE_THINK_BUDGET="1800")
    ev = subprocess.run([PY_EVAL, os.path.join(HERE, "hle_eval.py")], cwd=HERE, env=env)
    log(f"eval terminat (exit {ev.returncode})")
    try:
        proc.terminate(); proc.wait(timeout=60)
    except Exception:
        pass
    log("gata — vezi hle_results_ds.json")

if __name__ == "__main__":
    main()
