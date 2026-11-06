# run_hle_ds.py — run the SAME HLE sample against a second (slower) model
#
# Flow: start your server on HLE_PORT -> run hle_eval.py with env vars -> stop server.
# The point: identical sample (100 questions, seed 42) and identical 300 s/question
# budget for both models, so the comparison is time-boxed (time is the metric, not tokens).
#
# For a llama.cpp-class server, the equivalent is just:
#   HLE_PORT=8080 HLE_MODEL=your-model python hle_eval.py
# DeepSeek-specific bits below (RAM gate, server spawn) are kept as an example only.

import os, subprocess, sys, time, urllib.request

HERE   = os.path.dirname(os.path.abspath(__file__))
PORT   = 8081

def free_ram_gb():
    import ctypes
    class M(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong)]
    m = M(); m.dwLength = ctypes.sizeof(M)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2**30

def log(msg):
    print(msg, flush=True)

def main():
    # our box: the MoE needs ~76 GB free RAM for its expert store; skip this gate
    # if your model fits in VRAM
    free = free_ram_gb()
    if free < 82:
        log(f"free RAM {free:.1f} GB (< 82) — stop the other model server first.")
        return
    log(f"start your model server on :{PORT}, then re-run this script.")
    log(f"(free RAM {free:.1f} GB)")
    log("waiting for server (max 6 min)...")
    ready = False
    for _ in range(72):
        time.sleep(5)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=3).read()
            ready = True
            break
        except Exception:
            pass
    log(f"server ready: {ready}")
    if not ready:
        return
    # at ~9-14 tok/s, 300 s fits ~2700-4000 tokens total -> cap thinking so the
    # model does not exceed the per-question budget (client timeout mid-generation)
    env = dict(os.environ,
               HLE_PORT=str(PORT), HLE_MODEL="deepseek-v4.1-flash",
               HLE_OUT=os.path.join(HERE, "hle_results_ds.json"),
               HLE_PROG=os.path.join(HERE, "hle_progress_ds.log"),
               HLE_SETUP="HLE-DS-M5",
               HLE_MAXTOK="2800", HLE_THINK_BUDGET="1800")
    ev = subprocess.run([sys.executable, os.path.join(HERE, "hle_eval.py")], cwd=HERE, env=env)
    log(f"eval finished (exit {ev.returncode}) — see hle_results_ds.json")

if __name__ == "__main__":
    main()
