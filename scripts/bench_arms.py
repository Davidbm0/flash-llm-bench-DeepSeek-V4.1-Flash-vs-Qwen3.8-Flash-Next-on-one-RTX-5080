# bench_arms.py — A/B/C benchmark pentru DeepSeek V4.1-Flash pe Strata (:8081)
# Nu atinge porturile 8080/8082 (Flash-Next/Gemma). Opreste doar serverul DeepSeek
# pe care l-a pornit el + engine-ul strata-dsv41.exe.
import json, os, subprocess, sys, time, urllib.request

TUNE   = r"C:\Users\black\Desktop\Cronos Work\dsv41-tuning"
ORIG   = r"C:\Users\black\Desktop\Strata-dsv4-dl\Strata-main\strata-deepseek-v4.1-flash-ist-das-2.25bpw.json"
PY     = r"C:\Users\black\Desktop\Strata-dsv4-dl\Strata-main\.venv\Scripts\python.exe"
SERVER = r"C:\Users\black\Desktop\Strata-dsv4-dl\Strata-main\serve\server.py"
CWD    = r"C:\Users\black\Desktop\Strata-dsv4-dl\Strata-main"
OUT    = os.path.join(TUNE, "arms_results.json")
PORT   = 8081

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_orig():
    with open(ORIG, encoding="utf-8") as f:
        return json.load(f)

def arm_cfg(name, extra_args=(), extra_keys=None):
    cfg = load_orig()
    cfg["args"] = list(cfg.get("args", [])) + list(extra_args)
    cfg["log"] = os.path.join(TUNE, f"arm_{name}_engine.log")
    cfg["port"] = PORT
    for k, v in (extra_keys or {}).items():
        cfg[k] = v
    p = os.path.join(TUNE, f"cfg_{name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    return p

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

def chat(messages, max_tokens, think, temperature=0.1):
    body = {"model": "deepseek-v4.1-flash", "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature}
    if not think:
        body["reasoning_effort"] = "none"
    t0 = time.time()
    d = api("/v1/chat/completions", body, timeout=2400)
    dt = time.time() - t0
    u = d.get("usage", {})
    ct = u.get("completion_tokens", 0)
    pt = u.get("prompt_tokens", 0)
    return {"s": round(dt, 1), "pt": pt, "ct": ct,
            "tok_s": round(ct / dt, 2) if dt > 0 else 0}

PREFILL_TEXT = ("Ion este inginer de semnal si proiecteaza linii de transmisie pe PCB cu control de impedanta. " * 320)
THINK_TASK = ("Scrie o functie Python `transaction_engine(ops)` care simuleaza o engine de tranzactii cu lock-uri "
              "pe cheile A si B, savepoint, rollback care elibereaza lock-urile luate dupa savepoint, o coada de "
              "waiters care sunt re-executati la release, si abort cu detectie de deadlock. Returneaza lista de "
              "evenimente si valorile finale ale cheilor. Nu folosi librarii externe.")

def grep_engine_log(path):
    out = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if "RAM store:" in line or "CPU lane" in line or "experts per word" in line \
                        or "on the CPU lane" in line or "in VRAM" in line:
                    out.append(line.strip())
    except OSError:
        pass
    return out[-8:]

def run_arm(name, cfgpath, results, flash=False):
    log(f"ARM {name}: pornire server :{PORT}")
    logf = open(os.path.join(TUNE, f"arm_{name}_stdout.log"), "w", encoding="utf-8")
    proc = subprocess.Popen([PY, SERVER, "--engine", "strata", "--config", cfgpath, "--port", str(PORT)],
                            cwd=CWD, stdout=logf, stderr=subprocess.STDOUT)
    r = {"arm": name, "started": time.strftime("%d.%m %H:%M:%S"), "flash_next_contaminat": flash}
    ok = wait_ready(proc)
    r["ready"] = ok
    log(f"ARM {name}: ready={ok}")
    if ok:
        try:
            time.sleep(5)
            # warm-up: incalzeste cache-ul de fisiere + profilul adaptiv, nu se masoara
            chat([{"role": "user", "content": "Scrie o lista numerotata cu 120 de fapte stiintifice scurte."}],
                 300, False)
            log(f"ARM {name}: warm-up gata")
            r["decode"] = [chat([{"role": "user",
                                  "content": "Scrie o lista numerotata cu 150 de fapte stiintifice scurte."}],
                                800, False) for _ in range(3)]
            log(f"ARM {name}: decode {[d['tok_s'] for d in r['decode']]}")
            r["prefill"] = chat([{"role": "user", "content": PREFILL_TEXT + " Sumar: care e ideea principala? (un rand)"}], 1, False)
            log(f"ARM {name}: prefill {r['prefill']}")
            r["think"] = chat([{"role": "user", "content": THINK_TASK}], 4000, True, 0.6)
            log(f"ARM {name}: think {r['think']}")
        except Exception as e:
            r["error"] = repr(e)
            log(f"ARM {name}: EROARE {e!r}")
    # oprire: serverul nostru + engine-ul DeepSeek (NU strata.exe / llama-server.exe)
    try:
        proc.terminate()
    except Exception:
        pass
    time.sleep(6)
    subprocess.run(["taskkill", "/PID", str(proc.pid), "/F", "/T"], capture_output=True)
    subprocess.run(["taskkill", "/IM", "strata-dsv41.exe", "/F"], capture_output=True)
    time.sleep(4)
    r["engine_lines"] = grep_engine_log(os.path.join(TUNE, f"arm_{name}_engine.log"))
    r["ended"] = time.strftime("%d.%m %H:%M:%S")
    results.append(r)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    log(f"ARM {name}: gata")

def flash_next_running():
    import socket
    for port in (8080, 8082):
        s = socket.socket()
        s.settimeout(1.5)
        try:
            s.connect(("127.0.0.1", port))
            return port
        except Exception:
            pass
        finally:
            s.close()
    return None

def free_ram_gb():
    import ctypes
    class MSS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MSS(); m.dwLength = ctypes.sizeof(MSS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2**30

def main():
    results = []
    if os.path.exists(OUT):
        try:
            with open(OUT, encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            results = []
    done = {r.get("arm") for r in results if r.get("ready")}
    # curata leftovers (doar engine-ul DeepSeek, NU strata.exe/llama-server)
    subprocess.run(["taskkill", "/IM", "strata-dsv41.exe", "/F"], capture_output=True)
    all_arms = {
        "A": arm_cfg("A"),
        "B": arm_cfg("B", extra_args=["--cpu-threads", "16"]),
        "C": arm_cfg("C", extra_args=["--cpu-threads", "16"],
                     extra_keys={"reasoning_budget_tokens": 12000}),
        # D: layer split 5080+3080 -- DRUMUL CORECT pentru DeepSeek: "gpu":[0,1] il face
        # pe server sa adauge --layer-split auto. (--peer-device e doar pt engine-ul Qwen,
        # maya-ds4 il refuza -> "unknown option", exit 2.) Serverul aranjeaza singur:
        # placa mai lenta primele straturi, cea rapida ultimul stage + head.
        "D": arm_cfg("D", extra_keys={"gpu": [0, 1]}),
        # E: context 16K (KV mai mic pe VRAM -> mai multi experts in VRAM) + budget 12000
        "E": arm_cfg("E", extra_args=["--max-context", "16384"],
                     extra_keys={"reasoning_budget_tokens": 12000}),
        # F: policy LFU pe RAM store (pastreaza expertii fierbiti in loc de LRU)
        "F": arm_cfg("F", extra_args=["--ssd-policy", "lfu"]),
        # G: baseline + budget 12000 (fara cpu-threads 16, ca sa izolez efectul bugetului)
        "G": arm_cfg("G", extra_keys={"reasoning_budget_tokens": 12000}),
        # H: leverele bune cumulate pe o placa: LFU + budget 12000 (context 32K pastrat)
        "H": arm_cfg("H", extra_args=["--ssd-policy", "lfu"],
                     extra_keys={"reasoning_budget_tokens": 12000}),
        # I: split 5080+3080 + LFU + budget 12000 (cele mai bune 3 levere impreuna)
        "I": arm_cfg("I", extra_args=["--ssd-policy", "lfu"],
                     extra_keys={"gpu": [0, 1], "reasoning_budget_tokens": 12000}),
        # R: CONTROL = identic cu A, ca sa avem baseline proaspat in aceeasi runda
        # (profilul de experti se incalzeste peste zi -> A de la 19:00 nu-i comparabil
        #  cu D-G de la 22:50; R masoara driftul)
        "R": arm_cfg("R"),
        # J: H + --experts 6 (full quality) — cat costa viteza cand folosesti toti
        # 6 experti/word in loc de 5 (modul "faster" e default pe 16GB VRAM)
        "J": arm_cfg("J", extra_args=["--experts", "6", "--ssd-policy", "lfu"],
                     extra_keys={"reasoning_budget_tokens": 12000}),
        # K: split + --experts 5 — D/I par sa ruleze cu 6 (default full quality cand
        # VRAM total > 16 GiB); K compara split-ul la egalitate cu arms pe 5 experti
        "K": arm_cfg("K", extra_args=["--experts", "5", "--ssd-policy", "lfu"],
                     extra_keys={"gpu": [0, 1], "reasoning_budget_tokens": 12000}),
        # L/M/N: store fixat cu --ssd-cache (capa headroom-ul implicit de 12 GB).
        # La pornire curata RAM liber dupa weights ~82-84 GB:
        #   L = store 78 GB -> headroom ~4-6 GB (paso intermediar, sigur)
        #   M = store 81 GB -> headroom ~1-3 GB (ce a cerut Sami)
        #   N = store 82 GB -> headroom ~0-2 GB (extrem; daca Windows cedeaza, pică)
        # Totul pe H (LFU + budget 12000). Profilele fierbinti: ~8000 pairs;
        # 71.9 GB = 6314 slots -> 81 GB = ~7100 slots (mai multi experti fierbinti in RAM).
        "L": arm_cfg("L", extra_args=["--ssd-policy", "lfu", "--ssd-cache", "78"],
                     extra_keys={"reasoning_budget_tokens": 12000}),
        "M": arm_cfg("M", extra_args=["--ssd-policy", "lfu", "--ssd-cache", "81"],
                     extra_keys={"reasoning_budget_tokens": 12000}),
        "N": arm_cfg("N", extra_args=["--ssd-policy", "lfu", "--ssd-cache", "82"],
                     extra_keys={"reasoning_budget_tokens": 12000}),
    }
    wanted = [a.upper() for a in sys.argv[1:]] or list(all_arms)  # default: tot, sare ce-i facut
    for w in wanted:
        if w not in all_arms:
            log(f"ARM {w} nu exista (arm-urile disponibile: {', '.join(all_arms)})")
    fp = flash_next_running()
    if fp:
        log(f"ATENIE: Flash-Next/Gemma pe :{fp} — RAM contaminat, expect ~4-5 tok/s in loc de ~10")
    for name in wanted:
        if name not in all_arms:
            continue
        if name in done:
            log(f"ARM {name}: deja rulat, sarit (sterge arms_results.json ca sa-l reiei)")
            continue
        # L/M/N: store ADAPTIV — taie headroom-ul engineului la tinta dorita.
        # Citim RAM liber acum si fixam --ssd-cache = liber - headroom_tinta.
        # (marimile fixe 78/81/82 din prima versiune erau orbi: garda le refuza
        #  pentru ca liberul real e ~85, nu 86-90)
        headroom_target = {"L": 6, "M": 3, "N": 1.5}.get(name)
        if headroom_target is not None:
            free = free_ram_gb()
            if free < 70:
                log(f"[{name}] REFUZAT: cere RAM liber >= 70 GB, ai {free:.1f} GB. "
                    f"Ruleaza STOP-FLASH.bat primul, apoi RUN-BENCH.bat.")
                continue
            store = int(free - 5 - headroom_target)  # ~5 GB pentru weights+CUDA inainte de store
            cfgp = os.path.join(TUNE, f"cfg_{name}.json")
            with open(cfgp, encoding="utf-8") as f:
                c = json.load(f)
            i = c["args"].index("--ssd-cache")
            c["args"][i + 1] = str(store)
            with open(cfgp, "w", encoding="utf-8") as f:
                json.dump(c, f, indent=1)
            log(f"[{name}] store adaptat: liber {free:.1f} GB -> --ssd-cache {store} "
                f"(headroom tinta ~{headroom_target} GB)")
        run_arm(name, all_arms[name], results, flash=bool(fp))
    log("FINAL:")
    for r in results:
        dec = [d.get("tok_s") for d in r.get("decode", [])]
        th = r.get("think", {}).get("tok_s")
        log(f"  {r['arm']}: decode={dec} think={th} ready={r.get('ready')}")

if __name__ == "__main__":
    main()
