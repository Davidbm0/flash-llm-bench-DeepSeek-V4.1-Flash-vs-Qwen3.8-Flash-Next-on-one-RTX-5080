#!/usr/bin/env python3
"""Extreme Coding & Engineering Eval — 10 problems (P1-P5 + E1-E5), Pass@1 ... Pass@5 (self-repair).

Talks to any OpenAI-compatible /v1/chat/completions endpoint (llama.cpp, Strata, vLLM, ...).
Ground truth for E1-E5 is re-derivable with ref_extreme5.py (run it: 5/5 on the references).

Hyperparameters used in our runs:
 - max_tokens 16384 for all
 - think: temp 0.6, top_p 0.95, min_p 0.0 | nothink: temp 0.1
 - pass@5: max 4 self-repairs; the exact error (AssertionError + traceback) is fed back
   as a user message; a truncated response (finish=length) gets a "think less" repair instead
"""
import json, subprocess, sys, time, argparse, os, re, urllib.request

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

P1_PROMPT = """Write a Python function `parity_turn_dijkstra(n: int, edges: list[tuple[int, int, int, int]], start: int, dst: int) -> int` for a DIRECTED graph with n nodes (0..n-1); each edge is (u, v, weight, color).
Path cost rules:
1. Step count starts at 1 for the first edge taken.
2. On EVEN steps (2nd, 4th, 6th edge...), the edge weight is multiplied by 2.
3. On ODD steps (1st, 3rd, 5th edge...), the edge weight is normal (1x).
4. TURN PENALTY: if you traverse edge (u, v, w2, c2) immediately after (prev, u, w1, c1) and c1 != c2, you pay an additional penalty abs(c1 - c2) * 3 (added AFTER parity multiplication). The first edge has no turn penalty.
5. CONSECUTIVE SAME-COLOR BAN: you CANNOT traverse three edges of the exact same color in a row.
Return the minimum cost from start to dst, or -1 if unreachable."""
P1_ASSERTS = [
    "edges1 = [(0,1,5,1),(1,2,5,1),(2,3,5,1),(1,4,2,2),(4,2,2,2)]",
    "assert parity_turn_dijkstra(5, edges1, 0, 3) == 27",
    "edges3 = [(0,1,10,1),(1,2,100,1),(0,3,1,2),(3,1,1,1)]",
    "assert parity_turn_dijkstra(4, edges3, 0, 2) == 106",
    "assert parity_turn_dijkstra(3, [(0,1,1,1)], 0, 2) == -1",
]

P2_PROMPT = """Write a Python function `solve_microstrip_width(z_target: float, h_mm: float, er: float, t_oz: float) -> dict` using ONLY the Python standard library (`math`). Find the trace width `w_mm` achieving target impedance `z_target` (Ohms) for an outer-layer microstrip, using Hammerstad & Jensen equations with Wheeler strip-thickness correction:
1. t_mm = t_oz * 0.034798
2. u = w / h_mm. If t_mm > 0:
   du1 = (t_mm/h_mm)/math.pi * math.log(1.0 + (4.0*math.e)/((t_mm/h_mm) * (1.0/math.tanh(math.sqrt(6.517*u)))**2))
   du_r = 0.5*du1*(1.0 + 1.0/math.cosh(math.sqrt(er-1.0)))
   u1 = u + du1 ; ur = u + du_r. Else u1 = ur = u.
3. a = 1.0 + (1.0/49.0)*math.log((ur**4 + (ur/52.0)**2)/(ur**4 + 0.432)) + (1.0/18.7)*math.log(1.0 + (ur/18.1)**3)
   b = 0.564*((er-0.9)/(er+3.0))**0.053
   eeff = (er+1.0)/2.0 + ((er-1.0)/2.0)*(1.0 + 10.0/ur)**(-a*b)
4. f_u(x) = 6.0 + (2.0*math.pi - 6.0)*math.exp(-(30.666/x)**0.7528)
   z0_freespace(x) = (376.730313668/(2.0*math.pi)) * math.log(f_u(x)/x + math.sqrt(1.0 + (2.0/x)**2))
   Z0 = z0_freespace applied to the corrected width / sqrt(eeff)
Use numerical root-finding (bisection/Brent) on w_mm in [0.01, 20.0], accuracy 1e-7.
Return dict with keys rounded to 4 decimals: 'w_mm', 'w_mils' (1 mm = 39.3700787 mils), 'eeff' (at solution), 'delay_ps_per_mm' (propagation delay, c = 299792458 m/s)."""
P2_ASSERTS = [
    "res = solve_microstrip_width(50.0, 0.2, 4.2, 1.0)",
    # band covering both legitimate interpretations of the formula (A: 0.3514, B: 0.3658)
    "assert 0.349 <= res['w_mm'] <= 0.368",
    "assert 13.7 <= res['w_mils'] <= 14.5",
    "assert abs(res['w_mils'] - res['w_mm']*39.3700787) < 0.1",
    "assert abs(res['eeff'] - 3.199) < 0.015",
    "assert abs(res['delay_ps_per_mm'] - 5.965) < 0.03",
]

P3_PROMPT = """Write a Python function `run_tx_engine(ops: list[tuple]) -> list` simulating an in-memory transactional key-value store with pessimistic Exclusive Locks and Savepoints.
Operations:
- ("BEGIN", tx_id): start transaction, record "OK".
- ("SET", tx_id, key, val): requires exclusive lock on key.
  - If unlocked or already locked by tx_id: write val, record "OK".
  - If locked by another ACTIVE tx_other: check if adding edge tx_id -> tx_other creates a cycle in the Wait-For graph.
    - If DEADLOCK: record "DEADLOCK_ABORT", then abort tx_id (release ALL its locks, discard uncommitted writes), unblocking waiting transactions in FIFO order (their queued operations execute and record their results).
    - Otherwise tx_id enters waiting state: record "WAITING" and queue its operation.
- ("SAVEPOINT", tx_id, sp_name): snapshot of current writes AND locks acquired. Record "OK".
- ("ROLLBACK_TO", tx_id, sp_name): revert writes made after sp_name and release locks FIRST ACQUIRED after sp_name (triggering queued waiters in FIFO order — their queued ops execute now). Record "OK".
- ("COMMIT", tx_id): apply uncommitted writes to global store, release all locks (triggering queued waiters). Record "COMMITTED".
- ("GET", key): read latest COMMITTED value (or None), record it.
Return the list of recorded results for each operation in order (including results of queued operations when they are unblocked)."""
# validated variant (correct FIFO engine, 14 entries)
P3_ASSERTS_REF14 = [
    "ops = [('BEGIN','T1'),('BEGIN','T2'),('SET','T1','A',10),('SAVEPOINT','T1','sp1'),",
    "       ('SET','T1','B',20),('SET','T2','B',99),('ROLLBACK_TO','T1','sp1'),",
    "       ('SET','T1','B',30),('SET','T2','A',88),('COMMIT','T1'),('GET','A'),('GET','B')]",
    "assert run_tx_engine(ops) == ['OK','OK','OK','OK','OK','WAITING','OK','OK','WAITING','DEADLOCK_ABORT','OK','COMMITTED',10,30]",
]

P4_PROMPT = """Write a Python function `analyze_pdn_impedance(caps: list[tuple[float, float, float]], f_start_hz: float, f_stop_hz: float, num_points: int) -> dict`.
Each capacitor in caps is (C_farads, ESR_ohms, ESL_henries) connected in parallel between VDD and GND.
Evaluate at num_points logarithmically spaced frequencies (inclusive): f_k = f_start_hz * (f_stop_hz/f_start_hz)**(k/(num_points-1)), k = 0..num_points-1.
At each frequency compute total parallel impedance magnitude via complex admittance:
Y(w) = sum over caps of 1/(ESR_i + j*(w*ESL_i - 1/(w*C_i))), Z = 1/|Y|, with w = 2*pi*f.
Identify all local ANTI-RESONANCE peaks (strictly interior points 0 < k < num_points-1 where |Z[k]| > |Z[k-1]| and |Z[k]| > |Z[k+1]|).
Return dict with: 'max_anti_res_freq_mhz' (MHz, rounded 2 dec), 'max_anti_res_z_mohm' (milliohms at the HIGHEST peak, rounded 2 dec), 'min_z_mohm' (global minimum across sweep, rounded 2 dec)."""
P4_ASSERTS = [
    "caps = [(10e-6, 0.005, 800e-12), (100e-9, 0.015, 500e-12), (1e-9, 0.050, 400e-12)]",
    "res = analyze_pdn_impedance(caps, 1e5, 1e9, 4001)",
    "assert abs(res['max_anti_res_freq_mhz'] - 188.80) < 0.5",
    "assert abs(res['max_anti_res_z_mohm'] - 2366.58) < 5.0",
    "assert abs(res['min_z_mohm'] - 5.00) < 0.2",
]

P5_PROMPT = """Write a Python function `optimize_and_execute(ast_code: dict, initial_regs: dict) -> dict` for a mini-assembly language given as a nested AST dict.
Statement format (each stmt is a dict):
- {"op": "SET", "reg": r, "val": x}      -> regs[r] = x; x is a number OR a register name (string)
- {"op": "ADD"|"SUB"|"MUL", "dst": d, "a": x, "b": y} -> regs[d] = x op y; operands are numbers or register names
- {"op": "REPEAT", "n": N, "body": [...]} -> execute body N times; N is a number or register name
- {"op": "IF", "reg": r, "gt": v, "body": [...]} -> if regs[r] > v then execute body
ast_code = {"body": [stmts]}. Registers can be read/written inside nested bodies. A register that was never set reads as 0.
Perform constant folding and loop unrolling where possible, then return the final register state as a dict (keys you touched plus initial regs; extra keys allowed)."""
P5_ASSERTS = [
    "ast1 = {'body':[{'op':'SET','reg':'a','val':2},{'op':'SET','reg':'b','val':3},",
    "  {'op':'REPEAT','n':4,'body':[{'op':'ADD','dst':'a','a':'a','b':2}]},",
    "  {'op':'SET','reg':'c','val':'a'},",
    "  {'op':'IF','reg':'c','gt':5,'body':[{'op':'MUL','dst':'c','a':'c','b':3}]},",
    "  {'op':'SUB','dst':'d','a':'c','b':'a'}]}",
    "out1 = optimize_and_execute(ast1, {})",
    "assert all(out1.get(k) == v for k, v in {'a':10,'b':3,'c':30,'d':20}.items())",
    "ast2 = {'body':[{'op':'SET','reg':'f','val':1},",
    "  {'op':'REPEAT','n':3,'body':[{'op':'MUL','dst':'f','a':'f','b':2}]},",
    "  {'op':'ADD','dst':'g','a':'f','b':'x'}]}",
    "out2 = optimize_and_execute(ast2, {'x':5})",
    "assert all(out2.get(k) == v for k, v in {'f':8,'g':13}.items())",
    "ast3 = {'body':[{'op':'SET','reg':'n','val':2},{'op':'SET','reg':'acc','val':1},",
    "  {'op':'REPEAT','n':'n','body':[{'op':'REPEAT','n':2,'body':[{'op':'ADD','dst':'acc','a':'acc','b':1}]}]},",
    "  {'op':'IF','reg':'acc','gt':3,'body':[{'op':'MUL','dst':'acc','a':'acc','b':2}]}]}",
    "out3 = optimize_and_execute(ast3, {})",
    "assert out3.get('acc') == 10",
]

def build_problems(p3_variant):
    p3 = P3_ASSERTS_REF14
    return [
        ("p1_dijkstra_parity", P1_PROMPT, P1_ASSERTS),
        ("p2_microstrip", P2_PROMPT, P2_ASSERTS),
        ("p3_tx_engine", P3_PROMPT, p3),
        ("p4_pdn", P4_PROMPT, P4_ASSERTS),
        ("p5_ast", P5_PROMPT, P5_ASSERTS),
    ]

def ask(base, model, messages, mode, max_tokens, http_timeout=600, presence=0.0):
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if presence:
        payload["presence_penalty"] = presence
    if mode == "think":
        payload.update({"temperature": 0.6, "top_p": 0.95, "min_p": 0.0,
                        "chat_template_kwargs": {"thinking": True}})
    else:
        payload.update({"temperature": 0.1,
                        "chat_template_kwargs": {"thinking": False}})
    body = json.dumps(payload).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=http_timeout) as r:
        j = json.load(r)
    dt = time.time() - t0
    msg = j["choices"][0].get("message", {})
    usage = j.get("usage", {})
    finish = j["choices"][0].get("finish_reason", "?")
    return (msg.get("content") or "", msg.get("reasoning_content") or "",
            dt, usage.get("completion_tokens", 0), finish)

THINK_MARKERS = re.compile(
    r"(That's it|I'll wrap|I will output|Let's double check|Let's verify|"
    r"Self-Correction|The user wants|No extra text|Done\.|I need to|Proceed\.)")

def _compilable(block):
    compile(block, "<extract>", "exec")
    return True

def extract_code(text):
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.S)
    good = []
    for b in blocks:
        if "def " in b and not THINK_MARKERS.search(b):
            try:
                _compilable(b)
                good.append(b)
            except SyntaxError:
                pass
    if good:
        return max(good, key=len)
    if blocks:
        return max(blocks, key=len)
    lines = []
    for ln in text.splitlines():
        if ln.strip().startswith(("def ", "import ", "from ")):
            lines.append(ln)
        elif lines:
            lines.append(ln)
    return "\n".join(lines)

def run_tests(code, asserts):
    script = code + "\n" + "\n".join(asserts) + "\nprint('ALL_OK')\n"
    import tempfile
    tmp = None
    try:
        # Windows: 'python -c <script>' hits the ~32k command-line limit (WinError 206)
        # on large code -> write the script to a temp file instead.
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                         encoding="utf-8") as f:
            f.write(script)
            tmp = f.name
        p = subprocess.run([sys.executable, "-I", tmp],
                           capture_output=True, text=True, timeout=30)
        ok = p.returncode == 0 and "ALL_OK" in p.stdout
        err = (p.stderr or "")[-1500:]
        return ok, err
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT (30s) - likely infinite loop"
    finally:
        if tmp:
            try: os.unlink(tmp)
            except OSError: pass

CHARS_PER_TOKEN = 3.5  # estimate; labeled as such in the report

def attempt_metrics(content, reasoning, dt, ctok):
    """think vs code metrics. tg total is exact (completion_tokens/dt);
    the thinking/code split is an ESTIMATE from char count (APIs don't expose per-phase times)."""
    rchars = len(reasoning)
    r_tok_est = round(rchars / CHARS_PER_TOKEN)
    c_tok_est = max(0, ctok - r_tok_est)
    tg = round(ctok / dt, 1) if dt > 0 and ctok else 0
    return {"t_s": round(dt, 1), "completion_tokens": ctok,
            "reasoning_chars": rchars, "tok_gandire_est": r_tok_est,
            "tok_cod_est": c_tok_est, "tg_total": tg}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--setup", required=True)
    ap.add_argument("--mode", choices=["nothink", "think"], default="nothink")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--http-timeout", type=int, default=600)
    ap.add_argument("--max-repairs", type=int, default=4, help="pass@5 = 1 + 4 repairs; 0 = single step (pass@1)")
    ap.add_argument("--problems", default="", help="e.g. 'p3' or 'p1,p3' — run only those")
    ap.add_argument("--thinking-cap", action="store_true",
                    help="system prompt with a thinking ceiling")
    ap.add_argument("--presence", type=float, default=0.0, help="presence_penalty (e.g. 0.15)")
    ap.add_argument("--suite", default="full", choices=["p", "e", "full"],
                    help="p = P1-P5, e = E1-E5, full = all 10")
    a = ap.parse_args()
    PROBLEMS = build_problems("ref14")
    if a.suite in ("e", "full"):
        import new_problems
        if a.suite == "full":
            PROBLEMS = PROBLEMS + new_problems.build_new_problems()
        else:
            PROBLEMS = new_problems.build_new_problems()
    if a.problems:
        sel = a.problems.split(",")
        PROBLEMS = [p for p in PROBLEMS if any(p[0].startswith(s) for s in sel)]
    if a.thinking_cap:
        THINKING_CAP = (" CRITICAL: Keep your reasoning concise and architectural "
                        "(maximum 4,000 words). Do NOT dry-run the entire test array "
                        "step-by-step inside thinking. Design the data structures and "
                        "edge-cases quickly, then write the complete Python code.")
    base = f"http://127.0.0.1:{a.port}/v1"
    out = {"setup": a.setup, "model": a.model, "mode": a.mode,
           "hyperparams": {"max_tokens": a.max_tokens,
                           "think": {"temperature": 0.6, "top_p": 0.95, "min_p": 0.0},
                           "nothink": {"temperature": 0.1}},
           "date": time.strftime("%Y-%m-%d %H:%M"),
           "n": len(PROBLEMS), "pass1": 0, "pass5": 0, "detail": {}}
    t0 = time.time()
    for name, prompt, asserts in PROBLEMS:
        attempts = []
        msgs = [{"role": "system", "content": "You are a Python expert. Complete the function. Reply ONLY with the code inside one ```python block. No explanations." + (THINKING_CAP if a.thinking_cap else "")},
                {"role": "user", "content": prompt}]
        content = ""
        try:
            for k in range(a.max_repairs + 1):
                content, reasoning, dt, ctok, finish = ask(base, a.model, msgs, a.mode,
                                                           a.max_tokens, a.http_timeout,
                                                           presence=a.presence)
                code = extract_code(content) or extract_code(reasoning)
                ok, err = run_tests(code, asserts)
                rec = attempt_metrics(content, reasoning, dt, ctok)
                rec["finish"] = finish
                rec.update({"attempt": k + 1, "ok": ok, "code": code[:400]})
                if not ok:
                    rec["err"] = err[-800:]
                attempts.append(rec)
                if ok:
                    break
                # self-repair: if the response was TRUNCATED (thinking ate the whole
                # budget), the repair message is about truncation, NOT the traceback
                if finish == "length":
                    repair = ("Your previous response was cut off because you spent too many "
                              "tokens inside thinking. Output the complete Python function "
                              "immediately with minimal thinking (under 1,000 tokens). "
                              "Reply ONLY with the complete code in one ```python block.")
                else:
                    # normal self-repair: the EXACT error (AssertionError + traceback)
                    extra = ""
                    if name == "p3_tx_engine":
                        extra = (" Analyze your logical mistake, especially regarding deadlock "
                                 "cycle detection and rollback FIFO queues, and rewrite the function.")
                    repair = ("Your code failed with this error:\n```\n" + err +
                              "\n```" + extra +
                              " Reply ONLY with the corrected FULL code in one ```python block.")
                msgs = msgs + [{"role": "assistant", "content": content or code},
                               {"role": "user", "content": repair}]
        except Exception as e:
            attempts.append({"error": str(e)[:300]})
        ok_chain = [bool(x.get("ok")) for x in attempts]
        out["pass1"] += bool(ok_chain and ok_chain[0])
        out["pass5"] += bool(any(ok_chain))
        out["detail"][name] = {"attempts": attempts,
                               "n_attempts": len(ok_chain),
                               "pass_at": next((i + 1 for i, v in enumerate(ok_chain) if v), None)}
        first = attempts[0] if attempts else {}
        print(f"{name}: pass@1={'PASS' if ok_chain and ok_chain[0] else 'fail'} "
              f"pass@{len(ok_chain)}={'PASS' if any(ok_chain) else 'fail'} "
              f"(tg={first.get('tg_total', 0)} t/s, {first.get('t_s', 0)}s)", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    out["score"] = f"{out['pass1']}/{len(PROBLEMS)} pass@1, {out['pass5']}/{len(PROBLEMS)} pass@5"
    path = os.path.join(RESULTS, f"{a.setup}_{a.mode}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"SCORE: {out['score']} in {out['seconds']}s -> {path}")

if __name__ == "__main__":
    main()
