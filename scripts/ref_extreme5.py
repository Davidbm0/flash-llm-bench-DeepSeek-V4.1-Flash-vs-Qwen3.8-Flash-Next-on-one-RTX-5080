# ref_extreme5.py — implementari de REFERINTA pentru suite-ul E1-E5.
# Scop: verific ca asserturile din document sunt corecte (ground-truth).
# NU se da niciodata modelului. Ruleaza: python3 ref_extreme5.py
import math, cmath

# ---------------- E1 ----------------
def analyze_coupled_microstrip(L11_nh, L12_nh, C11_pf, C12_pf, length_m, freq_hz, z0_ohms):
    L11 = L11_nh * 1e-9; L12 = L12_nh * 1e-9
    C11 = C11_pf * 1e-12; C12 = C12_pf * 1e-12
    Lo, Co = L11 - L12, C11 - C12
    Le, Ce = L11 + L12, C11 + C12
    Zo, tau_o = math.sqrt(Lo / Co), math.sqrt(Lo * Co)
    Ze, tau_e = math.sqrt(Le / Ce), math.sqrt(Le * Ce)

    def modal(Zm, taum):
        th = 2 * math.pi * freq_hz * taum * length_m
        D = 2 * math.cos(th) + 1j * (Zm / z0_ohms + z0_ohms / Zm) * math.sin(th)
        S11 = 1j * (Zm / z0_ohms - z0_ohms / Zm) * math.sin(th) / D
        S21 = 2 / D
        return S11, S21

    S11e, S21e = modal(Ze, tau_e)
    S11o, S21o = modal(Zo, tau_o)
    S31 = (S11e - S11o) / 2          # NEXT
    S41 = (S21e - S21o) / 2          # FEXT
    return {
        "z0_odd": round(Zo, 4), "z0_even": round(Ze, 4),
        "z_diff": round(2 * Zo, 4), "z_comm": round(Ze / 2, 4),
        "tau_odd_ns_per_m": round(tau_o * 1e9, 4), "tau_even_ns_per_m": round(tau_e * 1e9, 4),
        "next_mag": round(abs(S31), 4), "next_db": round(20 * math.log10(abs(S31)), 4),
        "fext_mag": round(abs(S41), 4), "fext_db": round(20 * math.log10(abs(S41)), 4),
    }

# ---------------- E2 ----------------
def _fdm_solve(nx, ny, j_sub, strip, er_sub):
    er_below, er_above = er_sub, 1.0
    V = [[0.0] * ny for _ in range(nx)]
    fixed = set()
    for i in range(nx):
        for j in range(ny):
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                fixed.add((i, j))
    for i in strip:
        V[i][j_sub] = 1.0
        fixed.add((i, j_sub))
    w_below = 2 * er_below / (er_above + er_below)
    w_above = 2 * er_above / (er_above + er_below)
    free = [(i, j) for i in range(nx) for j in range(ny) if (i, j) not in fixed]
    for _ in range(2_000_000):
        delta = 0.0
        for (i, j) in free:
            if j == j_sub:
                nv = 0.25 * (V[i-1][j] + V[i+1][j]
                             + w_below * V[i][j-1] + w_above * V[i][j+1])
            else:
                nv = 0.25 * (V[i-1][j] + V[i+1][j] + V[i][j-1] + V[i][j+1])
            delta = max(delta, abs(nv - V[i][j]))
            V[i][j] = nv
        if delta < 1e-11:
            break
    q = 0.0
    for i in strip:
        q += (er_below * (1 - V[i][j_sub-1])
              + er_above * (1 - V[i][j_sub+1])
              + (er_below + er_above) / 2 * (2 - V[i-1][j_sub] - V[i+1][j_sub]))
    return q

def solve_2d_microstrip_fdm(nx, ny, j_sub, strip_i_list, er_sub):
    cn = _fdm_solve(nx, ny, j_sub, strip_i_list, er_sub)
    c0 = _fdm_solve(nx, ny, j_sub, strip_i_list, 1.0)
    eeff = cn / c0
    z0 = 376.730313668 / math.sqrt(cn * c0)
    return {"c_norm": round(cn, 4), "c0_norm": round(c0, 4),
            "eeff": round(eeff, 4), "z0_ohms": round(z0, 4)}

# ---------------- E3 ----------------
def simulate_nonlinear_diode_rc(v_in_dc, v_in_step, r1_ohms, r2_ohms, c2_farads, dt_sec):
    Is, VT = 1e-12, 0.025
    def id_(v1, v2):
        x = (v1 - v2) / VT
        if x > 700: x = 700.0
        return Is * (math.exp(x) - 1.0)

    def solve(vin, v2_prev, gc):
        # f(v2) monotonic decreasing in v2 => bisection sigura
        def f(v2):
            # din ec node1: v1 = vin - R1*ID(v1,v2) -> rezolvam v1 prin bisection si el
            lo, hi = v2, vin + 1.0
            for _ in range(200):
                mid = (lo + hi) / 2
                # la fix v2, ID creste cu v1; ec1: (v1-vin)/R1 + ID(v1,v2) = 0 monotonic cresc in v1
                g1 = (mid - vin) / r1_ohms + id_(mid, v2)
                if g1 > 0: hi = mid
                else: lo = mid
            v1 = (lo + hi) / 2
            iD = id_(v1, v2)
            return v2 / r2_ohms + gc * (v2 - v2_prev) - iD
        lo, hi = 0.0, vin + 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if f(mid) > 0: hi = mid   # f crescatoare: mid pozitiv => hi
            else: lo = mid
        v2 = (lo + hi) / 2
        lo2, hi2 = v2, vin + 1.0
        for _ in range(200):
            mid = (lo2 + hi2) / 2
            if (mid - vin) / r1_ohms + id_(mid, v2) > 0: hi2 = mid
            else: lo2 = mid
        v1 = (lo2 + hi2) / 2
        return v1, v2, id_(v1, v2)

    v1d, v2d, idd = solve(v_in_dc, 0.0, 0.0)
    gc = c2_farads / dt_sec
    v1s, v2s, ids = solve(v_in_step, v2d, gc)
    return {"v1_dc": round(v1d, 5), "v2_dc": round(v2d, 5), "id_dc_ma": round(idd * 1e3, 5),
            "v1_step": round(v1s, 5), "v2_step": round(v2s, 5), "id_step_ma": round(ids * 1e3, 5)}

# ---------------- E4 ----------------
def analyze_serdes_eye(t_ui_ps, num_samples, dfe_taps):
    T = t_ui_ps
    t1, t2 = T / math.log(2), T / math.log(4)
    def s(t):
        if t <= 0: return 0.0
        return 1 - (t1 * math.exp(-t / t1) - t2 * math.exp(-t / t2)) / (t1 - t2)
    p = [s(m * T) - s((m - 1) * T) for m in range(1, num_samples + 1)]
    m_main = max(range(len(p)), key=lambda i: p[i])

    def eye(h, main, K):
        pre = sum(abs(h[i]) for i in range(main))
        post = sum(abs(h[i]) for i in range(main + 1 + K, len(h)))
        return max(0.0, 2.0 * (h[main] - pre - post))

    a1, a2 = math.exp(-T / t1), math.exp(-T / t2)
    w = [1.0, -(a1 + a2), a1 * a2]
    sw = sum(abs(x) for x in w)
    w = [x / sw for x in w]
    heq = []
    for m in range(1, num_samples + 1):
        acc = 0.0
        for k in range(3):
            idx = m - k              # p idx 1-based; p[idx]=0 pt idx<=0
            if idx >= 1:
                acc += w[k] * p[idx - 1]
        heq.append(acc)
    return {"p1": round(p[0], 5), "p2": round(p[1], 5), "p3": round(p[2], 5),
            "raw_eye_no_dfe": round(eye(p, m_main, 0), 5),
            "raw_eye_with_dfe": round(eye(p, m_main, dfe_taps), 5),
            "ffe_dfe_eye": round(eye(heq, 0, dfe_taps), 5)}

# ---------------- E5 ----------------
def synthesize_rf_conjugate_match(s11, s12, s21, s22, z0):
    s11 = complex(s11); s12 = complex(s12); s21 = complex(s21); s22 = complex(s22)
    delta = s11 * s22 - s12 * s21
    K = (1 - abs(s11)**2 - abs(s22)**2 + abs(delta)**2) / (2 * abs(s12 * s21))
    mu1 = (1 - abs(s11)**2) / (abs(s22 - delta * s11.conjugate()) + abs(s12 * s21))
    B1 = 1 + abs(s11)**2 - abs(s22)**2 - abs(delta)**2
    C1 = s11 - delta * s22.conjugate()
    B2 = 1 + abs(s22)**2 - abs(s11)**2 - abs(delta)**2
    C2 = s22 - delta * s11.conjugate()
    gms = (B1 - math.copysign(1, B1) * math.sqrt(B1**2 - 4 * abs(C1)**2)) / (2 * C1)
    gml = (B2 - math.copysign(1, B2) * math.sqrt(B2**2 - 4 * abs(C2)**2)) / (2 * C2)
    ZS = z0 * (1 + gms) / (1 - gms)
    ZL = z0 * (1 + gml) / (1 - gml)
    mag = abs(s21) / abs(s12) * (K - math.sqrt(K**2 - 1))
    return {"delta_mag": round(abs(delta), 4), "k_factor": round(K, 4),
            "mu1_factor": round(mu1, 4),
            "gamma_ms_mag": round(abs(gms), 4), "gamma_ml_mag": round(abs(gml), 4),
            "zs_real": round(ZS.real, 4), "zs_imag": round(ZS.imag, 4),
            "zl_real": round(ZL.real, 4), "zl_imag": round(ZL.imag, 4),
            "mag_db": round(10 * math.log10(mag), 4)}

# ---------------- test ----------------
if __name__ == "__main__":
    fails = []
    def check(name, cond):
        print(f"  {name}: {'OK' if cond else 'FAIL'}")
        if not cond: fails.append(name)

    r = analyze_coupled_microstrip(350.0, 100.0, 90.0, -10.0, 0.125, 1.0e9, 50.0)
    print("E1:", r)
    check("e1 z0_odd", abs(r["z0_odd"] - 50.0) < 1e-4)
    check("e1 z0_even", abs(r["z0_even"] - 75.0) < 1e-4)
    check("e1 z_diff", abs(r["z_diff"] - 100.0) < 1e-4)
    check("e1 z_comm", abs(r["z_comm"] - 37.5) < 1e-4)
    check("e1 tau_o", abs(r["tau_odd_ns_per_m"] - 5.0) < 1e-4)
    check("e1 tau_e", abs(r["tau_even_ns_per_m"] - 6.0) < 1e-4)
    check("e1 next_mag", abs(r["next_mag"] - 0.1923) < 1e-3)
    check("e1 next_db", abs(r["next_db"] - (-14.3201)) < 1e-2)
    check("e1 fext_mag", abs(r["fext_mag"] - 0.3697) < 1e-3)
    check("e1 fext_db", abs(r["fext_db"] - (-8.6436)) < 1e-2)

    r = solve_2d_microstrip_fdm(5, 4, 1, [2], 4.0)
    print("E2:", r)
    check("e2 c_norm", abs(r["c_norm"] - 142/17) < 1e-3)
    check("e2 c0_norm", abs(r["c0_norm"] - 161/52) < 1e-3)
    check("e2 eeff", abs(r["eeff"] - 7384/2737) < 1e-3)
    check("e2 z0", abs(r["z0_ohms"] - 74.0802) < 1e-2)

    r = simulate_nonlinear_diode_rc(5.0, 8.0, 1000.0, 2000.0, 1.0e-7, 1.0e-4)
    print("E3:", r)
    check("e3 id_dc", abs(r["id_dc_ma"] - 1.49065) < 1e-4)
    check("e3 v2_dc", abs(r["v2_dc"] - 2.98129) < 1e-4)
    check("e3 v1_dc", abs(r["v1_dc"] - 3.50935) < 1e-4)
    check("e3 id_st", abs(r["id_step_ma"] - 3.27882) < 1e-4)
    check("e3 v2_st", abs(r["v2_step"] - 4.17341) < 1e-4)
    check("e3 v1_st", abs(r["v1_step"] - 4.72118) < 1e-4)

    r = analyze_serdes_eye(100.0, 25, 1)
    print("E4:", r)
    check("e4 p1", abs(r["p1"] - 0.25) < 1e-5)
    check("e4 p2", abs(r["p2"] - 0.3125) < 1e-5)
    check("e4 p3", abs(r["p3"] - 0.203125) < 1e-4)
    check("e4 eye0", abs(r["raw_eye_no_dfe"] - 0.0) < 1e-5)
    check("e4 eye_dfe", abs(r["raw_eye_with_dfe"] - 0.0) < 1e-5)
    check("e4 ffe", abs(r["ffe_dfe_eye"] - 4/15) < 1e-4)

    r = synthesize_rf_conjugate_match(0.6+0j, 0+0.1j, 0+2j, 0.5+0j, 50.0)
    print("E5:", r)
    check("e5 delta", abs(r["delta_mag"] - 0.5) < 1e-4)
    check("e5 K", abs(r["k_factor"] - 1.6) < 1e-4)
    check("e5 mu1", abs(r["mu1_factor"] - 1.6) < 1e-4)
    check("e5 gms", abs(r["gamma_ms_mag"] - 0.5149) < 1e-3)
    check("e5 gml", abs(r["gamma_ml_mag"] - 0.3510) < 1e-3)
    check("e5 zs", abs(r["zs_real"] - 156.1249) < 1e-2)
    check("e5 zl", abs(r["zl_real"] - 104.0833) < 1e-2)
    check("e5 mag", abs(r["mag_db"] - 8.4634) < 1e-3)

    print("\nVERDICT:", "5/5 ground-truth VERIFICAT" if not fails else f"FAILS: {fails}")
