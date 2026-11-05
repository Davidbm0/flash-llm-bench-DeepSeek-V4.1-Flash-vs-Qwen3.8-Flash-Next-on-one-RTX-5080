# new_problems.py — suite E1-E5 (extreme_engineering_5_hardest_evaluation_suite.md)
# Ground-truth VERIFICAT 05.11 prin ref_extreme5.py (5/5 reproduci).
# accuracy_5v6.py -> acc_eval.py le include automat (build_problems inlocuit).

E1_PROMPT = """Write a Python function `analyze_coupled_microstrip(L11_nh: float, L12_nh: float, C11_pf: float, C12_pf: float, length_m: float, freq_hz: float, z0_ohms: float) -> dict` that computes the modal parameters and exact frequency-domain 4-port scattering parameters (including Near-End and Far-End Crosstalk) of a symmetric coupled microstrip pair. Use ONLY the Python standard library (math, cmath).

Physical & Mathematical Rules:
1. Input Matrices (per unit length):
   - Self inductance L11 = L11_nh * 1e-9 H/m, mutual inductance L12 = L12_nh * 1e-9 H/m.
   - Maxwell self capacitance C11 = C11_pf * 1e-12 F/m, Maxwell mutual capacitance C12 = C12_pf * 1e-12 F/m (note C12 < 0 in Maxwell capacitance form).
2. Odd and Even Mode Decomposition:
   - Odd Mode (antisymmetric V1 = -V2): L_odd = L11 - L12, C_odd = C11 - C12; Z0_odd = sqrt(L_odd/C_odd), tau_odd = sqrt(L_odd*C_odd), Z_diff = 2*Z0_odd.
   - Even Mode (symmetric V1 = +V2): L_even = L11 + L12, C_even = C11 + C12; Z0_even = sqrt(L_even/C_even), tau_even = sqrt(L_even*C_even), Z_comm = Z0_even/2.
3. Modal 2-Port S-Parameters at frequency f = freq_hz:
   For each mode m in {even, odd} with characteristic impedance Z_m, delay tau_m, electrical length theta_m = 2*pi*f*tau_m*length_m, and port reference impedance Z0 = z0_ohms:
   D_m = 2*cos(theta_m) + j*(Z_m/Z0 + Z0/Z_m)*sin(theta_m)
   S11_m = j*(Z_m/Z0 - Z0/Z_m)*sin(theta_m) / D_m ; S21_m = 2 / D_m
4. Single-Ended 4-Port Superposition (Port 1 = Aggressor Near-End, Port 2 = Aggressor Far-End, Port 3 = Victim Near-End NEXT, Port 4 = Victim Far-End FEXT):
   S11 = (S11_even + S11_odd)/2 ; S21 = (S21_even + S21_odd)/2
   S31 = (S11_even - S11_odd)/2 (NEXT) ; S41 = (S21_even - S21_odd)/2 (FEXT)

Return a dictionary with values rounded to 4 decimal places:
'z0_odd', 'z0_even', 'z_diff', 'z_comm' (ohms); 'tau_odd_ns_per_m', 'tau_even_ns_per_m' (ns/m);
'next_mag' = |S31|, 'next_db' = 20*log10(|S31|); 'fext_mag' = |S41|, 'fext_db' = 20*log10(|S41|)."""
E1_ASSERTS = [
    "res = analyze_coupled_microstrip(L11_nh=350.0, L12_nh=100.0, C11_pf=90.0, C12_pf=-10.0, length_m=0.125, freq_hz=1.0e9, z0_ohms=50.0)",
    "assert abs(res['z0_odd'] - 50.0) < 1e-4",
    "assert abs(res['z0_even'] - 75.0) < 1e-4",
    "assert abs(res['z_diff'] - 100.0) < 1e-4",
    "assert abs(res['z_comm'] - 37.5) < 1e-4",
    "assert abs(res['tau_odd_ns_per_m'] - 5.0) < 1e-4",
    "assert abs(res['tau_even_ns_per_m'] - 6.0) < 1e-4",
    "assert abs(res['next_mag'] - 0.1923) < 1e-3",
    "assert abs(res['next_db'] - (-14.3201)) < 1e-2",
    "assert abs(res['fext_mag'] - 0.3697) < 1e-3",
    "assert abs(res['fext_db'] - (-8.6436)) < 1e-2",
]

E2_PROMPT = """Write a Python function `solve_2d_microstrip_fdm(nx: int, ny: int, j_sub: int, strip_i_list: list, er_sub: float) -> dict` that solves the 2D electrostatic Laplace equation div(eps_r(x,y) grad V) = 0 on a discrete grid nx x ny (0 <= i < nx, 0 <= j < ny, square cell dx = dy = 1) to extract per-unit-length capacitance, effective permittivity and characteristic impedance. Use ONLY the Python standard library (math).

Rules:
1. Geometry & Dirichlet boundaries:
   - Outer PEC shield: V(i,j) = 0.0 on all boundary nodes (i=0, i=nx-1, j=0, j=ny-1).
   - Signal trace conductor: V(i, j_sub) = 1.0 for all i in strip_i_list.
   - Substrate below j = j_sub (0 <= j < j_sub) with eps_below = er_sub; air above (j_sub < j < ny) with eps_above = 1.0.
2. Finite-difference stencil (iterate to convergence < 1e-11):
   - Free interior node (i,j) with j != j_sub: V(i,j) = (V(i-1,j)+V(i+1,j)+V(i,j-1)+V(i,j+1))/4.
   - Free node on dielectric interface (i, j_sub), i not in strip_i_list:
     V(i,j_sub) = ( V(i-1,j_sub) + V(i+1,j_sub) + (2*eps_below/(eps_above+eps_below))*V(i,j_sub-1) + (2*eps_above/(eps_above+eps_below))*V(i,j_sub+1) ) / 4
3. Discrete Gauss's law charge & capacitance:
   For each conductor node (i_s, j_sub), i_s in strip_i_list:
   q(i_s) = eps_below*(1 - V(i_s,j_sub-1)) + eps_above*(1 - V(i_s,j_sub+1)) + ((eps_below+eps_above)/2)*(2 - V(i_s-1,j_sub) - V(i_s+1,j_sub))
   C_norm = sum of q over conductor nodes.
   Repeat the whole solve with er_sub = 1.0 to get C0_norm.
   eeff = C_norm / C0_norm ; Z0 = 376.730313668 / sqrt(C_norm * C0_norm).

Return dict rounded to 4 decimals: 'c_norm', 'c0_norm', 'eeff', 'z0_ohms'."""
E2_ASSERTS = [
    "res = solve_2d_microstrip_fdm(nx=5, ny=4, j_sub=1, strip_i_list=[2], er_sub=4.0)",
    "assert abs(res['c_norm'] - (142.0/17.0)) < 1e-3",
    "assert abs(res['c0_norm'] - (161.0/52.0)) < 1e-3",
    "assert abs(res['eeff'] - (7384.0/2737.0)) < 1e-3",
    "assert abs(res['z0_ohms'] - 74.0802) < 1e-2",
]

E3_PROMPT = """Write a Python function `simulate_nonlinear_diode_rc(v_in_dc: float, v_in_step: float, r1_ohms: float, r2_ohms: float, c2_farads: float, dt_sec: float) -> dict` computing the exact DC operating point at t=0 for input v_in_dc and one non-linear Backward Euler transient step at t=dt_sec when the input steps to v_in_step. Use ONLY the Python standard library (math).

Circuit topology & equations:
- Node 1: connected to input voltage source Vin(t) via R1 = r1_ohms, and to the anode of Shockley diode D1.
- Node 2: connected to cathode of D1, load resistor R2 = r2_ohms to GND, filter capacitor C2 = c2_farads to GND.
- Shockley diode: ID(V1,V2) = Is*(exp((V1-V2)/VT) - 1) with Is = 1e-12 A, VT = 0.025 V.
Step 1 - DC operating point at t=0 (C2 open circuit, Vin(0) = v_in_dc): solve with Newton-Raphson (tolerance < 1e-12):
  (V1(0)-Vin(0))/R1 + ID(V1(0),V2(0)) = 0 ; V2(0)/R2 - ID(V1(0),V2(0)) = 0
Step 2 - Backward Euler step at t=dt (Vin(dt) = v_in_step): capacitor replaced by companion model (Gc = C2/dt in parallel with history current source Ihist = Gc*V2(0)):
  (V1(dt)-Vin(dt))/R1 + ID(V1(dt),V2(dt)) = 0
  V2(dt)/R2 + (C2/dt)*(V2(dt)-V2(0)) - ID(V1(dt),V2(dt)) = 0

Return dict rounded to 5 decimals: 'v1_dc', 'v2_dc', 'id_dc_ma' (mA at t=0), 'v1_step', 'v2_step', 'id_step_ma' (mA at t=dt)."""
E3_ASSERTS = [
    "res = simulate_nonlinear_diode_rc(v_in_dc=5.0, v_in_step=8.0, r1_ohms=1000.0, r2_ohms=2000.0, c2_farads=1.0e-7, dt_sec=1.0e-4)",
    "assert abs(res['id_dc_ma'] - 1.49065) < 1e-4",
    "assert abs(res['v2_dc'] - 2.98129) < 1e-4",
    "assert abs(res['v1_dc'] - 3.50935) < 1e-4",
    "assert abs(res['id_step_ma'] - 3.27882) < 1e-4",
    "assert abs(res['v2_step'] - 4.17341) < 1e-4",
    "assert abs(res['v1_step'] - 4.72118) < 1e-4",
]

E4_PROMPT = """Write a Python function `analyze_serdes_eye(t_ui_ps: float, num_samples: int, dfe_taps: int) -> dict` computing the single-bit pulse response of a 2-pole overdamped SerDes interconnect, the un-equalized worst-case vertical eye opening, and a 3-tap FFE + dfe_taps-tap DFE equalized eye. Use ONLY the Python standard library (math).

Rules:
1. Channel step response, two real poles tau1 = T_UI/ln(2), tau2 = T_UI/ln(4):
   s(t) = 0 for t <= 0 ; s(t) = 1 - (tau1*exp(-t/tau1) - tau2*exp(-t/tau2))/(tau1 - tau2) for t > 0.
2. Baud-rate sampled pulse response, m = 1..num_samples: p[m] = s(m*T_UI) - s((m-1)*T_UI).
   m_main = index m maximizing p[m] (main cursor h0 = p[m_main]).
3. Worst-case vertical eye height (NRZ bipolar b_k in {-1,+1}) for any sequence h[m], m=1..M, main cursor at m_main:
   ISI_pre = sum |h[m]| for m < m_main ; ISI_post_rem = sum |h[m]| for m > m_main + K where K = dfe_taps (first K post-cursors cancelled);
   EH = max(0.0, 2.0*(h[m_main] - ISI_pre - ISI_post_rem)).
4. 3-tap pole-cancelling FFE: causal FIR w = [w0,w1,w2] normalized to unit peak-power (|w0|+|w1|+|w2| = 1.0) with pole-zero cancellation ratios 1 : -(exp(-T_UI/tau1)+exp(-T_UI/tau2)) : exp(-T_UI/tau1 - T_UI/tau2). Equalized response h_eq[m] = sum_{k=0..2} w_k * p[m-k] with p[idx]=0 for idx <= 0. Evaluate its eye with main cursor at m=1 and dfe_taps post-cursors cancelled.

Return dict rounded to 5 decimals: 'p1','p2','p3' (raw samples m=1,2,3), 'raw_eye_no_dfe' (eye of p with 0 DFE taps), 'raw_eye_with_dfe' (eye of p with dfe_taps DFE taps), 'ffe_dfe_eye' (eye of h_eq, cursor m=1, dfe_taps DFE taps)."""
E4_ASSERTS = [
    "res = analyze_serdes_eye(t_ui_ps=100.0, num_samples=25, dfe_taps=1)",
    "assert abs(res['p1'] - 0.25) < 1e-5",
    "assert abs(res['p2'] - 0.3125) < 1e-5",
    "assert abs(res['p3'] - 0.203125) < 1e-4",
    "assert abs(res['raw_eye_no_dfe'] - 0.0) < 1e-5",
    "assert abs(res['raw_eye_with_dfe'] - 0.0) < 1e-5",
    "assert abs(res['ffe_dfe_eye'] - (4.0/15.0)) < 1e-4",
]

E5_PROMPT = """Write a Python function `synthesize_rf_conjugate_match(s11: complex, s12: complex, s21: complex, s22: complex, z0: float) -> dict` performing full bilateral RF amplifier stability analysis and simultaneous conjugate match synthesis. Use ONLY the Python standard library (math, cmath).

Rules:
1. Delta = S11*S22 - S12*S21 ; K = (1 - |S11|^2 - |S22|^2 + |Delta|^2) / (2*|S12*S21|).
2. mu1 = (1 - |S11|^2) / (|S22 - Delta*conj(S11)| + |S12*S21|).
3. Simultaneous conjugate match:
   B1 = 1 + |S11|^2 - |S22|^2 - |Delta|^2 ; C1 = S11 - Delta*conj(S22)
   B2 = 1 + |S22|^2 - |S11|^2 - |Delta|^2 ; C2 = S22 - Delta*conj(S11)
   Gamma_MS = (B1 - sign(B1)*sqrt(B1^2 - 4|C1|^2)) / (2*C1)
   Gamma_ML = (B2 - sign(B2)*sqrt(B2^2 - 4|C2|^2)) / (2*C2)
4. ZS = z0*(1+Gamma_MS)/(1-Gamma_MS) ; ZL = z0*(1+Gamma_ML)/(1-Gamma_ML)
   MAG_linear = (|S21|/|S12|)*(K - sqrt(K^2-1)) ; MAG_db = 10*log10(MAG_linear).

Return dict rounded to 4 decimals: 'delta_mag', 'k_factor', 'mu1_factor', 'gamma_ms_mag', 'gamma_ml_mag', 'zs_real', 'zs_imag', 'zl_real', 'zl_imag', 'mag_db'."""
E5_ASSERTS = [
    "res = synthesize_rf_conjugate_match(s11=0.6+0.0j, s12=0.0+0.1j, s21=0.0+2.0j, s22=0.5+0.0j, z0=50.0)",
    "assert abs(res['delta_mag'] - 0.5000) < 1e-4",
    "assert abs(res['k_factor'] - 1.6000) < 1e-4",
    "assert abs(res['mu1_factor'] - 1.6000) < 1e-4",
    "assert abs(res['gamma_ms_mag'] - 0.5149) < 1e-3",
    "assert abs(res['gamma_ml_mag'] - 0.3510) < 1e-3",
    "assert abs(res['zs_real'] - 156.1249) < 1e-2",
    "assert abs(res['zl_real'] - 104.0833) < 1e-2",
    "assert abs(res['mag_db'] - 8.4634) < 1e-3",
]

SUITE_NEW = [
    ("e1_coupled_mtl", E1_PROMPT, E1_ASSERTS),
    ("e2_fdm_laplace", E2_PROMPT, E2_ASSERTS),
    ("e3_spice_diode_rc", E3_PROMPT, E3_ASSERTS),
    ("e4_serdes_eye", E4_PROMPT, E4_ASSERTS),
    ("e5_rf_conj_match", E5_PROMPT, E5_ASSERTS),
]

def build_new_problems():
    return list(SUITE_NEW)
