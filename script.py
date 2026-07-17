"""
Numerical reproduction of Wu et al. (2016) Scientific Reports 6:28
"Model of electrical activity in cardiac tissue under electromagnetic induction"

3-variable FitzHugh-Nagumo model with magnetic flux (memristor coupling),
eq. (3) of the paper:
  du/dt  = -k*u*(u-a)*(u-1) - u*v + k0*rho(phi)*u  [+ Du*laplacian(u) in 2D]
  dv/dt  = eps(u,v) * (-v - k*u*(u-a-1))
  dphi/dt = k1*u - k2*phi                          [+ noise on phi if applicable]

where:
  eps(u,v) = eps0 + mu1*v / (u + mu2)   (state-dependent timescale)
  rho(phi) = alpha + 3*beta*phi^2        (memductance)

NOTE — the memristive term is +k0*rho(phi)*u
Usage:
  python script.py              → interactive menu
  python script.py 1            → generate fig 1 only
  python script.py 1 3 7        → generate figs 1, 3, 7
  python script.py all          → generate all figures
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import multiprocessing

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures_2.0')
os.makedirs(FIGDIR, exist_ok=True)

# ═══════════════════════════════════════════════
# PARAMETERS — edit these to iterate quickly
# ═══════════════════════════════════════════════
P = dict(
    # --- model constants ---
    a      = 0.15,     # excitability threshold
    eps0   = 0.002,    # base timescale ratio
    mu1    = 0.2,      # recovery modulation coefficient
    mu2    = 0.3,      # recovery denominator offset
    k      = 8.0,      # excitability scaling
    alpha  = 1.0,      # memristor baseline conductance
    beta   = 2.0,      # memristor nonlinearity
    Du     = 1.0,      # diffusion coefficient
    L      = 350.0,    # spatial domain side length

    # --- observation node (paper's 1-indexed (100,100) → 0-indexed (99,99)) ---
    obs_node      = (99, 99),

    # --- bifurcation reference node (figs 2 & 4): 1-indexed (125,75) → 0-indexed (124,74) ---
    bif_node      = (124, 74),

    # --- fig 1: time series ---
    fig1_k0_vals  = [0.1, 0.4, 0.9],
    fig1_k1       = 0.5,
    fig1_k2       = 1.0,
    fig1_t_end    = 200,

    # --- fig 2: bifurcation vs k0 ---
    fig2_k0_min   = 0.0,
    fig2_k0_max   = 1.2,
    fig2_k0_steps = 80,
    fig2_k1       = 0.5,
    fig2_k2       = 1.0,
    fig2_t_transient = 200,
    fig2_t_end    = 300,

    # --- fig 4: bifurcation vs k2 ---
    fig4_k2_min   = 0.0,
    fig4_k2_max   = 1.0,
    fig4_k2_steps = 80,
    fig4_k0       = 0.1,
    fig4_k1       = 0.5,
    fig4_t_end    = 300,
    fig4_t_transient = 200,

    # --- 2D common ---
    N_grid   = 200,
    dt_2d    = 0.03,
    k0_2d    = 0.1,
    k1_2d    = 0.5,

    # --- fig 3: spiral formation ---
    fig3_k2          = 1.0,
    fig3_snap_times  = [10, 60, 100, 200],

    # --- fig 7: spiral at low k2 ---
    fig7_k2          = 0.15,
    fig7_snap_times  = [10, 60, 100, 200],

    # --- fig 11: EM radiation spiral disruption ---
    fig11_k2          = 1.6,
    fig11_rad_cx      = 83,
    fig11_rad_cy      = 83,
    fig11_rad_A       = 12.0,
    fig11_rad_m       = 0.10,
    fig11_snap_times  = [10, 120, 600, 800],

    # --- fig 14: noise-driven breakup ---
    fig14_k2          = 1.6,
    fig14_warm_time   = 200,
    fig14_noise_cx    = 59,
    fig14_noise_cy    = 59,
    fig14_noise_r     = 40,
    fig14_noise_D     = 1.5,
    fig14_snap_times  = [10, 100, 200, 300, 500, 1000],
    fig14_seed        = 42,
    fig14_dt          = 0.03,


    # --- fig 16: spiral at medium k2 ---
    fig16_k2          = 0.5,
    fig16_snap_times  = [10, 60, 100, 200],

    # --- fig 17: spiral at high k2 ---
    fig17_k2          = 3.0,
    fig17_snap_times  = [10, 60, 100, 200],

    # --- fig 18: varying k1 comparison ---
    fig18_k1_vals     = [0.2, 1.45, 1.9],
    fig18_k2          = 1.5,
    fig18_snap_times  = [10, 60, 100, 200],

    # --- fig 19: varying k0 comparison (fig 2 conditions) ---
    fig19_k0_vals     = [0.1, 0.4, 0.9],
    fig19_k1          = 0.5,
    fig19_k2          = 1.6,
    fig19_snap_times  = [10, 60, 100, 200],
)


# ═══════════════════════════════════════════════
# PDE helpers
# ═══════════════════════════════════════════════

def laplacian2d(U, dx2):
    lap = np.zeros_like(U)
    # interior
    lap[1:-1, 1:-1] = (U[2:, 1:-1] + U[:-2, 1:-1] +
                       U[1:-1, 2:] + U[1:-1, :-2] - 4.0 * U[1:-1, 1:-1]) / dx2
    # edges: zero normal derivative (no-flux) via one-sided difference
    lap[0, 1:-1]  = (2*U[1, 1:-1]  + U[0, 2:]  + U[0, :-2]  - 4*U[0, 1:-1])  / dx2
    lap[-1, 1:-1] = (2*U[-2, 1:-1] + U[-1, 2:] + U[-1, :-2] - 4*U[-1, 1:-1]) / dx2
    lap[1:-1, 0]  = (U[2:, 0]  + U[:-2, 0]  + 2*U[1:-1, 1]  - 4*U[1:-1, 0])  / dx2
    lap[1:-1, -1] = (U[2:, -1] + U[:-2, -1] + 2*U[1:-1, -2] - 4*U[1:-1, -1]) / dx2
    # corners
    lap[0, 0]   = (2*U[1, 0]   + 2*U[0, 1]   - 4*U[0, 0])   / dx2
    lap[0, -1]  = (2*U[1, -1]  + 2*U[0, -2]  - 4*U[0, -1])  / dx2
    lap[-1, 0]  = (2*U[-2, 0]  + 2*U[-1, 1]  - 4*U[-1, 0])  / dx2
    lap[-1, -1] = (2*U[-2, -1] + 2*U[-1, -2] - 4*U[-1, -1]) / dx2
    return lap


def init_wedge(N):
    u = np.zeros((N, N))
    v = np.zeros((N, N))
    phi = np.zeros((N, N))
    # paper ICs (1-indexed 92:97 → 0-indexed 91:97, cols 1:115 → 0:115)
    u[91:97, 0:115]   = 1.0

    u[97:103, 0:115]   = 0.7
    v[97:103, 0:115]   = 0.6
    phi[97:103, 0:115]  = 0.1

    u[103:109, 0:115]   = 0.0
    v[103:109, 0:115]   = 0.8
    phi[103:109, 0:115] = 0.2
    return u, v, phi


def run_2d_sim(N, dx, dt, n_steps, k0, k1, k2,
               U0, V0, Phi0, noise_params=None,
               em_radiation=None,
               snapshot_steps=None, record_nodes=None,
               clamp_U=None, seed=None):
    U, V, Phi = U0.copy(), V0.copy(), Phi0.copy()
    snapshots = []
    dx2 = dx**2
    snap_set = set(snapshot_steps) if snapshot_steps else set()

    rng = np.random.default_rng(seed)
    noise_idx = None
    n_noise = 0
    D_noise = 0.0
    if noise_params is not None:
        cx, cy = noise_params['cx'], noise_params['cy']
        r = noise_params['r']
        D_noise = noise_params['D']
        xs = np.arange(N)
        xx, yy = np.meshgrid(xs, xs, indexing='ij')
        noise_mask = ((xx - cx)**2 + (yy - cy)**2 <= r**2)
        noise_idx = np.nonzero(noise_mask)
        n_noise = len(noise_idx[0])

    F_rad = None
    if em_radiation is not None:
        cx, cy = em_radiation['cx'], em_radiation['cy']
        A_rad = em_radiation['A']
        m_rad = em_radiation['m']
        xs = np.arange(N)
        xx, yy = np.meshgrid(xs, xs, indexing='ij')
        r_field = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        F_rad = A_rad * np.exp(-m_rad * r_field)

    node_traces = {c: [] for c in (record_nodes or [])}

    def _rhs(U, V, Phi):
        rho = P['alpha'] + 3.0 * P['beta'] * Phi**2
        # Guard the eps singularity: the denominator (u + mu2) blows up for
        # u < -mu2. Same protection as in bifurcation_2d_peaks.
        eps_field = P['eps0'] + P['mu1'] * V / np.maximum(U + P['mu2'], 1e-6)
        lap = laplacian2d(U, dx2)
        dU   = -P['k'] * U * (U - P['a']) * (U - 1.0) - U * V + k0 * rho * U + P['Du'] * lap
        dV   = eps_field * (-V - P['k'] * U * (U - P['a'] - 1.0))
        dPhi = k1 * U - k2 * Phi
        if F_rad is not None:
            dPhi += F_rad
        return dU, dV, dPhi

    for n in range(n_steps):
        dU, dV, dPhi = _rhs(U, V, Phi)
        U   += dt * dU
        V   += dt * dV
        Phi += dt * dPhi

        if clamp_U is not None:
            U = np.clip(U, clamp_U[0], clamp_U[1])

        if noise_idx is not None:
            # Euler-Maruyama: the stochastic increment scales as sqrt(dt), NOT dt.
            # The old `dt*sqrt(2D)` form gave an effective D_eff = D*dt, so the
            # noise vanished as dt->0 and never converged. This is the fix.
            Phi[noise_idx] += np.sqrt(2.0 * D_noise * dt) * rng.standard_normal(n_noise)

        if n in snap_set:
            snapshots.append((n, U.copy(), Phi.copy()))

        for c in (record_nodes or []):
            node_traces[c].append(U[c])

    return snapshots, (U, V, Phi), node_traces


def bifurcation_2d_peaks(k0, k1, k2, n_total, n_transient, N, dx, dt, node):
    """Run 2D PDE and return on-the-fly detected peaks at *node* (vectorised).
    A sample is recorded as a peak when it exceeds its post-adjacent sample
    and is >= its pre-adjacent sample. Returns None if the simulation diverges.
    """
    U, V, Phi = init_wedge(N)
    dx2 = dx ** 2
    prev2 = float(U[node])
    prev1 = float(U[node])
    peaks = []

    a = P['a']
    eps0 = P['eps0']
    mu1 = P['mu1']
    mu2 = P['mu2']
    k_param = P['k']
    alpha_val = P['alpha']
    beta_val = P['beta']
    Du = P['Du']

    with np.errstate(over='ignore', invalid='ignore'):
        for n in range(n_total):
            rho = alpha_val + 3.0 * beta_val * Phi ** 2
            eps_f = eps0 + mu1 * V / np.maximum(U + mu2, 1e-6)
            lap = laplacian2d(U, dx2)
            dU = -k_param * U * (U - a) * (U - 1.0) - U * V + k0 * rho * U + Du * lap
            dV = eps_f * (-V - k_param * U * (U - a - 1.0))
            dPhi = k1 * U - k2 * Phi
            U += dt * dU
            V += dt * dV
            Phi += dt * dPhi

            curr = float(U[node])
            if not np.isfinite(curr):
                return None

            if n > n_transient and prev1 >= prev2 and prev1 > curr:
                peaks.append(prev1)
            prev2 = prev1
            prev1 = curr

    return np.array(peaks) if peaks else np.array([prev1])


def _bifurcation_worker(args):
    """Unpack args tuple and call bifurcation_2d_peaks."""
    param_val, k0, k1, k2, n_total, n_transient, N, dx, dt, node = args
    peaks = bifurcation_2d_peaks(k0, k1, k2, n_total, n_transient, N, dx, dt, node)
    return param_val, peaks


# ═══════════════════════════════════════════════
# Figure functions
# ═══════════════════════════════════════════════

def make_fig_init():
    """Fig init — Initial condition (wedge perturbation) on the 200×200 grid at t=0."""
    print("  Generating Fig init (initial perturbation at t=0)...")
    N = P['N_grid']
    U0, V0, Phi0 = init_wedge(N)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    fields = [('u (membrane potential)', U0, 'jet'),
              ('v (recovery variable)', V0, 'viridis'),
              ('φ (magnetic flux)', Phi0, 'plasma')]
    for ax, (label, field, cmap) in zip(axes, fields):
        im = ax.imshow(field.T, cmap=cmap, origin='lower',
                       aspect='equal', extent=[0, N, 0, N])
        ax.set_title(label, fontsize=13)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle('Initial perturbation (wedge) at t = 0\n'
                 f'{N}×{N} grid, L={P["L"]}', fontsize=14)
    path = os.path.join(FIGDIR, 'fig_init_perturbation.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig init saved → {path}")


def make_fig1():
    """Fig 1 — Time series at node (100,100) for different k0 (2D PDE)."""
    print("  Generating Fig 1 (time series from 2D sim, varying k0)...")
    k0_vals = P['fig1_k0_vals']
    t_end = P['fig1_t_end']
    N = P['N_grid']
    dx = P['L'] / N
    dt = P['dt_2d']
    n_steps = int(t_end / dt)
    node = P['obs_node']

    labels = [f'k₀={v}' for v in k0_vals]
    colors = ['steelblue', 'darkorange', 'crimson', 'forestgreen', 'purple']

    fig, axes = plt.subplots(len(k0_vals), 1, figsize=(10, 2.5 * len(k0_vals)))
    if len(k0_vals) == 1:
        axes = [axes]
    for ax, k0, lbl, col in zip(axes, k0_vals, labels, colors):
        print(f"    Running 2D sim for k₀={k0}...")
        U0, V0, Phi0 = init_wedge(N)
        _, _, traces = run_2d_sim(N, dx, dt, n_steps,
                                  k0, P['fig1_k1'], P['fig1_k2'],
                                  U0, V0, Phi0, record_nodes=[node])
        u_arr = np.array(traces[node])
        t_arr = np.arange(len(u_arr)) * dt
        ax.plot(t_arr, u_arr, color=col, lw=0.8)
        ax.set_ylabel('u', fontsize=11)
        ax.set_title(lbl, fontsize=11, pad=2)
        ax.set_ylim(-0.3, 1.5)
        ax.tick_params(axis='x', labelbottom=True)
        ax.set_xlabel('time', fontsize=11)
        ax.grid(alpha=0.3)
    fig.suptitle(f'Fig. 1 — Membrane potential at node (100,100)\n'
                 f'k₁={P["fig1_k1"]}, k₂={P["fig1_k2"]}, '
                 f'α={P["alpha"]}, β={P["beta"]}', fontsize=12, y=1.01)
    fig.tight_layout()
    path = os.path.join(FIGDIR, 'fig1_timeseries.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig 1 saved → {path}")


def make_fig2():
    """Fig 2 — Bifurcation diagram: max(u) vs k0 from node (100,100) in 2D PDE."""
    print("  Generating Fig 2 (bifurcation vs k0, 2D sim)...")
    k0_range = np.linspace(P['fig2_k0_min'], P['fig2_k0_max'], P['fig2_k0_steps'])
    N = P['N_grid']
    dx = P['L'] / N
    dt = P['dt_2d']
    n_total = int(P['fig2_t_end'] / dt)
    n_transient = int(P['fig2_t_transient'] / dt)
    node = P['bif_node']

    args_list = [
        (k0, k0, P['fig2_k1'], P['fig2_k2'], n_total, n_transient, N, dx, dt, node)
        for k0 in k0_range
    ]
    n_workers = os.cpu_count()
    print(f"    Sweeping k₀ with {n_workers} workers...")
    with multiprocessing.Pool(n_workers) as pool:
        results = pool.map(_bifurcation_worker, args_list)

    peaks_k0 = [(param_val, peaks) for param_val, peaks in results if peaks is not None]

    fig, ax = plt.subplots(figsize=(8, 5))
    for k0_val, peaks in peaks_k0:
        ax.scatter([k0_val]*len(peaks), peaks, s=0.5, c='k', alpha=0.6)
    ax.set_xlabel('k₀', fontsize=13)
    ax.set_ylabel('max u', fontsize=13)
    ax.set_title(f'Fig. 2 — Bifurcation: max u vs k₀  (node 125,75)\n'
                 f'k₁={P["fig2_k1"]}, k₂={P["fig2_k2"]}, '
                 f'α={P["alpha"]}, β={P["beta"]}', fontsize=12)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGDIR, 'fig2_bifurcation_k0.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig 2 saved → {path}")


def make_fig4():
    """Fig 4 — Bifurcation diagram: max(u) vs k2 from node (100,100) in 2D PDE."""
    print("  Generating Fig 4 (bifurcation vs k2, 2D sim)...")
    k2_range = np.linspace(P['fig4_k2_min'], P['fig4_k2_max'], P['fig4_k2_steps'])
    N = P['N_grid']
    dx = P['L'] / N
    dt = P['dt_2d']
    n_total = int(P['fig4_t_end'] / dt)
    n_transient = int(P['fig4_t_transient'] / dt)
    node = P['bif_node']

    args_list = [
        (k2_val, P['fig4_k0'], P['fig4_k1'], k2_val, n_total, n_transient, N, dx, dt, node)
        for k2_val in k2_range
    ]
    n_workers = os.cpu_count()
    print(f"    Sweeping k₂ with {n_workers} workers...")
    with multiprocessing.Pool(n_workers) as pool:
        results = pool.map(_bifurcation_worker, args_list)

    peaks_k2 = [(param_val, peaks) for param_val, peaks in results if peaks is not None]

    fig, ax = plt.subplots(figsize=(8, 5))
    for k2_val, peaks in peaks_k2:
        ax.scatter([k2_val]*len(peaks), peaks, s=0.5, c='navy', alpha=0.6)
    ax.set_xlabel('k₂', fontsize=13)
    ax.set_ylabel('max u', fontsize=13)
    ax.set_title(f'Fig. 4 — Bifurcation: max u vs k₂  (node 125,75)\n'
                 f'k₀={P["fig4_k0"]}, k₁={P["fig4_k1"]}, '
                 f'α={P["alpha"]}, β={P["beta"]}', fontsize=12)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGDIR, 'fig4_bifurcation_k2.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig 4 saved → {path}")


def make_fig3():
    """Fig 3 — Spiral wave formation (standard k2)."""
    print("  Generating Fig 3 (spiral wave formation)...")
    N = P['N_grid']
    dx = P['L'] / N
    dt = P['dt_2d']
    times = P['fig3_snap_times']
    snap_steps = [int(t / dt) for t in times]

    U0, V0, Phi0 = init_wedge(N)
    snapshots, _, _ = run_2d_sim(N, dx, dt, max(snap_steps) + 1,
                                 P['k0_2d'], P['k1_2d'], P['fig3_k2'],
                                 U0, V0, Phi0, snapshot_steps=snap_steps)

    n_snaps = len(snapshots)
    fig, axes = plt.subplots(1, n_snaps, figsize=(4 * n_snaps, 4), constrained_layout=True)
    if n_snaps == 1:
        axes = [axes]
    for ax, (step, U_snap, _), t in zip(axes, snapshots, times):
        im = ax.imshow(U_snap.T, cmap='jet', origin='lower',
                       vmin=-0.2, vmax=1.0, aspect='equal', extent=[0, N, 0, N])
        ax.set_title(f't = {t}', fontsize=13)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    plt.colorbar(im, ax=axes, shrink=0.8, label='u (membrane potential)')
    fig.suptitle(f'Fig. 3 — Spiral wave formation\n'
                 f'k₀={P["k0_2d"]}, k₁={P["k1_2d"]}, k₂={P["fig3_k2"]}, '
                 f'α={P["alpha"]}, β={P["beta"]}', fontsize=13)
    path = os.path.join(FIGDIR, 'fig3_spiral_formation.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig 3 saved → {path}")


def make_fig7():
    """Fig 7 — Spiral breakup at low k2."""
    print("  Generating Fig 7 (spiral pattern, low k2)...")
    N = P['N_grid']
    dx = P['L'] / N
    dt = P['dt_2d']
    times = P['fig7_snap_times']
    snap_steps = [int(t / dt) for t in times]

    U0, V0, Phi0 = init_wedge(N)

    snapshots, _, _ = run_2d_sim(N, dx, dt, max(snap_steps) + 1,
                                 P['k0_2d'], P['k1_2d'], P['fig7_k2'],
                                 U0, V0, Phi0, snapshot_steps=snap_steps)

    n_snaps = len(snapshots)
    fig, axes = plt.subplots(1, n_snaps, figsize=(4 * n_snaps, 4), constrained_layout=True)
    if n_snaps == 1:
        axes = [axes]
    for ax, (step, U_snap, _), t in zip(axes, snapshots, times):
        im = ax.imshow(U_snap.T, cmap='jet', origin='lower',
                       vmin=-0.2, vmax=1.0, aspect='equal', extent=[0, N, 0, N])
        ax.set_title(f't = {t}', fontsize=13)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    plt.colorbar(im, ax=axes, shrink=0.8, label='u (membrane potential)')
    fig.suptitle(f'Fig. 7 — Spatial pattern at k₂={P["fig7_k2"]}\n'
                 f'k₀={P["k0_2d"]}, k₁={P["k1_2d"]}, '
                 f'α={P["alpha"]}, β={P["beta"]}', fontsize=13)
    path = os.path.join(FIGDIR, 'fig7_spiral_k2_low.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig 7 saved → {path}")


def make_fig11():
    """Fig 11 — Spiral disruption by localised EM radiation."""
    print("  Generating Fig 11 (EM radiation spiral disruption)...")
    N = P['N_grid']
    dx = P['L'] / N
    dt = P['dt_2d']
    times = P['fig11_snap_times']
    snap_steps = [int(t / dt) for t in times]

    em_rad = {
        'cx': P['fig11_rad_cx'], 'cy': P['fig11_rad_cy'],
        'A': P['fig11_rad_A'], 'm': P['fig11_rad_m'],
    }

    U0, V0, Phi0 = init_wedge(N)
    snapshots, _, _ = run_2d_sim(N, dx, dt, max(snap_steps) + 1,
                                 P['k0_2d'], P['k1_2d'], P['fig11_k2'],
                                 U0, V0, Phi0,
                                 em_radiation=em_rad,
                                 snapshot_steps=snap_steps)

    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes_flat = axes.ravel()
    for i, (ax, (step, U_snap, Phi_snap), t) in enumerate(
            zip(axes_flat, snapshots, times)):
        im = ax.imshow(U_snap.T, cmap='jet', origin='lower',
                       vmin=0, vmax=3, aspect='equal', extent=[0, N, 0, N])
        lbl = panel_labels[i] if i < len(panel_labels) else ''
        ax.set_title(f'{lbl} t = {t}', fontsize=13)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    fig.subplots_adjust(right=0.88, hspace=0.30, wspace=0.30)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.025, 0.70])
    fig.colorbar(im, cax=cbar_ax, label='u (membrane potential)')
    fig.suptitle(f'Fig. 11 — Spiral disruption by EM radiation\n'
                 f'A={P["fig11_rad_A"]}, m={P["fig11_rad_m"]}, '
                 f'center ({P["fig11_rad_cx"]+1},{P["fig11_rad_cy"]+1}), '
                 f'k₂={P["fig11_k2"]}', fontsize=13)
    path = os.path.join(FIGDIR, 'fig11_em_radiation.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig 11 saved → {path}")


def make_fig14():
    """Fig 14 — Spiral breakup under EM noise."""
    print("  Generating Fig 14 (noise-driven spiral breakup)...")
    N = P['N_grid']
    dx = P['L'] / N
    dt = P['fig14_dt']

    noise_params = {
        'cx': P['fig14_noise_cx'], 'cy': P['fig14_noise_cy'],
        'r': P['fig14_noise_r'], 'D': P['fig14_noise_D'],
    }

    U0, V0, Phi0 = init_wedge(N)
    warm_time = P['fig14_warm_time']
    warm_steps = int(warm_time / dt)
    all_times = P['fig14_snap_times']
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    pre_noise_times = [t for t in all_times if t < warm_time]
    at_onset_times = [t for t in all_times if t == warm_time]
    post_noise_times = [t for t in all_times if t > warm_time]

    pre_snap_steps = [int(t / dt) for t in pre_noise_times]

    snapshots_pre, (U_warm, V_warm, Phi_warm), _ = run_2d_sim(
        N, dx, dt, warm_steps,
        P['k0_2d'], P['k1_2d'], P['fig14_k2'],
        U0, V0, Phi0, snapshot_steps=pre_snap_steps,
        clamp_U=(-0.5, 2.0))

    snapshots_onset = []
    if at_onset_times:
        snapshots_onset.append((warm_steps, U_warm.copy(), Phi_warm.copy()))

    post_noise_relative = [int((t - warm_time) / dt) for t in post_noise_times]

    # Time-accounting sanity checks: snapshots from the two phases are
    # concatenated and zipped *positionally* against all_times, so a silent
    # mislabelling happens unless the times are ascending, non-negative, and
    # each lands inside the simulated span. Fail loudly here instead.
    noisy_duration = (max(post_noise_relative) + 1) * dt if post_noise_relative else 0.0
    t_end_sim = warm_time + noisy_duration
    assert all_times == sorted(all_times), (
        f"fig14_snap_times must be ascending (got {all_times}); the pre/onset/"
        f"post snapshots are concatenated in time order and zipped by position.")
    for t in all_times:
        assert 0 <= t <= t_end_sim, (
            f"fig14_snap_times contains t={t}, outside the simulated interval "
            f"[0, {t_end_sim:.1f}] = warm_time({warm_time}) + noisy phase "
            f"({noisy_duration:.1f}). Adjust fig14_snap_times or fig14_warm_time.")

    snapshots_post, _, _ = run_2d_sim(
        N, dx, dt, max(post_noise_relative) + 1,
        P['k0_2d'], P['k1_2d'], P['fig14_k2'],
        U_warm, V_warm, Phi_warm,
        noise_params=noise_params,
        snapshot_steps=post_noise_relative,
        clamp_U=(-0.5, 2.0),
        seed=P['fig14_seed'])

    all_snapshots = snapshots_pre + snapshots_onset + snapshots_post

    # Each requested time must have produced exactly one snapshot; otherwise the
    # positional zip below would pair frames with the wrong time labels.
    assert len(all_snapshots) == len(all_times), (
        f"fig14 produced {len(all_snapshots)} snapshots for {len(all_times)} "
        f"requested times {all_times}: labels would be misaligned. Ensure each "
        f"fig14_snap_times value maps to a distinct simulated step (t/dt).")

    # Fig 14 — spatial snapshots (2×3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.ravel()
    for i, (ax, (step, U_snap, _), t) in enumerate(zip(axes_flat, all_snapshots, all_times)):
        im = ax.imshow(U_snap.T, cmap='jet', origin='lower',
                       vmin=0, vmax=1.6, aspect='equal', extent=[0, N, 0, N])
        label = panel_labels[i] if i < len(panel_labels) else ''
        if t < warm_time:
            note = ' (before noise)'
        elif t == warm_time:
            note = ' (noise onset)'
        else:
            note = ' (after noise onset)'
        ax.set_title(f'{label} t = {t}{note}', fontsize=13)
        circle = plt.Circle((noise_params['cx'], noise_params['cy']),
                             noise_params['r'], color='white', fill=False,
                             lw=1.5, ls='--', alpha=0.8)
        ax.add_patch(circle)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    for j in range(len(all_snapshots), len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.subplots_adjust(right=0.88, hspace=0.30, wspace=0.30)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.025, 0.70])
    fig.colorbar(im, cax=cbar_ax, label='u (membrane potential)')
    fig.suptitle(f'Fig. 14 — Spiral breakup under Gaussian EM noise\n'
                 f'k₂={P["fig14_k2"]}, D={P["fig14_noise_D"]}, '
                 f'noise onset at t={warm_time}, '
                 f'noise center ({noise_params["cx"]},{noise_params["cy"]}), '
                 f'r<={noise_params["r"]}', fontsize=13)
    path14 = os.path.join(FIGDIR, 'fig14_spiral_breakup.png')
    fig.savefig(path14, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig 14 saved → {path14}")


def make_fig16():
    """Fig 16 — Spiral wave at medium k2=0.5."""
    print("  Generating Fig 16 (spiral pattern, k2=0.5)...")
    N = P['N_grid']
    dx = P['L'] / N
    dt = P['dt_2d']
    times = P['fig16_snap_times']
    snap_steps = [int(t / dt) for t in times]

    U0, V0, Phi0 = init_wedge(N)
    snapshots, _, _ = run_2d_sim(N, dx, dt, max(snap_steps) + 1,
                                 P['k0_2d'], P['k1_2d'], P['fig16_k2'],
                                 U0, V0, Phi0, snapshot_steps=snap_steps)

    n_snaps = len(snapshots)
    fig, axes = plt.subplots(1, n_snaps, figsize=(4 * n_snaps, 4), constrained_layout=True)
    if n_snaps == 1:
        axes = [axes]
    for ax, (step, U_snap, _), t in zip(axes, snapshots, times):
        im = ax.imshow(U_snap.T, cmap='jet', origin='lower',
                       vmin=-0.2, vmax=1.0, aspect='equal', extent=[0, N, 0, N])
        ax.set_title(f't = {t}', fontsize=13)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    plt.colorbar(im, ax=axes, shrink=0.8, label='u (membrane potential)')
    fig.suptitle(f'Fig. 16 — Spatial pattern at k₂={P["fig16_k2"]}\n'
                 f'k₀={P["k0_2d"]}, k₁={P["k1_2d"]}, '
                 f'α={P["alpha"]}, β={P["beta"]}', fontsize=13)
    path = os.path.join(FIGDIR, 'fig16_spiral_k2_medium.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig 16 saved → {path}")


def make_fig17():
    """Fig 17 — Spiral wave at high k2=3.0."""
    print("  Generating Fig 17 (spiral pattern, k2=3.0)...")
    N = P['N_grid']
    dx = P['L'] / N
    dt = P['dt_2d']
    times = P['fig17_snap_times']
    snap_steps = [int(t / dt) for t in times]

    U0, V0, Phi0 = init_wedge(N)
    snapshots, _, _ = run_2d_sim(N, dx, dt, max(snap_steps) + 1,
                                 P['k0_2d'], P['k1_2d'], P['fig17_k2'],
                                 U0, V0, Phi0, snapshot_steps=snap_steps)

    n_snaps = len(snapshots)
    fig, axes = plt.subplots(1, n_snaps, figsize=(4 * n_snaps, 4), constrained_layout=True)
    if n_snaps == 1:
        axes = [axes]
    for ax, (step, U_snap, _), t in zip(axes, snapshots, times):
        im = ax.imshow(U_snap.T, cmap='jet', origin='lower',
                       vmin=-0.2, vmax=1.0, aspect='equal', extent=[0, N, 0, N])
        ax.set_title(f't = {t}', fontsize=13)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    plt.colorbar(im, ax=axes, shrink=0.8, label='u (membrane potential)')
    fig.suptitle(f'Fig. 17 — Spatial pattern at k₂={P["fig17_k2"]}\n'
                 f'k₀={P["k0_2d"]}, k₁={P["k1_2d"]}, '
                 f'α={P["alpha"]}, β={P["beta"]}', fontsize=13)
    path = os.path.join(FIGDIR, 'fig17_spiral_k2_high.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig 17 saved → {path}")


def make_fig18():
    """Fig 18 — Spiral patterns for varying k1 (rows) at k2=1.5."""
    print("  Generating Fig 18 (varying k1 comparison)...")
    N = P['N_grid']
    dx = P['L'] / N
    dt = P['dt_2d']
    k1_vals = P['fig18_k1_vals']
    k2 = P['fig18_k2']
    times = P['fig18_snap_times']
    snap_steps = [int(t / dt) for t in times]

    n_rows = len(k1_vals)
    n_cols = len(times)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4.2 * n_rows))

    for row, k1 in enumerate(k1_vals):
        print(f"    Running 2D sim for k₁={k1}...")
        U0, V0, Phi0 = init_wedge(N)
        snapshots, _, _ = run_2d_sim(N, dx, dt, max(snap_steps) + 1,
                                     P['k0_2d'], k1, k2,
                                     U0, V0, Phi0, snapshot_steps=snap_steps)
        for col, ((step, U_snap, _), t) in enumerate(zip(snapshots, times)):
            ax = axes[row, col]
            im = ax.imshow(U_snap.T, cmap='jet', origin='lower',
                           vmin=-0.2, vmax=1.0, aspect='equal',
                           extent=[0, N, 0, N])
            ax.set_title(f't = {t}', fontsize=11)
            if col == 0:
                ax.set_ylabel(f'k₁ = {k1}', fontsize=13)
            else:
                ax.set_ylabel('')
            ax.set_xlabel('x' if row == n_rows - 1 else '', fontsize=11)
            ax.tick_params(labelbottom=(row == n_rows - 1))

    fig.subplots_adjust(right=0.88, hspace=0.30, wspace=0.20)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.025, 0.70])
    fig.colorbar(im, cax=cbar_ax, label='u (membrane potential)')
    fig.suptitle(f'Fig. 18 — Effect of k₁ on spiral dynamics\n'
                 f'k₀={P["k0_2d"]}, k₂={k2}, '
                 f'α={P["alpha"]}, β={P["beta"]}', fontsize=14)
    path = os.path.join(FIGDIR, 'fig18_varying_k1.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig 18 saved → {path}")


def make_fig19():
    """Fig 19 — Spiral patterns for varying k0 (rows) at fig 2 conditions (k1=0.5, k2=1.0)."""
    print("  Generating Fig 19 (varying k0 comparison, fig 2 conditions)...")
    N = P['N_grid']
    dx = P['L'] / N
    dt = P['dt_2d']
    k0_vals = P['fig19_k0_vals']
    k1 = P['fig19_k1']
    k2 = P['fig19_k2']
    times = P['fig19_snap_times']
    snap_steps = [int(t / dt) for t in times]

    n_rows = len(k0_vals)
    n_cols = len(times)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4.2 * n_rows))

    for row, k0 in enumerate(k0_vals):
        print(f"    Running 2D sim for k₀={k0}...")
        U0, V0, Phi0 = init_wedge(N)
        snapshots, _, _ = run_2d_sim(N, dx, dt, max(snap_steps) + 1,
                                     k0, k1, k2,
                                     U0, V0, Phi0, snapshot_steps=snap_steps)
        for col, ((step, U_snap, _), t) in enumerate(zip(snapshots, times)):
            ax = axes[row, col]
            im = ax.imshow(U_snap.T, cmap='jet', origin='lower',
                           vmin=-0.2, vmax=1.0, aspect='equal',
                           extent=[0, N, 0, N])
            ax.set_title(f't = {t}', fontsize=11)
            if col == 0:
                ax.set_ylabel(f'k₀ = {k0}', fontsize=13)
            else:
                ax.set_ylabel('')
            ax.set_xlabel('x' if row == n_rows - 1 else '', fontsize=11)
            ax.tick_params(labelbottom=(row == n_rows - 1))

    fig.subplots_adjust(right=0.88, hspace=0.30, wspace=0.20)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.025, 0.70])
    fig.colorbar(im, cax=cbar_ax, label='u (membrane potential)')
    fig.suptitle(f'Fig. 19 — Effect of k₀ on spiral dynamics (Fig. 2 conditions)\n'
                 f'k₁={k1}, k₂={k2}, '
                 f'α={P["alpha"]}, β={P["beta"]}', fontsize=14)
    path = os.path.join(FIGDIR, 'fig19_varying_k0.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig 19 saved → {path}")


# ═══════════════════════════════════════════════
# Figure registry & interactive menu
# ═══════════════════════════════════════════════

FIGURES = {
    'init':  ('Initial perturbation at t=0',             make_fig_init),
    '1':     ('Time series for different k0',            make_fig1),
    '2':     ('Bifurcation diagram: max(u) vs k0',       make_fig2),
    '3':     ('Spiral wave formation',                    make_fig3),
    '4':     ('Bifurcation diagram: max(u) vs k2',       make_fig4),
    '7':     ('Spiral breakup at low k2',                 make_fig7),
    '11':    ('EM radiation spiral disruption',            make_fig11),
    '14':    ('Noise-driven spiral breakup',              make_fig14),
    '16':    ('Spiral pattern at medium k2=0.5',           make_fig16),
    '17':    ('Spiral pattern at high k2=3.0',             make_fig17),
    '18':    ('Varying k1 comparison (k2=1.5)',            make_fig18),
    '19':    ('Varying k0 comparison (fig 2 conditions)',  make_fig19),
}


def print_menu():
    print("\n╔══════════════════════════════════════════════╗")
    print("║   FHN-EM Figure Generator                    ║")
    print("╠══════════════════════════════════════════════╣")
    for key, (desc, _) in FIGURES.items():
        print(f"║  [{key:>5}]  {desc:<36}║")
    print("║  [  all]  Generate all figures               ║")
    print("║  [    q]  Quit                               ║")
    print("╚══════════════════════════════════════════════╝")


def run_figures(choices):
    if 'all' in choices:
        choices = list(FIGURES.keys())
    seen = set()
    for c in choices:
        if c in FIGURES:
            fn = FIGURES[c][1]
            if id(fn) not in seen:
                seen.add(id(fn))
                fn()
        else:
            print(f"  Unknown figure key: '{c}', skipping.")
    print(f"\nDone. Figures saved to {FIGDIR}")


def interactive():
    while True:
        print_menu()
        raw = input("\nChoose figures (comma/space separated, or 'all'): ").strip().lower()
        if raw in ('q', 'quit', 'exit', ''):
            break
        choices = [s.strip() for s in raw.replace(',', ' ').split()]
        run_figures(choices)
        again = input("\nGenerate more? [y/N]: ").strip().lower()
        if again not in ('y', 'yes'):
            break


if __name__ == '__main__':
    multiprocessing.freeze_support()
    if len(sys.argv) > 1:
        choices = [a.strip().lower() for a in sys.argv[1:]]
        run_figures(choices)
    else:
        interactive()
