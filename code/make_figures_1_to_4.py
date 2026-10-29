# %% Cell 2


import numpy as np, math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pathlib

import scienceplots  # noqa: F401

plt.style.use(["science", "nature"])

# Keep the plotting settings close to the original notebooks
plt.rcParams.update({
    "font.size": 24,
    "axes.labelsize": 24,
    "axes.titlesize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 20,
    "lines.linewidth": 2.0,
    "figure.dpi": 200,
})
LEGEND_FONTSIZE = 20


# %% Cell 3


np.set_printoptions(precision=6, suppress=True)

# --------------------------
# Inputs / outputs
# --------------------------
CODE_DIR = pathlib.Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
NPZ_A = CODE_DIR / "hetero_ring_instance.npz"          # strongly heterogeneous instance (provided)
NPZ_D = CODE_DIR / "orb_order1_instance.npz"            # near-homogeneous instance (provided)
NPZ_B = CODE_DIR / "kappa_below1_ring_instance.npz"     # dedicated K<1 instance (generated once)
NPZ_K = CODE_DIR / "korb_tight_ring_instance.npz"       # dedicated kORB instance (generated once)

OUTPUT_DIR = ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG1_OUT = str(OUTPUT_DIR / "fig1_strong_heterogeneity.png")
FIG2_OUT = str(OUTPUT_DIR / "fig2_near_homogeneous.png")
FIG3_OUT = str(OUTPUT_DIR / "fig3_kappa_below1.png")
FIG4_OUT = str(OUTPUT_DIR / "fig4_multimode_korb.png")

GMIN, GMAX, NG = 0.05, 1.50, 80
GAMMAS = np.linspace(GMIN, GMAX, NG)
GMIN_K, GMAX_K, NG_K = 0.50, 1.50, 70
GAMMAS_K = np.linspace(GMIN_K, GMAX_K, NG_K)
CLIP = 7.0
TOL_NEG = 1e-10
TOL_FLUX = 1e-12
BINS_H = 16


# Parameters for the dedicated K<1 instance, extracted from the local-search notebook.
LOG_S_B = np.array([
    -0.94444347, -0.50647629, -0.29469266,  0.09559974, -0.40137469,
    -0.10100789,  0.52430664, -1.10725268, -1.53979490,  0.27711451,
     0.31501532,  0.58080998, -0.05160088,  0.91272070
], dtype=float)
PHI_B = np.array([
     0.87788177, -1.33465519,  0.70971933,  0.48334775, -0.43116095,
    -0.33490353,  0.66146665,  0.72638122,  0.11254697, -1.93063325,
     0.70826334,  1.61746474, -2.61076945,  1.93356425
], dtype=float)

# Parameters for the dedicated kORB instance.
# Rationale: the kinetic ORB uses J_{ij}^2 / Q_{ij} <= 4 Q_{ij}; this becomes tight
# when the edge currents are close to one-way flow, i.e. |J_{ij}| ~ 2 Q_{ij}. A nearly
# translation-invariant ring with a strong uniform affinity makes this mechanism visible.
M_K = 24
PHI_BASE_K = 4.0
S_BASE_K = 1.0

# --------------------------
# Core ring utilities
# Numerical convention. The matrix built below is W_num = W^T
# relative to the manuscript convention. Hence rows sum to zero. The flux
# variable fij = pi[i] * W[i, j] is the transpose-image of the manuscript
# flux F_{ij} = pi_j W_{ij}, so its orientation is reversed, but all plotted
# quantities use the corresponding undirected edge scalars.
# --------------------------
def build_ring_generator_general(s, phi_vec):
    m = len(s)
    W = np.zeros((m, m), float)
    for i in range(m):
        j = (i + 1) % m
        ph = float(phi_vec[i])
        ef = math.exp(ph / 2.0)
        eb = math.exp(-ph / 2.0)
        a = float(s[i] * ef)
        b = float(s[i] * eb)
        W[i, j] += a
        W[j, i] += b
    W[np.diag_indices(m)] = -np.sum(W, axis=1)
    return W


def stationary_pi_dense(W):
    # Because rows of W sum to zero here, pi solves W^T pi = 0.
    m = W.shape[0]
    A = W.T.copy()
    b = np.zeros(m, float)
    A[-1, :] = 1.0
    b[-1] = 1.0
    pi = np.linalg.solve(A, b)
    pi = np.maximum(pi, 0.0)
    pi /= pi.sum()
    return pi


def oscillatory_modes_dense(W, imag_tol=TOL_NEG, real_tol=TOL_NEG):
    # Right eigenvectors of W_num are the manuscript left eigenvectors.
    vals, vecs = np.linalg.eig(W)
    modes = []
    for idx, val in enumerate(vals):
        if np.real(val) < -real_tol and np.imag(val) > imag_tol:
            psi = vecs[:, idx].astype(complex)
            modes.append((complex(val), psi, int(idx)))
    modes.sort(key=lambda item: np.real(item[0]), reverse=True)
    return modes


def osb_lambda1_and_vec_dense(W, tol=TOL_NEG):
    modes = oscillatory_modes_dense(W, imag_tol=tol, real_tol=tol)
    if not modes:
        return None, None
    lam1, psi1, _ = modes[0]
    return lam1, psi1


def entropy_sigma_and_edge_ring(W, pi):
    m = W.shape[0]
    sigma = 0.0
    edge_sig = np.zeros(m, float)
    for i in range(m):
        j = (i + 1) % m
        fij = pi[i] * W[i, j]
        fji = pi[j] * W[j, i]
        J = fij - fji
        term = J * math.log(fij / fji)
        edge_sig[i] = term
        sigma += term
    return float(sigma), edge_sig


def rho_from_edge_sig_ring(edge_sig):
    m = len(edge_sig)
    rho = np.zeros(m, float)
    for i in range(m):
        rho[i] = 0.5 * (edge_sig[i] + edge_sig[(i - 1) % m])
    return rho


def kappa_star_from_rho(pi, sigma, rho):
    return float(np.max(rho / (sigma * pi)))


def kappa_eff_from_mode(psi, pi, sigma, rho):
    abs2 = (psi.conj() * psi).real
    denom = float(np.sum(pi * abs2))
    return float(np.sum(rho * abs2) / (sigma * denom))


def mode_weights(psi, pi):
    abs2 = (psi.conj() * psi).real
    denom = float(np.sum(pi * abs2))
    return (pi * abs2) / denom


def activity_mode_ring(W, pi, psi):
    abs2 = (psi.conj() * psi).real
    denom = float(np.sum(pi * abs2))
    m = W.shape[0]
    total = 0.0
    for i in range(m):
        j = (i + 1) % m
        fij = pi[i] * W[i, j]
        fji = pi[j] * W[j, i]
        # This Q_{ij} is the same undirected-edge scalar as in the manuscript,
        # even though the oriented flux labels are transposed in the notebook.
        Q = 0.5 * (fij + fji)
        total += 2.0 * Q * 0.5 * (abs2[i] + abs2[j])
    return float(total / denom)


def chi2_from_weights(w, pi):
    return float(np.sum((w - pi) ** 2 / pi))


def var_pi(values, pi):
    mean = float(np.sum(pi * values))
    return float(np.sum(pi * (values - mean) ** 2))


def envelope_sigma_ratio(W, lam, sigma, tol=TOL_FLUX):
    # The envelope uses a manuscript right eigenvector, so here we diagonalize W_num^T.
    vals, vecs = np.linalg.eig(W.T)
    idx = int(np.argmin(np.abs(vals - lam)))
    v = vecs[:, idx].astype(complex)
    r = np.abs(v).astype(float)
    ssum = float(np.sum(r))
    if not np.isfinite(ssum) or ssum <= 0:
        return np.inf
    r /= ssum

    m = W.shape[0]
    sig = 0.0
    for i in range(m):
        j = (i + 1) % m
        fij = r[i] * W[i, j]
        fji = r[j] * W[j, i]
        if fij <= tol or fji <= tol:
            return np.inf
        J = fij - fji
        sig += J * math.log(fij / fji)
    return float(sig / sigma)


def mode_metrics_from_W(W, pi, sigma, rho, lam, psi):
    w = mode_weights(psi, pi)
    h = rho / (sigma * pi)
    K = float(kappa_eff_from_mode(psi, pi, sigma, rho))
    Kstar = float(kappa_star_from_rho(pi, sigma, rho))
    A_mode = float(activity_mode_ring(W, pi, psi))
    Env = float(envelope_sigma_ratio(W, lam, sigma))
    chi2 = float(chi2_from_weights(w, pi))
    var_h = float(np.sum(pi * (h - 1.0) ** 2))
    hl_bound = math.sqrt(max(var_h, 0.0)) * math.sqrt(max(chi2, 0.0))
    lhs_dim = float((lam.imag ** 2) / abs(lam.real))
    lower = 1.0 - hl_bound
    upper = 1.0 + hl_bound
    return {
        "LHSdim": lhs_dim,
        "R": lhs_dim / sigma,
        "SigmaMode": sigma * K,
        "Kinetic": 2.0 * A_mode,
        "SigmaKstar": sigma * Kstar,
        "SigmaEnv": sigma * Env,
        "sigma": float(sigma),
        "K": K,
        "Kstar": Kstar,
        "Env": Env,
        "A_mode": A_mode,
        "chi2": chi2,
        "var_h": var_h,
        "HLBound": hl_bound,
        "KLower": lower,
        "KUpper": upper,
        "w": w,
        "h": h,
    }


def sweep_instance(s, phi, gammas):
    rows = []
    for gamma in gammas:
        phi_t = np.clip(gamma * phi, -CLIP, CLIP)
        W = build_ring_generator_general(s, phi_t)
        pi = stationary_pi_dense(W)
        sigma, edge_sig = entropy_sigma_and_edge_ring(W, pi)
        lam1, psi = osb_lambda1_and_vec_dense(W)
        if lam1 is None or sigma <= 0 or lam1.real >= 0:
            rows.append(dict(
                gamma=float(gamma),
                LHSdim=np.nan, R=np.nan, SigmaMode=np.nan, Kinetic=np.nan,
                SigmaKstar=np.nan, SigmaEnv=np.nan,
                sigma=np.nan, K=np.nan, Kstar=np.nan, Env=np.nan,
                A_mode=np.nan, chi2=np.nan, var_h=np.nan, HLBound=np.nan,
                KLower=np.nan, KUpper=np.nan,
            ))
            continue

        rho = rho_from_edge_sig_ring(edge_sig)
        metrics = mode_metrics_from_W(W, pi, sigma, rho, lam1, psi)

        if metrics["LHSdim"] > metrics["SigmaMode"] + 1e-9:
            raise RuntimeError(
                f"thermodynamic ORB check failed at gamma={gamma:.6g}: "
                f"{metrics['LHSdim']} vs {metrics['SigmaMode']}"
            )
        if metrics["LHSdim"] > metrics["Kinetic"] + 1e-9:
            raise RuntimeError(
                f"kinetic ORB check failed at gamma={gamma:.6g}: "
                f"{metrics['LHSdim']} vs {metrics['Kinetic']}"
            )
        if metrics["SigmaMode"] > metrics["SigmaKstar"] + 1e-9:
            raise RuntimeError(
                f"definition check failed at gamma={gamma:.6g}: SigmaMode > SigmaKstar"
            )

        rows.append(dict(gamma=float(gamma), **{k: v for k, v in metrics.items() if k not in ["w", "h"]}))
    return pd.DataFrame(rows)


def multimode_dataframe(W, pi, sigma, rho):
    modes = oscillatory_modes_dense(W)
    rows = []
    for k, (lam, psi, eig_idx) in enumerate(modes, start=1):
        metrics = mode_metrics_from_W(W, pi, sigma, rho, lam, psi)
        rows.append(dict(
            mode=k,
            eig_index=eig_idx,
            Re=float(np.real(lam)),
            Im=float(np.imag(lam)),
            **{k2: v2 for k2, v2 in metrics.items() if k2 not in ["w", "h"]}
        ))
    return pd.DataFrame(rows)


def ensure_kappa_below1_instance(npz_path=NPZ_B):
    path = pathlib.Path(npz_path)
    if path.exists():
        dat = np.load(path)
        return dat["s"].astype(float), dat["phi"].astype(float)

    s = np.exp(LOG_S_B)
    phi = PHI_B.copy()
    np.savez(
        path,
        s=s,
        phi=phi,
        note=(
            "Dedicated K<1 instance extracted from the local-search notebook. "
            "The parameters are chosen so that the slowest oscillatory mode can "
            "satisfy LHS < K_eff < 1 over part of the gamma sweep."
        ),
    )
    return s, phi


def ensure_korb_tight_instance(npz_path=NPZ_K, m=M_K, s_base=S_BASE_K, phi_base=PHI_BASE_K):
    path = pathlib.Path(npz_path)
    if path.exists():
        dat = np.load(path)
        return dat["s"].astype(float), dat["phi"].astype(float)

    s = np.full(m, float(s_base), dtype=float)
    phi = np.full(m, float(phi_base), dtype=float)
    np.savez(
        path,
        s=s,
        phi=phi,
        note=(
            "Dedicated nearly translation-invariant ring for the kinetic ORB. "
            "Uniform rates and strong uniform affinity are chosen so that "
            "|J_{ij}| is close to 2 Q_{ij} on each edge, making the inequality "
            "J_{ij}^2 / Q_{ij} <= 4 Q_{ij} comparatively tight."
        ),
        m=int(m),
        s_base=float(s_base),
        phi_base=float(phi_base),
    )
    return s, phi


def panel_label(ax, text):
    ax.text(0.01, 0.965, text, transform=ax.transAxes,
            ha="left", va="top", fontweight="bold", color="black")


# --------------------------
# Load / prepare the instances
# --------------------------
datA = np.load(NPZ_A)
sA = datA["s"].astype(float)
phiA = datA["phi"].astype(float)
dfA = sweep_instance(sA, phiA, GAMMAS)

idx_star_A = int(np.nanargmax(dfA["R"].values))
gamma_star_A = float(dfA["gamma"].values[idx_star_A])
phi_star_A = np.clip(gamma_star_A * phiA, -CLIP, CLIP)
W_star_A = build_ring_generator_general(sA, phi_star_A)
pi_star_A = stationary_pi_dense(W_star_A)
sigma_star_A, edge_sig_star_A = entropy_sigma_and_edge_ring(W_star_A, pi_star_A)
rho_star_A = rho_from_edge_sig_ring(edge_sig_star_A)
lam1_star_A, psi1_star_A = osb_lambda1_and_vec_dense(W_star_A)
metrics_star_A = mode_metrics_from_W(W_star_A, pi_star_A, sigma_star_A, rho_star_A, lam1_star_A, psi1_star_A)
w_star_A = metrics_star_A["w"]
h_star_A = metrics_star_A["h"]
K_star_val_A = metrics_star_A["K"]
mode_df_A = multimode_dataframe(W_star_A, pi_star_A, sigma_star_A, rho_star_A)
mode_df_other_A = mode_df_A.loc[mode_df_A["mode"] > 1].copy()
if mode_df_other_A.empty:
    mode_df_other_A = mode_df_A.copy()

datD = np.load(NPZ_D)
sD = datD["s"].astype(float)
phiD = datD["phi"].astype(float)
dfD = sweep_instance(sD, phiD, GAMMAS)

idx_star_D = int(np.nanargmax(dfD["R"].values))
gamma_star_D = float(dfD["gamma"].values[idx_star_D])
phi_star_D = np.clip(gamma_star_D * phiD, -CLIP, CLIP)
W_star_D = build_ring_generator_general(sD, phi_star_D)
pi_star_D = stationary_pi_dense(W_star_D)
sigma_star_D, edge_sig_star_D = entropy_sigma_and_edge_ring(W_star_D, pi_star_D)
rho_star_D = rho_from_edge_sig_ring(edge_sig_star_D)
lam1_star_D, psi1_star_D = osb_lambda1_and_vec_dense(W_star_D)
metrics_star_D = mode_metrics_from_W(W_star_D, pi_star_D, sigma_star_D, rho_star_D, lam1_star_D, psi1_star_D)
w_star_D = metrics_star_D["w"]
h_star_D = metrics_star_D["h"]
K_star_val_D = metrics_star_D["K"]

# Dedicated K<1 instance
sB, phiB = ensure_kappa_below1_instance(NPZ_B)
dfB = sweep_instance(sB, phiB, GAMMAS)
maskB = np.isfinite(dfB["K"].values)
if np.any(maskB & (dfB["K"].values < 1.0)):
    idx_star_B = int(np.nanargmin(dfB["K"].values))
else:
    idx_star_B = int(np.nanargmin(np.abs(dfB["K"].values - 1.0)))
gamma_star_B = float(dfB["gamma"].values[idx_star_B])
phi_star_B = np.clip(gamma_star_B * phiB, -CLIP, CLIP)
W_star_B = build_ring_generator_general(sB, phi_star_B)
pi_star_B = stationary_pi_dense(W_star_B)
sigma_star_B, edge_sig_star_B = entropy_sigma_and_edge_ring(W_star_B, pi_star_B)
rho_star_B = rho_from_edge_sig_ring(edge_sig_star_B)
lam1_star_B, psi1_star_B = osb_lambda1_and_vec_dense(W_star_B)
metrics_star_B = mode_metrics_from_W(W_star_B, pi_star_B, sigma_star_B, rho_star_B, lam1_star_B, psi1_star_B)
w_star_B = metrics_star_B["w"]
h_star_B = metrics_star_B["h"]
K_star_val_B = metrics_star_B["K"]

# Dedicated kinetic-ORB instance, generated once then reloaded.
sK, phiK = ensure_korb_tight_instance(NPZ_K)
dfK = sweep_instance(sK, phiK, GAMMAS_K)

# %% Cell 4

# --------------------------
# --------------------------
def panel_label_fixed(ax, label_text):
    """
    Place panel labels in axes coordinates with a fixed horizontal offset.
    """
    ax.text(-0.18, 1.02, rf"\textbf{{{label_text}}}", transform=ax.transAxes,
            fontsize=20, fontweight='bold', va='bottom', ha='left')
# --------------------------
# Plotting helpers and colors
# --------------------------
COL_LHS = "#547BB4"
COL_SIGMA = "#629C35"
COL_KIN = "#7A4DA3"
COL_KSTAR = "#8C9A2B"
COL_ENV = "#C0321A"
COL_HL = "#D4842A"
COL_VAR = "#4A5F7E"
COL_CHI = "#C0321A"
COL_BASE = "#4D4D4D"

# Colors from the original histogram panel
COL_HIST_BASE = "#4A5F7E"
COL_HIST_MODE = "#E29135"
COL_HIST_LINE = "#C0321A"
# --------------------------
# --------------------------
fig1, axes1 = plt.subplots(1, 4, figsize=(24, 4.7))
x = dfA["gamma"].values

common_legend_kwargs = {
    "fontsize": LEGEND_FONTSIZE, 
    "handlelength": 1.0,
    "handletextpad": 0.3,
    "frameon": False,
}

# (a)
ax = axes1[0]
ax.plot(x, dfA["R"].values, marker="o", markersize=3, linewidth=1.6, label=r"$\mathcal{R}(\lambda_1)$", color=COL_LHS)
ax.plot(x, dfA["K"].values, linewidth=1.6, label=r"$\mathcal{K}(\psi_1)$", color=COL_SIGMA)
ax.plot(x, dfA["Kstar"].values, linestyle=":", linewidth=1.6, label=r"$\mathcal{K}_\star$", color=COL_KSTAR)
ax.plot(x, dfA["Env"].values, linestyle="-.", linewidth=1.6, label=r"$\sigma(r)/\sigma$", color=COL_ENV)
ax.axhline(1.0, linestyle="--", linewidth=1.6, color=COL_BASE, label=r"$1$")
ax.set_xlabel(r"$\gamma$")
ax.set_ylabel(r"bound value")
ax.set_xlim(GMIN, GMAX)
ax.set_ylim(0.8, 5)
ax.legend(loc="upper center", **common_legend_kwargs,bbox_to_anchor=(0.51, 1.05))
for spine in ax.spines.values():
    spine.set_linewidth(1.6)
ax.text(-0.15, 0.94, "(a)", transform=ax.transAxes,
            fontsize=24, fontweight='bold', va='bottom', ha='left')

# (b)
ax = axes1[1]
ax.plot(x, dfA["KUpper"].values,linestyle="-.", linewidth=1.6, label=r"$1 + B_{\rm HL}(\psi_1)$", color=COL_HL)
ax.plot(x, dfA["KLower"].values, linewidth=1.6, label=r"$1 - B_{\rm HL}(\psi_1)$", color=COL_HL, linestyle=":")
ax.plot(x, dfA["K"].values, linewidth=1.6, label=r"$\mathcal{K}(\psi_1)$", color=COL_SIGMA)
ax.axhline(1.0, linestyle="--", linewidth=1.6, color=COL_BASE, label=r"$1$")

ax.set_xlabel(r"$\gamma$")
ax.set_ylabel(r"value")
ax.set_xlim(GMIN, GMAX)
ax.set_ylim(0, 4)
ax.legend(loc="upper left", **common_legend_kwargs)
for spine in ax.spines.values():
    spine.set_linewidth(1.6)
ax.text(-0.15, 0.94, "(b)", transform=ax.transAxes,
            fontsize=24, fontweight='bold', va='bottom', ha='left')

# (c)
ax = axes1[2]
ax.plot(x, dfA["var_h"].values,linestyle="--", linewidth=1.6, label=r"${\rm Var}_\pi(h)$", color=COL_VAR)
ax.plot(x, dfA["chi2"].values, linewidth=1.6, label=r"$\chi^2(w(\psi_1)\Vert\pi)$", color=COL_CHI)
ax.set_xlabel(r"$\gamma$")
ax.set_ylabel(r"factor value")
ax.set_xlim(GMIN, GMAX)
ax.legend(loc="upper left", **common_legend_kwargs)
for spine in ax.spines.values():
    spine.set_linewidth(1.6)
ax.text(-0.15, 0.94, "(c)", transform=ax.transAxes,
            fontsize=24, fontweight='bold', va='bottom', ha='left')

ax = axes1[3]
edges = np.linspace(float(np.min(h_star_A)), float(np.max(h_star_A)), BINS_H + 1)
ax.hist(h_star_A, bins=edges, density=True, alpha=0.8, label=r"$h_i$", color=COL_HIST_BASE)
ax.hist(h_star_A, bins=edges, density=True, weights=w_star_A,
        histtype="step", linewidth=2.8, linestyle="--",
        label=r"$w_i h_i$", color=COL_HIST_LINE)
ax.axvline(K_star_val_A, linewidth=1.6,
           label=rf"$\mathcal{{K}}(\psi_1)\approx {K_star_val_A:.2f}$", color="black")
ax.set_xlabel(r"value of landscape")
ax.set_ylabel("density")
ax.legend(loc="upper right", **common_legend_kwargs)
for spine in ax.spines.values():
    spine.set_linewidth(1.6)
ax.text(-0.22, 0.94, "(d)", transform=ax.transAxes,
            fontsize=24, fontweight='bold', va='bottom', ha='left')

plt.subplots_adjust(wspace=0.25)
fig1.savefig(FIG1_OUT, dpi=200, bbox_inches="tight", pad_inches=0.05)


# %% Cell 5

# --------------------------
# Fig. 2: near-homogeneous instance, 1x4 thermodynamic comparison for lambda_1
# Layout intentionally kept identical to Fig. 1.
# --------------------------
fig2, axes2 = plt.subplots(1, 4, figsize=(24, 4.7))
x = dfD["gamma"].values

# (a)
ax = axes2[0]
ax.plot(x, dfD["R"].values, marker="o", markersize=3, linewidth=1.6,
        label=r"$\mathcal{R}(\lambda_1)$", color=COL_LHS)
ax.plot(x, dfD["K"].values, linewidth=1.6,
        label=r"$\mathcal{K}(\psi_1)$", color=COL_SIGMA)
ax.plot(x, dfD["Kstar"].values, linestyle=":", linewidth=1.6,
        label=r"$\mathcal{K}_\star$", color=COL_KSTAR)
ax.plot(x, dfD["Env"].values, linestyle="-.", linewidth=1.6,
        label=r"$\sigma(r)/\sigma$", color=COL_ENV)
ax.axhline(1.0, linestyle="--", linewidth=1.6, color=COL_BASE, label=r"$1$")
ax.set_xlabel(r"$\gamma$")
ax.set_ylabel(r"bound value")
ax.set_xlim(GMIN, GMAX)
ax.set_ylim(0.9, 1.2)
ax.legend(loc="upper center", **common_legend_kwargs, bbox_to_anchor=(0.51, 1.05))
for spine in ax.spines.values():
    spine.set_linewidth(1.6)
ax.text(-0.24, 0.94, "(a)", transform=ax.transAxes,
        fontsize=24, fontweight='bold', va='bottom', ha='left')

# (b)
ax = axes2[1]
ax.plot(x, dfD["KUpper"].values, linestyle="-.", linewidth=1.6,
        label=r"$1 + B_{\rm HL}(\psi_1)$", color=COL_HL)
ax.plot(x, dfD["KLower"].values, linewidth=1.6,
        label=r"$1 - B_{\rm HL}(\psi_1)$", color=COL_HL, linestyle=":")
ax.plot(x, dfD["K"].values, linewidth=1.6,
        label=r"$\mathcal{K}(\psi_1)$", color=COL_SIGMA)
ax.axhline(1.0, linestyle="--", linewidth=1.6, color=COL_BASE, label=r"$1$")
ax.set_xlabel(r"$\gamma$")
ax.set_ylabel(r"value")
ax.set_xlim(GMIN, GMAX)
ax.set_ylim(0.99, 1.01)
ax.legend(loc="lower right", **common_legend_kwargs)
for spine in ax.spines.values():
    spine.set_linewidth(1.6)
ax.text(-0.33, 0.94, "(b)", transform=ax.transAxes,
        fontsize=24, fontweight='bold', va='bottom', ha='left')

# (c)
ax = axes2[2]
ax.plot(x, dfD["var_h"].values, linestyle="--", linewidth=1.6,
        label=r"${\rm Var}_\pi(h)$", color=COL_VAR)
ax.plot(x, dfD["chi2"].values, linewidth=1.6,
        label=r"$\chi^2(w(\psi_1)\Vert\pi)$", color=COL_CHI)
ax.set_xlabel(r"$\gamma$")
ax.set_ylabel(r"factor value")
ax.set_xlim(GMIN, GMAX)
ymax = float(np.nanmax(np.vstack([dfD["var_h"].values, dfD["chi2"].values])))
ax.set_ylim(0.0, 1.05 * ymax)
ax.legend(loc="upper left", **common_legend_kwargs)
for spine in ax.spines.values():
    spine.set_linewidth(1.6)
ax.text(-0.28, 0.94, "(c)", transform=ax.transAxes,
        fontsize=24, fontweight='bold', va='bottom', ha='left')

# (d)
ax = axes2[3]
edges = np.linspace(float(np.min(h_star_D)), float(np.max(h_star_D)), BINS_H + 1)
ax.hist(h_star_D, bins=edges, density=True, alpha=0.8,
        label=r"$h_i$", color=COL_HIST_BASE)
ax.hist(h_star_D, bins=edges, density=True, weights=w_star_D,
        histtype="step", linewidth=2.8, linestyle="--",
        label=r"$w_i h_i$", color=COL_HIST_LINE)
ax.axvline(K_star_val_D, linewidth=1.6,
           label=rf"$\mathcal{{K}}(\psi_1)\approx {K_star_val_D:.2f}$", color="black")
ax.set_xlabel(r"value of landscape")
ax.set_ylabel("density")
ax.legend(loc="upper left", borderaxespad=0.15, **common_legend_kwargs)
for spine in ax.spines.values():
    spine.set_linewidth(1.6)
ax.text(-0.22, 0.94, "(d)", transform=ax.transAxes,
        fontsize=24, fontweight='bold', va='bottom', ha='left')

plt.subplots_adjust(wspace=0.35)
fig2.savefig(FIG2_OUT, dpi=200, bbox_inches="tight", pad_inches=0.05)
print("Saved:", FIG2_OUT)

# %% Cell 6

# --------------------------
# Fig. 3: dedicated K<1 instance, 1x4 thermodynamic comparison for lambda_1
# Layout intentionally kept identical to Fig. 1.
# --------------------------
fig3, axes3 = plt.subplots(1, 4, figsize=(24, 4.7))
x = dfB["gamma"].values

# (a)
ax = axes3[0]
ax.plot(x, dfB["R"].values, marker="o", markersize=3, linewidth=1.6,
        label=r"$\mathcal{R}(\lambda_1)$", color=COL_LHS)
ax.plot(x, dfB["K"].values, linewidth=1.6,
        label=r"$\mathcal{K}(\psi_1)$", color=COL_SIGMA)
ax.plot(x, dfB["Kstar"].values, linestyle=":", linewidth=1.6,
        label=r"$\mathcal{K}_\star$", color=COL_KSTAR)
ax.plot(x, dfB["Env"].values, linestyle="-.", linewidth=1.6,
        label=r"$\sigma(r)/\sigma$", color=COL_ENV)
ax.axhline(1.0, linestyle="--", linewidth=1.6, color=COL_BASE, label=r"$1$")
ax.set_xlabel(r"$\gamma$")
ax.set_ylabel(r"bound value")
ax.set_xlim(0.656,1.5)
ymax = float(np.nanmax(np.vstack([
    dfB["R"].values,
    dfB["K"].values,
    dfB["Kstar"].values,
    dfB["Env"].values,
    np.ones_like(dfB["R"].values),
])))
ymin = float(np.nanmin(np.vstack([
    dfB["R"].values,
    dfB["K"].values,
    np.ones_like(dfB["R"].values),
])))
ax.set_ylim(0,5)
ax.legend(loc="upper center", **common_legend_kwargs, bbox_to_anchor=(0.51, 1.05))
for spine in ax.spines.values():
    spine.set_linewidth(1.6)
ax.text(-0.15, 0.94, "(a)", transform=ax.transAxes,
        fontsize=24, fontweight='bold', va='bottom', ha='left')

# (b)
ax = axes3[1]

ax.plot(x, dfB["K"].values, linewidth=1.6,
        label=r"$\mathcal{K}(\psi_1)$", color=COL_SIGMA)
ax.axhline(1.0, linestyle="--", linewidth=1.6, color=COL_BASE, label=r"$1$")
ax.plot(x, dfB["KUpper"].values, linestyle="-.", linewidth=1.6,
        label=r"$1 + B_{\rm HL}(\psi_1)$", color=COL_HL)
ax.plot(x, dfB["KLower"].values, linewidth=1.6,
        label=r"$1 - B_{\rm HL}(\psi_1)$", color=COL_HL, linestyle=":")
ax.set_xlabel(r"$\gamma$")
ax.set_ylabel(r"value")
ax.set_xlim(0.656,1.5)
ymax = float(np.nanmax(np.vstack([
    dfB["K"].values,
    dfB["KUpper"].values,
    np.ones_like(dfB["K"].values),
])))
ymin = float(np.nanmin(np.vstack([
    dfB["K"].values,
    dfB["KLower"].values,
    np.ones_like(dfB["K"].values),
])))
pad = 0.05 * max(ymax - ymin, 1e-6)
ax.set_ylim(0,2)
ax.legend(loc="lower left", **common_legend_kwargs,ncol=2,columnspacing=0.5)
for spine in ax.spines.values():
    spine.set_linewidth(1.6)
ax.text(-0.23, 0.94, "(b)", transform=ax.transAxes,
        fontsize=24, fontweight='bold', va='bottom', ha='left')

# (c)
ax = axes3[2]
ax.plot(x, dfB["var_h"].values, linestyle="--", linewidth=1.6,
        label=r"${\rm Var}_\pi(h)$", color=COL_VAR)
ax.plot(x, dfB["chi2"].values, linewidth=1.6,
        label=r"$\chi^2(w(\psi_1)\Vert\pi)$", color=COL_CHI)
ax.set_xlabel(r"$\gamma$")
ax.set_ylabel(r"factor value")
ax.set_xlim(0.656,1.5)
ymax = float(np.nanmax(np.vstack([dfB["var_h"].values, dfB["chi2"].values])))
ax.set_ylim(0.0, 3)
ax.legend(loc="upper left", **common_legend_kwargs)
for spine in ax.spines.values():
    spine.set_linewidth(1.6)
ax.text(-0.17, 0.94, "(c)", transform=ax.transAxes,
        fontsize=24, fontweight='bold', va='bottom', ha='left')

# (d)
ax = axes3[3]
edges = np.linspace(float(np.min(h_star_B)), float(np.max(h_star_B)), BINS_H + 1)
ax.hist(h_star_B, bins=edges, density=True, alpha=0.8,
        label=r"$h_i$", color=COL_HIST_BASE)
ax.hist(h_star_B, bins=edges, density=True, weights=w_star_B,
        histtype="step", linewidth=2.8, linestyle="--",
        label=r"$w_i h_i$", color=COL_HIST_LINE)
ax.axvline(K_star_val_B, linewidth=1.6,
           label=rf"$\mathcal{{K}}(\psi_1)\approx {K_star_val_B:.2f}$", color="black")
ax.set_xlabel(r"value of landscape")
ax.set_ylabel("density")
ax.legend(loc="upper right", **common_legend_kwargs)
for spine in ax.spines.values():
    spine.set_linewidth(1.6)
ax.text(-0.22, 0.94, "(d)", transform=ax.transAxes,
        fontsize=24, fontweight='bold', va='bottom', ha='left')

plt.subplots_adjust(wspace=0.24)
fig3.savefig(FIG3_OUT, dpi=200, bbox_inches="tight", pad_inches=0.05)
print("Saved:", FIG3_OUT)

# %% Cell 7


# --------------------------
# Fig. 4: (a) other-mode thermodynamic ORB on Fig. 1 network; (b) dedicated kORB instance
# --------------------------
fig4, axes4 = plt.subplots(1, 2, figsize=(14, 4.7))

# (a) other oscillatory modes on the strongly heterogeneous network at gamma_star_A
ax = axes4[0]
m = mode_df_other_A["mode"].values
ax.plot(m, mode_df_other_A["R"].values, marker="o", linewidth=1.6,
        label=r"$\mathcal{R}(\lambda_{\ell}^{\rm osc})$", color=COL_LHS)
ax.plot(m, mode_df_other_A["K"].values, marker="s", linewidth=1.6,
        label=r"$\mathcal{K}(\psi_{\ell}^{\rm osc})$", color=COL_SIGMA)
# ax.axhline(1.0, linestyle="--", linewidth=1.6,
#            label=r"$1$", color=COL_BASE)
ax.set_xlabel(r"mode index $\ell$")
ax.set_ylabel(r"bound value")
ax.set_xticks(m)
ymax = float(np.nanmax(np.vstack([
    mode_df_other_A["R"].values,
    mode_df_other_A["K"].values,
    np.ones_like(mode_df_other_A["R"].values),
])))
ax.set_ylim(0.0, 1.05 * ymax)
ax.legend(loc="center right", fontsize=LEGEND_FONTSIZE, frameon=False)
for spine in ax.spines.values():
    spine.set_linewidth(1.6)
ax.text(-0.13, 0.93, "(a)", transform=ax.transAxes,
        fontsize=24, fontweight='bold', va='bottom', ha='left')

# (b) dedicated kORB-tight instance: lambda_1 sweep
ax = axes4[1]
x = dfK["gamma"].values
ax.plot(x, dfK["LHSdim"].values, marker="o", markersize=3, linewidth=1.6,
        label=r"$\Im(\lambda_1)^2/|\Re(\lambda_1)|$", color=COL_LHS)
ax.plot(x, dfK["Kinetic"].values, linewidth=1.6,
        label=r"$2\mathcal{A}(\psi_1)$", color=COL_KIN)
ax.set_xlabel(r"$\gamma$")
ax.set_ylabel(r"rate scale")
ax.set_xlim(GMIN_K, GMAX_K)
ymax = float(np.nanmax(np.vstack([
    dfK["LHSdim"].values,
    dfK["Kinetic"].values,
])))
ax.set_ylim(0.0, 1.05 * ymax)
ax.legend(loc="upper left", fontsize=LEGEND_FONTSIZE, frameon=False)
for spine in ax.spines.values():
    spine.set_linewidth(1.6)
ax.text(-0.17, 0.93, "(b)", transform=ax.transAxes,
        fontsize=24, fontweight='bold', va='bottom', ha='left')

plt.subplots_adjust(wspace=0.20)
fig4.savefig(FIG4_OUT, dpi=200, bbox_inches="tight", pad_inches=0.05)
print("Saved:", FIG4_OUT)

# %% Cell 8

# Save the numerical values used in Figs. 1--4.
DATA_DIR = ROOT / "source_data"
DATA_DIR.mkdir(exist_ok=True)
for name, frame in [("fig1_sweep", dfA), ("fig2_sweep", dfD),
                    ("fig3_sweep", dfB), ("fig4a_modes", mode_df_other_A),
                    ("fig4b_sweep", dfK)]:
    frame.to_csv(DATA_DIR / (name + ".csv"), index=False)
for name, pi, values in [("fig1", pi_star_A, metrics_star_A),
                         ("fig2", pi_star_D, metrics_star_D),
                         ("fig3", pi_star_B, metrics_star_B)]:
    pd.DataFrame({"state": np.arange(len(pi)), "pi": pi,
                  "h": values["h"], "w": values["w"]}).to_csv(
                      DATA_DIR / (name + "_state_profile.csv"), index=False)
