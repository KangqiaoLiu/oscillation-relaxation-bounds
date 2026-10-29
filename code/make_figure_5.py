import argparse
import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image

import scienceplots  # noqa: F401

plt.style.use(["science", "nature"])

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
COL_LHS = "#547BB4"
COL_SIGMA = "#629C35"
COL_KSTAR = "#8C9A2B"
COL_ENV = "#C0321A"
COL_HL = "#D4842A"
COL_VAR = "#4A5F7E"
COL_BASE = "#4D4D4D"
COL_HIST_BASE = "#4A5F7E"
COL_HIST_MODE = "#E29135"
COL_HIST_LINE = "#C0321A"
BINS_H = 16
TARGET_PNG_SIZE = (3860, 912)

LABELS_PLAIN = ["C1", "C2", "O1", "O2", "I1", "I2"]
LABELS_TEX = [r"$C_1$", r"$C_2$", r"$O_1$", r"$O_2$", r"$I_1$", r"$I_2$"]

EDGE_LIST = [
    (0, 1, 1.25, 1.10),
    (1, 2, 0.85, 1.50),
    (2, 3, 1.10, 1.20),
    (3, 5, 0.75, 1.00),
    (5, 4, 1.40, 1.30),
    (4, 0, 0.95, 1.40),
    (1, 5, 0.22, -0.40),
    (2, 4, 0.18, 0.20),
]


def build_generator(gamma):
    n = 6
    W = np.zeros((n, n), float)
    for i, j, s, phi in EDGE_LIST:
        a = s * math.exp(gamma * phi / 2.0)
        b = s * math.exp(-gamma * phi / 2.0)
        W[i, j] += a
        W[j, i] += b
    W[np.diag_indices(n)] = -np.sum(W, axis=1)
    return W


def stationary_pi(W):
    n = W.shape[0]
    A = W.T.copy()
    b = np.zeros(n)
    A[-1, :] = 1.0
    b[-1] = 1.0
    pi = np.linalg.solve(A, b)
    pi = np.maximum(pi, 0.0)
    pi /= pi.sum()
    return pi


def leading_complex_mode(W):
    vals, vecs = np.linalg.eig(W)
    modes = []
    for k, val in enumerate(vals):
        if val.real < -1e-10 and val.imag > 1e-10:
            modes.append((complex(val), vecs[:, k].astype(complex)))
    modes.sort(key=lambda item: item[0].real, reverse=True)
    if not modes:
        return None, None
    return modes[0]


def envelope_sigma_ratio(W, lam, sigma, tol=1e-12):
    vals, vecs = np.linalg.eig(W.T)
    idx = int(np.argmin(np.abs(vals - lam)))
    v = vecs[:, idx].astype(complex)
    r = np.abs(v).astype(float)
    total = float(np.sum(r))
    if not np.isfinite(total) or total <= 0:
        return np.inf
    r /= total

    sigma_r = 0.0
    for i, j, _, _ in EDGE_LIST:
        fwd = r[i] * W[i, j]
        bwd = r[j] * W[j, i]
        if fwd <= tol or bwd <= tol:
            return np.inf
        sigma_r += (fwd - bwd) * math.log(fwd / bwd)
    return float(sigma_r / sigma)


def metrics(gamma):
    W = build_generator(gamma)
    pi = stationary_pi(W)
    lam, psi = leading_complex_mode(W)
    if lam is None:
        return None

    sigma = 0.0
    rho = np.zeros(6, float)
    for i, j, _, _ in EDGE_LIST:
        fwd = pi[i] * W[i, j]
        bwd = pi[j] * W[j, i]
        term = (fwd - bwd) * math.log(fwd / bwd)
        sigma += term
        rho[i] += 0.5 * term
        rho[j] += 0.5 * term

    abs2 = np.abs(psi) ** 2
    abs2 /= float(np.sum(pi * abs2))
    w = pi * abs2
    h = rho / (sigma * pi)
    K = float(np.sum(rho * abs2) / sigma)
    R = float(lam.imag ** 2 / (sigma * abs(lam.real)))
    Kstar = float(np.max(h))
    eta_inv = float(np.max(abs2))
    env = float(envelope_sigma_ratio(W, lam, sigma))
    var_h = float(np.sum(pi * (h - 1.0) ** 2))
    chi2 = float(np.sum((w - pi) ** 2 / pi))
    BHL = math.sqrt(max(var_h, 0.0)) * math.sqrt(max(chi2, 0.0))
    return dict(
        gamma=gamma,
        Re=lam.real,
        Im=lam.imag,
        sigma=sigma,
        R=R,
        K=K,
        Kstar=Kstar,
        Env=env,
        eta_inv=eta_inv,
        var_h=var_h,
        chi2=chi2,
        BHL=BHL,
        KUpper=1 + BHL,
        KLower=1 - BHL,
        pi=pi,
        h=h,
        w=w,
    )


def style_axes(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)


def panel_label(ax, label, x=-0.15):
    ax.text(
        x,
        0.94,
        label,
        transform=ax.transAxes,
        fontsize=24,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def draw_gating_scheme(ax):
    pos = np.array([
        [0.08, 0.68],
        [0.36, 0.88],
        [0.76, 0.70],
        [0.76, 0.30],
        [0.08, 0.32],
        [0.36, 0.12],
    ])
    label_offsets = [
        np.array([0.00, 0.075]),
        np.array([0.04, 0.070]),
        np.array([0.075, 0.00]),
        np.array([0.04, -0.070]),
        np.array([0.00, -0.075]),
        np.array([-0.075, 0.00]),
        np.array([-0.120, 0.085]),
        np.array([0.105, -0.045]),
    ]

    for e, (i, j, _, phi) in enumerate(EDGE_LIST):
        arrow = FancyArrowPatch(
            posA=pos[i],
            posB=pos[j],
            arrowstyle="<->",
            mutation_scale=14,
            linewidth=1.7,
            color=COL_BASE,
            shrinkA=22,
            shrinkB=22,
            zorder=1,
        )
        ax.add_patch(arrow)
        midpoint = 0.5 * (pos[i] + pos[j]) + label_offsets[e]
        ax.text(
            midpoint[0],
            midpoint[1],
            rf"${phi:.2f}$",
            ha="center",
            va="center",
            fontsize=17,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.95),
            zorder=4,
        )

    for k, (x, y) in enumerate(pos):
        node = plt.Circle(
            (x, y),
            0.075,
            facecolor="white",
            edgecolor=COL_VAR,
            linewidth=1.8,
            zorder=5,
        )
        ax.add_patch(node)
        ax.text(x, y, LABELS_TEX[k], ha="center", va="center", fontsize=22, zorder=6)

    ax.set_xlim(-0.03, 0.88)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    panel_label(ax, "(a)", x=-0.08)


def save_all_formats(fig, fig_file):
    fig_file = pathlib.Path(fig_file)
    fig_file.parent.mkdir(parents=True, exist_ok=True)
    if fig_file.suffix.lower() == ".png":
        tmp_file = fig_file.with_suffix(".tmp.png")
        fig.savefig(tmp_file, dpi=200, bbox_inches="tight", pad_inches=0.0)
        with Image.open(tmp_file) as img:
            img = img.convert("RGBA")
            target_w, target_h = TARGET_PNG_SIZE
            canvas_w = max(target_w, img.width)
            canvas_h = max(target_h, img.height)
            canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))
            x0 = (canvas_w - img.width) // 2
            y0 = (canvas_h - img.height) // 2
            canvas.alpha_composite(img, (x0, y0))
            canvas.convert("RGB").save(fig_file)
        tmp_file.unlink(missing_ok=True)
    else:
        fig.savefig(fig_file, dpi=200, bbox_inches="tight", pad_inches=0.0)

    for target in [fig_file.with_suffix(".pdf"), fig_file.with_suffix(".svg")]:
        fig.savefig(target, dpi=200, bbox_inches="tight", pad_inches=0.0)


def main(outdir, fig_path=None, sweep_csv=None, state_csv=None, param_json=None):
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    gammas = np.linspace(0.40, 1.50, 70)

    rows = []
    for g in gammas:
        m = metrics(float(g))
        if m is None:
            continue
        rows.append({k: v for k, v in m.items() if k not in ("pi", "h", "w")})
    df = pd.DataFrame(rows)

    idx = int(df["R"].idxmax())
    gstar = float(df.loc[idx, "gamma"])
    mstar = metrics(gstar)

    sweep_path = pathlib.Path(sweep_csv) if sweep_csv else outdir / "fig5_gating_model_sweep.csv"
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(sweep_path, index=False)

    state_path = pathlib.Path(state_csv) if state_csv else outdir / "fig5_gating_model_state_profile.csv"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "state": LABELS_PLAIN,
        "pi": mstar["pi"],
        "h": mstar["h"],
        "w": mstar["w"],
    }).to_csv(state_path, index=False)

    param_path = pathlib.Path(param_json) if param_json else outdir / "fig5_gating_model_parameters.json"
    param_path.parent.mkdir(parents=True, exist_ok=True)
    param_path.write_text(
        json.dumps({
            "states": LABELS_PLAIN,
            "edges": [
                {
                    "a": LABELS_PLAIN[i],
                    "b": LABELS_PLAIN[j],
                    "scale": s,
                    "local_affinity": phi,
                }
                for i, j, s, phi in EDGE_LIST
            ],
            "gamma_min": float(gammas[0]),
            "gamma_max": float(gammas[-1]),
            "gamma_points": int(len(gammas)),
            "gamma_displayed": gstar,
        }, indent=2),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 4, figsize=(24, 4.7))
    common_legend_kwargs = {
        "fontsize": LEGEND_FONTSIZE,
        "handlelength": 1.0,
        "handletextpad": 0.3,
        "frameon": False,
    }

    draw_gating_scheme(axes[0])

    x = df["gamma"].values

    ax = axes[1]
    ax.plot(x, df["R"].values, marker="o", markersize=3, linewidth=1.6,
            label=r"$\mathcal{R}(\lambda_1)$", color=COL_LHS)
    ax.plot(x, df["K"].values, linewidth=1.6,
            label=r"$\mathcal{K}(\psi_1)$", color=COL_SIGMA)
    ax.plot(x, df["Kstar"].values, linestyle=":", linewidth=1.6,
            label=r"$\mathcal{K}_\star$", color=COL_KSTAR)
    ax.plot(x, df["Env"].values, linestyle="-.", linewidth=1.6,
            label=r"$\sigma(r)/\sigma$", color=COL_ENV)
    ax.axhline(1.0, linestyle="--", linewidth=1.6, color=COL_BASE, label=r"$1$")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"bound value")
    ax.set_xlim(0.40, 1.50)
    ax.set_ylim(0.0, 1.6)
    ax.legend(loc="lower right", **common_legend_kwargs)
    style_axes(ax)
    panel_label(ax, "(b)", x=-0.23)

    ax = axes[2]
    ax.plot(x, df["KUpper"].values, linestyle="-.", linewidth=1.6,
            label=r"$1+B_{\rm HL}(\psi_1)$", color=COL_HL)
    ax.plot(x, df["KLower"].values, linestyle=":", linewidth=1.6,
            label=r"$1-B_{\rm HL}(\psi_1)$", color=COL_HL)
    ax.plot(x, df["K"].values, linewidth=1.6,
            label=r"$\mathcal{K}(\psi_1)$", color=COL_SIGMA)
    ax.axhline(1.0, linestyle="--", linewidth=1.6, color=COL_BASE, label=r"$1$")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"value", labelpad=0)
    ax.set_xlim(0.40, 1.50)
    yvals = np.concatenate([
        df["KUpper"].values,
        df["KLower"].values,
        df["K"].values,
        np.ones(len(df)),
    ])
    ymin = float(np.nanmin(yvals))
    ymax = float(np.nanmax(yvals))
    pad = max(0.02, 0.15 * (ymax - ymin))
    ax.set_ylim(ymin - pad, ymax + pad)
    hl_legend_kwargs = dict(common_legend_kwargs)
    hl_legend_kwargs.update({
        "fontsize": LEGEND_FONTSIZE - 4,
        "handlelength": 0.7,
        "handletextpad": 0.2,
    })
    ax.legend(loc="lower right", bbox_to_anchor=(0.98, 0.02),
              borderaxespad=0.0, **hl_legend_kwargs)
    style_axes(ax)
    panel_label(ax, "(c)", x=-0.23)

    ax = axes[3]
    h = np.asarray(mstar["h"], float)
    w = np.asarray(mstar["w"], float)
    edges = np.linspace(float(np.min(h)), float(np.max(h)), BINS_H + 1)
    ax.hist(h, bins=edges, density=True, alpha=0.8,
            label=r"$h_i$", color=COL_HIST_BASE)
    ax.hist(h, bins=edges, density=True, weights=w,
            histtype="step", linewidth=2.8, linestyle="--",
            label=r"$w_i h_i$", color=COL_HIST_LINE)
    ax.axvline(mstar["K"], linewidth=1.6,
               label=rf"$\mathcal{{K}}(\psi_1)\approx {mstar['K']:.2f}$", color="black")
    ax.set_xlabel(r"value of landscape")
    ax.set_ylabel("density")
    hist_legend_kwargs = dict(common_legend_kwargs)
    hist_legend_kwargs.update({
        "fontsize": LEGEND_FONTSIZE - 2,
        "handlelength": 0.8,
        "handletextpad": 0.2,
    })
    ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98),
              borderaxespad=0.0, **hist_legend_kwargs)
    style_axes(ax)
    panel_label(ax, "(d)", x=-0.22)

    plt.subplots_adjust(wspace=0.30)
    fig_file = pathlib.Path(fig_path) if fig_path else outdir / "fig5_gating_model.png"
    save_all_formats(fig, fig_file)
    print(
        "gamma_star",
        gstar,
        "K",
        mstar["K"],
        "R",
        mstar["R"],
        "Env",
        float(df.loc[idx, "Env"]),
        "eta_inv",
        mstar["eta_inv"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig", default=str(ROOT / "figures/fig5_gating_model.png"))
    parser.add_argument("--sweep-csv", default=str(ROOT / "source_data/fig5_gating_model_sweep.csv"))
    parser.add_argument("--state-csv", default=str(ROOT / "source_data/fig5_gating_model_state_profile.csv"))
    parser.add_argument("--parameters-json", default=str(ROOT / "source_data/fig5_gating_model_parameters.json"))
    args = parser.parse_args()
    main(
        pathlib.Path(args.sweep_csv).parent,
        fig_path=args.fig,
        sweep_csv=args.sweep_csv,
        state_csv=args.state_csv,
        param_json=args.parameters_json,
    )
