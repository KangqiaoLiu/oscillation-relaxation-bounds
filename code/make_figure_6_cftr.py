"""Reproduce the published CFTR-model calculation and manuscript Fig. 6.

The numerical generator q acts on row probabilities (q = W.T).
Published rates: Levring et al., Nature 616, 606--614 (2023).
"""
from __future__ import annotations
import csv
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Iterable
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFTR_STATE_SIGNALS = [
    ("S1", "low FRET", "no current"),
    ("S2", "low FRET", "no current"),
    ("S3", "high FRET", "no current"),
    ("S4", "high FRET", "no current"),
    ("S5", "high FRET", "current"),
    ("S6", "high FRET", "current"),
    ("S7", "high FRET", "no current"),
    ("S8", "high FRET", "no current"),
]

CFTR_RATE_TABLE = [
    ("k1_to_2", 1, 2, 1.0, "uM^-1 s^-1", "ATP_uM"),
    ("k2_to_1", 2, 1, 50.0, "s^-1", ""),
    ("k2_to_3", 2, 3, 10.0, "s^-1", ""),
    ("k3_to_2", 3, 2, 0.3, "s^-1", ""),
    ("k1_to_8", 1, 8, 0.3, "s^-1", ""),
    ("k8_to_1", 8, 1, 0.3, "s^-1", ""),
    ("k3_to_8", 3, 8, 0.6, "s^-1", ""),
    ("k8_to_3", 8, 3, 0.012, "uM^-1 s^-1", "ATP_uM"),
    ("k3_to_4", 3, 4, 1.3, "s^-1", ""),
    ("k4_to_3", 4, 3, 0.002, "s^-1", ""),
    ("k5_to_4", 5, 4, 20.0, "s^-1", ""),
    ("k4_to_5", 4, 5, 100.0, "s^-1", ""),
    ("k5_to_6", 5, 6, 2.0, "s^-1", "WT; set to 0 for E1371Q"),
    ("k6_to_5", 6, 5, 0.0, "s^-1", "hard zero in published WT simulation"),
    ("k6_to_7", 6, 7, 4.0, "s^-1", ""),
    ("k7_to_6", 7, 6, 0.0, "s^-1", "hard zero in published WT simulation"),
    ("k7_to_8", 7, 8, 4.5, "s^-1", ""),
    ("k8_to_7", 8, 7, 0.0, "s^-1", "hard zero in published WT simulation"),
]

CFTR_PROVENANCE = {
    "model": "Levring et al. 2023 CFTR 8-state kinetic simulation scheme",
    "article": {
        "citation": (
            "Levring J, Terry DS, Kilic Z, Fitzgerald G, Blanchard SC, "
            "Chen J. CFTR function, pathology and pharmacology at "
            "single-molecule resolution. Nature 616, 606-614 (2023)."
        ),
        "doi": "10.1038/s41586-023-05854-7",
        "url": "https://www.nature.com/articles/s41586-023-05854-7",
        "license": "CC BY 4.0",
    },
    "supplement": {
        "url": (
            "https://static-content.springer.com/esm/art%3A10.1038%2F"
            "s41586-023-05854-7/MediaObjects/41586_2023_5854_MOESM1_ESM.pdf"
        ),
        "used_sections": [
            "Supplementary Methods, lines 9-40: CTMC equations and generator matrix",
            "Supplementary Table 1: state signal assignments",
            "Supplementary Table 2: kinetic rate parameters used in simulations",
            "Extended Data Fig. 10a: topology and rates of simulated model",
        ],
    },
    "condition": {
        "ATP_uM": 3000.0,
        "description": "fully phosphorylated WT CFTR at 3 mM ATP",
    },
}

@dataclass
class ModeMetrics:
    eigenvalue_real: float
    eigenvalue_imag: float
    lhs_dimensional: float
    sigma: float
    R: float
    K: float
    Kstar: float
    Env: float
    BHL: float
    pi_min: float
    pi_max: float
    h_min: float
    h_max: float
    w_max: float
    complex_mode_count: int

def ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def write_csv(path: pathlib.Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def build_cftr_generator(atp_um: float, epsilon: float = 0.0) -> np.ndarray:
    q = np.zeros((8, 8), dtype=float)
    for _, src, dst, value, unit, note in CFTR_RATE_TABLE:
        rate = value * atp_um if unit == "uM^-1 s^-1" else value
        if rate > 0:
            q[src - 1, dst - 1] = rate
        elif epsilon > 0 and "hard zero" in note:
            q[src - 1, dst - 1] = epsilon

    # Regularize only published hard-zero reverse steps of positive forward
    # transitions. This keeps the original topology but makes Schnakenberg EP
    # finite for sensitivity checks.
    if epsilon > 0:
        for i, j in [(6, 5), (7, 6), (8, 7)]:
            q[i - 1, j - 1] = epsilon

    np.fill_diagonal(q, -np.sum(q, axis=1))
    return q

def stationary_dense(q: np.ndarray) -> np.ndarray:
    n = q.shape[0]
    a = q.T.copy()
    b = np.zeros(n)
    a[-1, :] = 1.0
    b[-1] = 1.0
    pi = np.linalg.solve(a, b)
    tiny_negative = (pi < 0) & (pi > -1e-12)
    pi[tiny_negative] = 0.0
    pi /= pi.sum()
    return pi

def undirected_pairs_from_generator(q: np.ndarray) -> list[tuple[int, int]]:
    rows, cols = np.nonzero(q > 0)
    pairs: set[tuple[int, int]] = set()
    for i, j in zip(rows, cols):
        if i == j:
            continue
        pairs.add(tuple(sorted((int(i), int(j)))))
    return sorted(pairs)

def complex_modes_dense(q: np.ndarray, imag_tol: float = 1e-10) -> list[tuple[complex, np.ndarray]]:
    vals, vecs = np.linalg.eig(q)
    modes: list[tuple[complex, np.ndarray]] = []
    for k, val in enumerate(vals):
        if val.real < -imag_tol and val.imag > imag_tol:
            modes.append((complex(val), vecs[:, k].astype(complex)))
    modes.sort(key=lambda item: item[0].real, reverse=True)
    return modes

def entropy_profile(q: np.ndarray, pi: np.ndarray) -> tuple[float, np.ndarray, bool]:
    sigma = 0.0
    rho = np.zeros(q.shape[0], dtype=float)
    finite = True
    for i, j in undirected_pairs_from_generator(q):
        fij = pi[i] * q[i, j]
        fji = pi[j] * q[j, i]
        if fij <= 0 or fji <= 0:
            finite = False
            continue
        term = (fij - fji) * math.log(fij / fji)
        sigma += term
        rho[i] += 0.5 * term
        rho[j] += 0.5 * term
    return float(sigma), rho, finite

def envelope_ratio_dense(q: np.ndarray, lam: complex, sigma: float) -> float:
    vals, vecs = np.linalg.eig(q.T)
    idx = int(np.argmin(np.abs(vals - lam)))
    r = np.abs(vecs[:, idx]).astype(float)
    r /= r.sum()
    sigma_r = 0.0
    for i, j in undirected_pairs_from_generator(q):
        fij = r[i] * q[i, j]
        fji = r[j] * q[j, i]
        if fij <= 0 or fji <= 0:
            return math.inf
        sigma_r += (fij - fji) * math.log(fij / fji)
    return float(sigma_r / sigma)

def metrics_for_dense_generator(q: np.ndarray) -> ModeMetrics | None:
    modes = complex_modes_dense(q)
    if not modes:
        return None
    lam, psi = modes[0]
    pi = stationary_dense(q)
    sigma, rho, finite = entropy_profile(q, pi)
    if (not finite) or sigma <= 0:
        return None

    abs2 = np.abs(psi) ** 2
    abs2 /= float(np.sum(pi * abs2))
    w = pi * abs2
    h = rho / (sigma * pi)
    k_eff = float(np.sum(rho * abs2) / sigma)
    k_star = float(np.max(h))
    lhs_dim = float((lam.imag**2) / abs(lam.real))
    r_ratio = float(lhs_dim / sigma)
    env = envelope_ratio_dense(q, lam, sigma)
    bhl = math.sqrt(max(float(np.sum(pi * (h - 1.0) ** 2)), 0.0))
    bhl *= math.sqrt(max(float(np.sum((w - pi) ** 2 / pi)), 0.0))

    return ModeMetrics(
        eigenvalue_real=float(lam.real),
        eigenvalue_imag=float(lam.imag),
        lhs_dimensional=lhs_dim,
        sigma=float(sigma),
        R=r_ratio,
        K=k_eff,
        Kstar=k_star,
        Env=env,
        BHL=float(bhl),
        pi_min=float(np.min(pi)),
        pi_max=float(np.max(pi)),
        h_min=float(np.min(h)),
        h_max=float(np.max(h)),
        w_max=float(np.max(w)),
        complex_mode_count=len(modes),
    )

def metrics_to_dict(metrics: ModeMetrics | None) -> dict:
    if metrics is None:
        return {}
    return metrics.__dict__.copy()

def run_cftr(output_dir: pathlib.Path) -> dict:
    atp_um = CFTR_PROVENANCE["condition"]["ATP_uM"]
    rate_rows = []
    for name, src, dst, value, unit, note in CFTR_RATE_TABLE:
        effective = value * atp_um if unit == "uM^-1 s^-1" else value
        rate_rows.append(
            {
                "rate_name": name,
                "source_state": f"S{src}",
                "destination_state": f"S{dst}",
                "published_value": value,
                "published_unit": unit,
                "ATP_uM": atp_um if unit == "uM^-1 s^-1" else "",
                "effective_rate_s^-1": effective,
                "note": note,
            }
        )
    write_csv(
        output_dir / "cftr_levring_2023_rates.csv",
        [
            "rate_name",
            "source_state",
            "destination_state",
            "published_value",
            "published_unit",
            "ATP_uM",
            "effective_rate_s^-1",
            "note",
        ],
        rate_rows,
    )

    state_rows = [
        {"state": state, "smFRET_signal": fret, "electrophysiology_signal": current}
        for state, fret, current in CFTR_STATE_SIGNALS
    ]
    write_csv(
        output_dir / "cftr_levring_2023_states.csv",
        ["state", "smFRET_signal", "electrophysiology_signal"],
        state_rows,
    )

    eps_values = [0.0, 1e-12, 1e-10, 1e-9, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1]
    sensitivity_rows = []
    selected_profile_rows = []
    raw_modes = complex_modes_dense(build_cftr_generator(atp_um, 0.0))
    raw_mode_summary = {
        "complex_mode_count": len(raw_modes),
        "leading_complex_eigenvalue": {
            "real": float(raw_modes[0][0].real) if raw_modes else None,
            "imag": float(raw_modes[0][0].imag) if raw_modes else None,
            "lhs_dimensional": (
                float(raw_modes[0][0].imag**2 / abs(raw_modes[0][0].real)) if raw_modes else None
            ),
        },
        "finite_entropy_production": False,
        "reason": "published generator has hard-zero reverse rates q65=q76=q87=0",
    }

    for eps in eps_values:
        q = build_cftr_generator(atp_um, eps)
        metrics = metrics_for_dense_generator(q)
        row = {"epsilon_s^-1": eps}
        if eps == 0.0:
            raw_eig = raw_mode_summary["leading_complex_eigenvalue"]
            row["eigenvalue_real"] = raw_eig["real"]
            row["eigenvalue_imag"] = raw_eig["imag"]
            row["lhs_dimensional"] = raw_eig["lhs_dimensional"]
            row["finite_entropy_production"] = False
            row["complex_mode_count"] = raw_mode_summary["complex_mode_count"]
        elif metrics is not None:
            row.update(metrics_to_dict(metrics))
            row["finite_entropy_production"] = True
        sensitivity_rows.append(row)

        if eps == 1e-6 and metrics is not None:
            pi = stationary_dense(q)
            sigma, rho, _ = entropy_profile(q, pi)
            _, psi = complex_modes_dense(q)[0]
            abs2 = np.abs(psi) ** 2
            abs2 /= float(np.sum(pi * abs2))
            w = pi * abs2
            h = rho / (sigma * pi)
            for idx in range(8):
                selected_profile_rows.append(
                    {
                        "epsilon_s^-1": eps,
                        "state": f"S{idx + 1}",
                        "pi": pi[idx],
                        "rho": rho[idx],
                        "h": h[idx],
                        "mode_weight_w": w[idx],
                    }
                )

    fieldnames = [
        "epsilon_s^-1",
        "finite_entropy_production",
        "complex_mode_count",
        "eigenvalue_real",
        "eigenvalue_imag",
        "lhs_dimensional",
        "sigma",
        "R",
        "K",
        "Kstar",
        "Env",
        "BHL",
        "pi_min",
        "pi_max",
        "h_min",
        "h_max",
        "w_max",
    ]
    write_csv(output_dir / "cftr_levring_2023_epsilon_sensitivity.csv", fieldnames, sensitivity_rows)
    write_csv(
        output_dir / "cftr_levring_2023_state_profile_eps1e-6.csv",
        ["epsilon_s^-1", "state", "pi", "rho", "h", "mode_weight_w"],
        selected_profile_rows,
    )

    best_metrics = metrics_for_dense_generator(build_cftr_generator(atp_um, 1e-6))
    return {
        "provenance": CFTR_PROVENANCE,
        "raw_model": raw_mode_summary,
        "epsilon_reference": 1e-6,
        "epsilon_reference_metrics": metrics_to_dict(best_metrics),
        "output_files": [
            "cftr_levring_2023_rates.csv",
            "cftr_levring_2023_states.csv",
            "cftr_levring_2023_epsilon_sensitivity.csv",
            "cftr_levring_2023_state_profile_eps1e-6.csv",
        ],
    }

def plot_cftr_response_figure(output_dir: pathlib.Path, fig_path: pathlib.Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
        import scienceplots  # noqa: F401

        plt.style.use(["science", "nature"])
    except Exception:
        import matplotlib.pyplot as plt

    rows = []
    csv_path = output_dir / "cftr_levring_2023_epsilon_sensitivity.csv"
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["finite_entropy_production"] == "True":
                rows.append(row)

    eps = np.array([float(row["epsilon_s^-1"]) for row in rows])
    series = {
        r"$\mathcal{R}(\lambda_1)$": np.array([float(row["R"]) for row in rows]),
        r"$\mathcal{K}(\psi_1)$": np.array([float(row["K"]) for row in rows]),
        r"$\sigma(r)/\sigma$": np.array([float(row["Env"]) for row in rows]),
        r"$\mathcal{K}_\star$": np.array([float(row["Kstar"]) for row in rows]),
    }
    styles = {
        r"$\mathcal{R}(\lambda_1)$": dict(color="#547BB4", marker="o", linestyle="-"),
        r"$\mathcal{K}(\psi_1)$": dict(color="#629C35", marker="s", linestyle="-"),
        r"$\sigma(r)/\sigma$": dict(color="#C0321A", marker="D", linestyle="-."),
        r"$\mathcal{K}_\star$": dict(color="#8C9A2B", marker="^", linestyle="--"),
    }

    plt.rcParams.update({
        "font.size": 10.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.0,
        "lines.linewidth": 1.4,
        "axes.linewidth": 0.8,
        "figure.dpi": 300,
    })
    fig, ax = plt.subplots(figsize=(3.05, 2.25), constrained_layout=True)
    for label, values in series.items():
        ax.plot(eps, values, label=label, markersize=4.0, markeredgewidth=0.0, **styles[label])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"reverse rate $\epsilon$ (s$^{-1}$)")
    ax.set_ylabel("dimensionless quantity")
    ax.set_ylim(2e-2, 1.2e1)
    ax.set_xlim(np.min(eps) * 0.7, np.max(eps) * 1.4)
    ax.legend(frameon=False, ncol=2, handlelength=1.6, columnspacing=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    ensure_dir(fig_path.parent)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    pdf_path = fig_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [fig_path.name, pdf_path.name]

if __name__ == "__main__":
    output = ROOT / "source_data" / "cftr"
    ensure_dir(output)
    result = run_cftr(output)
    plot_cftr_response_figure(output, ROOT / "figures" / "fig6_cftr.png")
    (output / "cftr_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["epsilon_reference_metrics"], indent=2))
