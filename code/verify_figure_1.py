"""Reconstruct Fig. 1 independently and check R1.9, including the OSB domain.

Dependencies: numpy, scipy, mpmath. Run from any working directory.
The generator uses the manuscript convention: W[j, i] is the rate i -> j.
"""

from pathlib import Path
import json

import mpmath as mp
import numpy as np
import scipy
from scipy.linalg import eig, solve


sr = Path(__file__).resolve().parents[1]
data = np.load(sr / "code/hetero_ring_instance.npz")
s, phi = data["s"], data["phi"]
n = len(s)


def evaluate(gamma):
    W = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        W[j, i] = s[i] * np.exp(gamma * phi[i] / 2)
        W[i, j] = s[i] * np.exp(-gamma * phi[i] / 2)
    W[np.diag_indices(n)] = -W.sum(axis=0)
    A = W.copy()
    A[-1] = 1
    b = np.zeros(n)
    b[-1] = 1
    pi = solve(A, b)
    vals, right = eig(W)
    zero = np.argmin(abs(vals))
    indices = [i for i in range(n) if i != zero]
    k = sorted(indices, key=lambda i: (-vals[i].real, -vals[i].imag))[0]
    lam, v = vals[k], right[:, k]
    left_vals, left = eig(W.T)
    psi = left[:, np.argmin(abs(left_vals - lam))]
    psi /= np.sqrt(np.dot(pi, abs(psi) ** 2))
    r = abs(v) / sum(abs(v))
    rho = np.zeros(n)
    sigma = sigmar = 0.0
    for i in range(n):
        j = (i + 1) % n
        f, g = pi[i] * W[j, i], pi[j] * W[i, j]
        ep = (f - g) * np.log(f / g)
        sigma += ep
        rho[i] += ep / 2
        rho[j] += ep / 2
        f, g = r[i] * W[j, i], r[j] * W[i, j]
        sigmar += (f - g) * np.log(f / g)
    h, w = rho / (sigma * pi), pi * abs(psi) ** 2
    K = np.dot(rho, abs(psi) ** 2) / sigma
    B = np.sqrt(np.dot(pi, (h - 1) ** 2) * np.sum((w - pi) ** 2 / pi))
    omega_over_decay = abs(lam.imag / lam.real)
    return dict(
        gamma=float(gamma), lambda_real=float(lam.real),
        lambda_imag=float(lam.imag), sigma=float(sigma),
        R=float(lam.imag ** 2 / (-lam.real * sigma)), K=float(K),
        Kstar=float(max(h)), B_HL=float(B), envelope_ratio=float(sigmar / sigma),
        omega_over_decay=float(omega_over_decay),
        coherent_number=float(omega_over_decay / (2 * np.pi)),
        osb_domain=bool(omega_over_decay >= 1), pi_min=float(min(pi)),
        column_sum_error=float(np.max(abs(W.sum(axis=0)))),
        stationarity_residual=float(np.max(abs(W @ pi))),
        eigen_residual_relative=float(np.linalg.norm(W @ v - lam * v)
                                      / (np.linalg.norm(W) * np.linalg.norm(v))),
        next_nonconjugate_real=float(max(vals[i].real for i in indices
            if abs(vals[i] - lam) > 1e-9 and abs(vals[i] - lam.conjugate()) > 1e-9)),
    )


records = [evaluate(gamma) for gamma in np.linspace(0.05, 1.50, 80)]
peak = max(records, key=lambda x: x["R"])
mp.mp.dps = 60
gamma = mp.mpf(repr(peak["gamma"]))
M = mp.matrix(n)
for i in range(n):
    j = (i + 1) % n
    si, ph = mp.mpf(repr(float(s[i]))), mp.mpf(repr(float(phi[i])))
    M[j, i] = si * mp.exp(gamma * ph / 2)
    M[i, j] = si * mp.exp(-gamma * ph / 2)
for i in range(n):
    M[i, i] = -sum(M[j, i] for j in range(n))
A, b = M.copy(), mp.matrix(n, 1)
for i in range(n):
    A[n - 1, i] = 1
b[n - 1] = 1
pi = mp.lu_solve(A, b)
eigenvalues = mp.eig(M, left=False, right=False)
lam = sorted((z for z in eigenvalues if abs(z) > mp.mpf("1e-40")),
             key=lambda z: (-mp.re(z), -mp.im(z)))[0]
sigma = mp.mpf("0")
for i in range(n):
    j = (i + 1) % n
    f, g = pi[i] * M[j, i], pi[j] * M[i, j]
    sigma += (f - g) * mp.log(f / g)
high_precision = dict(
    decimal_precision=60, gamma=str(gamma),
    lambda_real=mp.nstr(mp.re(lam), 45),
    lambda_imag_abs=mp.nstr(abs(mp.im(lam)), 45),
    sigma=mp.nstr(sigma, 45),
    R=mp.nstr(mp.im(lam) ** 2 / (-mp.re(lam) * sigma), 45),
    omega_over_decay=mp.nstr(abs(mp.im(lam) / mp.re(lam)), 45),
)
violations = [x for x in records if x["R"] > 1]
summary = dict(
    points_R_above_one=len(violations),
    points_R_above_one_in_OSB_domain=sum(x["osb_domain"] for x in violations),
    largest_omega_over_decay_where_R_above_one=max(x["omega_over_decay"] for x in violations),
    leading_mode_complex_at_all_points=all(x["lambda_imag"] > 1e-10 for x in records),
    max_R_minus_K=max(x["R"] - x["K"] for x in records),
    max_stationarity_residual=max(x["stationarity_residual"] for x in records),
)
result = dict(
    input="code/hetero_ring_instance.npz",
    implementation="Independent column-generator construction, stationary linear solve, full eigenvalue spectrum, and undirected edge entropy production; 60-digit mpmath reconstruction at the sweep maximum.",
    numpy_version=np.__version__, scipy_version=scipy.__version__, mpmath_version=mp.__version__,
    leading_selection="Exclude the stationary eigenvalue, then maximize real part over all remaining modes, including real modes.",
    osb_domain="N >= 1/(2*pi), equivalently abs(Im(lambda_1))/abs(Re(lambda_1)) >= 1; Oberreiter et al., PRE 106, 014106 (2022), pp. 1 and 3.",
    summary=summary, peak=peak, high_precision=high_precision, records=records,
)
(sr / "source_data/fig1_independent_check.json").write_text(
    json.dumps(result, indent=2) + "\n")
print(json.dumps(dict(summary=summary, peak=peak, high_precision=high_precision), indent=2))
