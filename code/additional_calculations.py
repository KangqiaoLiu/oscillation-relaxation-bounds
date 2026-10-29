"""Four-state envelope example and Shiraishi-model modal sampling.

All generators in this file use the manuscript's column-probability convention.
The three-layer calculation uses its exact 3 x 3 first Fourier block, so large
ring sizes require no large matrices. Rate formulas are from Shiraishi,
arXiv:2609.34352v1; h, w, K and B_HL are evaluated from this manuscript.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def four_state_example():
    forward = np.array([0.272, 4.106, 0.609, 0.608])
    backward = np.array([0.321, 0.194, 0.148, 0.259])
    W = np.zeros((4, 4))
    for i in range(4):
        j = (i + 1) % 4
        W[j, i], W[i, j] = forward[i], backward[i]
    W[np.diag_indices(4)] = -W.sum(axis=0)
    A, b = W.copy(), np.array([0., 0., 0., 1.])
    A[-1] = 1
    pi = np.linalg.solve(A, b)
    vals, vecs = np.linalg.eig(W)
    indices = [k for k in range(4) if abs(vals[k]) > 1e-10]
    k = max(indices, key=lambda k: (vals[k].real, vals[k].imag))
    lam, v = vals[k], vecs[:, k]
    r = abs(v) / abs(v).sum()
    rho = np.zeros(4)
    sigma = sigma_r = housekeeping = 0.
    for i in range(4):
        j = (i + 1) % 4
        f, g = pi[i] * forward[i], pi[j] * backward[i]
        fr, gr = r[i] * forward[i], r[j] * backward[i]
        ep = (f - g) * np.log(f / g)
        sigma += ep
        rho[i] += ep / 2
        rho[j] += ep / 2
        sigma_r += (fr - gr) * np.log(fr / gr)
        housekeeping += (fr - gr) * np.log(f / g)
    phi = v / pi
    reverse = np.diag(pi) @ W.T @ np.diag(1 / pi)
    sigma_right = np.dot(rho, abs(phi)**2) / np.dot(pi, abs(phi)**2)
    return dict(
        forward_rates=forward.tolist(), backward_rates=backward.tolist(),
        lambda_real=float(lam.real), lambda_imag=float(lam.imag),
        sigma=float(sigma), sigma_r=float(sigma_r),
        envelope_ratio=float(sigma_r / sigma),
        housekeeping=float(housekeeping), nonadiabatic=float(sigma_r-housekeeping),
        spectral_ratio=float(lam.imag**2 / abs(lam.real)),
        sigma_right=float(sigma_right),
        time_reversal_eigen_residual=float(np.linalg.norm(reverse.T @ phi-lam*phi)),
        stationary_distribution=pi.tolist(), right_envelope=r.tolist())


def three_layer_model(L, q=0.75, c0=16., d=3.):
    epsilon, theta = L**(-q), 2*np.pi/L
    p = c0*epsilon**2
    a, u = 1-epsilon, (1-epsilon)*p/(1-p)
    kbc, v = (a+d*epsilon)/2, (d-1)*epsilon*np.sqrt(p)
    b = (d-1)*(1-p)*epsilon/(2*np.sqrt(p))
    Win = np.array([[-a, u-v, u+v],
                    [a/2+b, -u-kbc, kbc-v],
                    [a/2-b, kbc+v, -u-kbc]])
    P = np.array([p, (1-p)/2, (1-p)/2])
    block = Win.astype(complex) + np.diag([-epsilon+2j, -10., -10.])
    vals, vecs = np.linalg.eig(block.T)
    k = np.argmin(abs(vals-(-1+2j)))
    lam, psi = vals[k], vecs[:, k]
    psi /= np.sqrt(np.dot(P, abs(psi)**2))
    w = P * abs(psi)**2
    # 1-cos(theta) = 2*sin(theta/2)^2 avoids cancellation at large L.
    rate_base = epsilon/(4*np.sin(theta/2)**2)
    rate_bias = 1/np.sin(theta)
    sigma_A = p * (2*rate_bias) * (2*np.arctanh(rate_bias/rate_base))
    rho = np.array([sigma_A, 0., 0.])
    sigma_vertical = 0.
    for i in range(3):
        for j in range(i+1, 3):
            f, g = Win[j, i]*P[i], Win[i, j]*P[j]
            ep = (f-g)*np.log(f/g)
            sigma_vertical += ep
            rho[i] += ep/2
            rho[j] += ep/2
    sigma = rho.sum()
    h = rho/(sigma*P)
    K = np.dot(w, h)
    var_h = np.dot(P, (h-1)**2)
    chi2 = np.sum((w-P)**2/P)
    BHL = np.sqrt(var_h*chi2)
    R = lam.imag**2/(abs(lam.real)*sigma)
    return dict(L=L, q=q, c0=c0, d=d, epsilon=epsilon, p=p,
                lambda_real=lam.real, lambda_imag=lam.imag,
                sigma=sigma, sigma_A=sigma_A, sigma_vertical=sigma_vertical,
                h_A=h[0], h_B=h[1], h_C=h[2], w_A=w[0], w_B=w[1], w_C=w[2],
                K=K, Sigma=sigma*K, R=R, var_h=var_h, chi2=chi2, BHL=BHL,
                p_K=p*K, sigma_epsilon_over_p=sigma*epsilon/p,
                p_var_h=p*var_h, p_chi2=p*chi2,
                saturation=(K-1)/BHL, upper_saturation=K/(1+BHL),
                R_over_epsilon_K=R/(epsilon*K),
                modal_ratio=(w[1]+w[2])/w[0],
                stationary_residual=float(np.linalg.norm(Win@P)),
                eigen_residual=float(np.linalg.norm(block.T@psi-lam*psi)))


if __name__ == '__main__':
    output = ROOT/'source_data'
    output.mkdir(exist_ok=True)
    witness = four_state_example()
    (output/'four_state_envelope_example.json').write_text(json.dumps(witness, indent=2)+'\n')
    data = pd.DataFrame([three_layer_model(L) for L in [10000, 100000, 1000000, 10000000, 100000000]])
    data.to_csv(output/'shiraishi_asymptotics.csv', index=False)
    print(json.dumps(witness, indent=2))
    print(data[['L','p_K','sigma_epsilon_over_p','p_var_h','p_chi2','saturation','R_over_epsilon_K']].to_string(index=False))
