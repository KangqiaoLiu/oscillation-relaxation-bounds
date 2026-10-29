# Code and data for Oscillation-relaxation bounds for Markov jump processes

Kangqiao Liu and Zongping Gong. Version 1.0.0.

Code and source data accompanying *Oscillation-relaxation bounds for Markov jump processes*. This package contains the model inputs, numerical source data, and Python programs for the six manuscript figures and additional calculations. The examples include heterogeneous rings, a six-state network with coupled cycles, and the published eight-state wild-type cystic fibrosis transmembrane conductance regulator (CFTR) kinetic model.

## Reproduction

The calculations were run with Python 3.12.7. Package versions are recorded in `requirements.txt`. Figure generation uses the SciencePlots `science` and `nature` styles and a TeX installation; TeX Live 2024 was used for the supplied figures.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
```

Run `python reproduce.py --calculations-only` for the independent Fig. 1 reconstruction and the additional four-state and three-layer calculations. Individual scripts may be run from any working directory. They locate inputs relative to their own files and write results into `source_data/` and `figures/`.

## Files and scientific conventions

| Program or data | Contents |
|---|---|
| `code/make_figures_1_to_4.py` | Heterogeneous and homogeneous ring calculations and Figs. 1–4. |
| `code/*.npz` | Four fixed input instances containing traffic scales `s` and local affinities `phi`. |
| `source_data/fig1_sweep.csv` through `fig3_sweep.csv` | 80-point sweeps of the thermodynamic and kinetic quantities. Rows where the leading nonstationary mode is real have empty oscillatory-mode fields. |
| `source_data/fig1_state_profile.csv` through `fig3_state_profile.csv` | Stationary probability `pi`, landscape `h`, and modal weight `w` at the displayed histogram point. State indices start at zero. |
| `source_data/fig4a_modes.csv`, `fig4b_sweep.csv` | Other oscillatory modes and the 70-point kinetic comparison. |
| `code/make_figure_5.py` | Six-state network with coupled cycles; includes all eight bidirectional links and parameters. |
| `source_data/fig5_gating_model_*` | Parameters, 70-point sweep, and selected state profile. |
| `code/make_figure_6_cftr.py` | Eight-state WT CFTR calculation at 3 mM ATP and Fig. 6. |
| `source_data/cftr/` | Published rate table and state assignments, reverse-rate sweep, selected state profile, and provenance. Rates have units s⁻¹ after inserting ATP concentration in µM. |
| `code/verify_figure_1.py` | Independent column-generator reconstruction, full-spectrum leading-mode selection, and 60-digit mpmath calculation at the sampled maximum. |
| `source_data/fig1_independent_check.json` | Full Fig. 1 sweep, coherence-domain comparison, residuals, and increased-precision values. |
| `code/additional_calculations.py` | Four-state example with envelope entropy production below stationary entropy production; right-mode time-reversal identity; Shiraishi three-layer asymptotics. |
| `source_data/four_state_envelope_example.json` | Rates, stationary distribution, eigenvalue, envelope, and entropy-production decomposition for the four-state example. |
| `source_data/shiraishi_asymptotics.csv` | Exact first-Fourier-block calculations for L = 10⁴ to 10⁸, q = 3/4, c₀ = 16, d = 3. |
| `figures/` | The six manuscript figures. |

The manuscript uses a column-probability generator W. The figure scripts use row-sum-zero matrices q = Wᵀ; their right eigenvectors are observable modes. `verify_figure_1.py` and `additional_calculations.py` use the manuscript's column convention explicitly.

In the ring CSV files, `R` is Im(λ)²/(σ|Re(λ)|), `K` is the sampled factor, `Kstar` is max(h), `Env` is σ(r)/σ, and `HLBound` is B_HL. `LHSdim`, `SigmaMode`, `Kinetic`, `SigmaKstar`, and `SigmaEnv` are rate-valued quantities. The numerical rate unit for the constructed examples is arbitrary. CFTR rates use seconds. The Shiraishi model uses the time unit of its stated rate construction.

CFTR reverse rates on S6→S5, S7→S6, and S8→S7 are replaced by a common ε for finite entropy production. The source data include ε = 0 for the published generator; its entropy-production fields are empty and `finite_entropy_production` is false. The plotted positive sweep spans 10⁻¹² to 10⁻¹ s⁻¹. The manuscript quotes the small-reverse-rate range 10⁻¹² to 10⁻⁶ s⁻¹.

The three-layer data evaluate the original rate formulas through their 3 × 3 Fourier block, without constructing a 3L × 3L matrix. The manuscript's Methods provides the asymptotic derivation. `saturation` denotes (K−1)/B_HL; the full ORB has R/K ∼ ε in this family.

## Sources and attribution

- Levring, J. et al. *CFTR function, pathology and pharmacology at single-molecule resolution*. Nature **616**, 606–614 (2023). [Article and Supplementary Information](https://doi.org/10.1038/s41586-023-05854-7). The kinetic parameters and state labels come from Supplementary Tables 1–2 and the simulation scheme. The article is licensed CC BY 4.0; the ε prescription is specified above.
- Shiraishi, N. *Second largest eigenvalue does not bound stationary entropy production*. [arXiv:2609.34352v1](https://arxiv.org/abs/2609.34352v1). Its three-layer rate construction is used to calculate the sampled dissipation and heterogeneity-localization quantities defined in our manuscript.
- Oberreiter, L., Seifert, U. and Barato, A. C. *Universal minimal cost of coherent biochemical oscillations*. [Phys. Rev. E **106**, 014106 (2022)](https://doi.org/10.1103/PhysRevE.106.014106). The coherence-domain condition used in the independent Fig. 1 calculation is N ≥ (2π)⁻¹.

## Citation

Citation metadata for this software is provided in [CITATION.cff](CITATION.cff). GitHub displays it under **Cite this repository**. Cite the version used in your calculations:

> Liu, K. and Gong, Z. *Code and data for Oscillation-relaxation bounds for Markov jump processes*, version 1.0.0. https://github.com/KangqiaoLiu/oscillation-relaxation-bounds

## Licenses

- Python code and accompanying documentation: [MIT License](LICENSE).
- Model inputs (`code/*.npz`), numerical source data (`source_data/`), and figures (`figures/`): [Creative Commons Attribution 4.0 International](LICENSE-DATA.md).

The CFTR parameters and state assignments retain the attribution to Levring et al. above. The reverse-rate modification is described under **Files and scientific conventions**. Please retain the source attributions when reusing the associated materials.
