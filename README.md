# Oscillation-relaxation bounds for Markov jump processes

Code and data for the paper by Kangqiao Liu and Zongping Gong. Version 1.0.0.

## Contents

- `code/`: Python scripts and model inputs.
- `source_data/`: Numerical results.
- `figures/`: Figures 1–6.

## Run

Python 3.12.7 and TeX Live 2024 were used.

```sh
python -m pip install -r requirements.txt
python reproduce.py
```

For calculations without figure generation: `python reproduce.py --calculations-only`.

## Sources

- The cystic fibrosis transmembrane conductance regulator (CFTR) model uses rates and state assignments from [Levring et al. (2023)](https://doi.org/10.1038/s41586-023-05854-7), Supplementary Tables 1–2 (CC BY 4.0). Reverse rates S6→S5, S7→S6 and S8→S7 are set to a common ε.
- Three-layer model: [Shiraishi (2026)](https://arxiv.org/abs/2609.34352v1).
- Coherence comparison: [Oberreiter, Seifert and Barato (2022)](https://doi.org/10.1103/PhysRevE.106.014106).

## Citation and licenses

Citation: [10.5281/zenodo.23757111](https://doi.org/10.5281/zenodo.23757111) ([CITATION.cff](CITATION.cff)).

Code and documentation: [MIT](LICENSE). Model inputs, data and figures: [CC BY 4.0](LICENSE-DATA.md).
