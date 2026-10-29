"""Reproduce the figures and numerical values supplied with the manuscript."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--calculations-only', action='store_true',
                    help='Run the independent Fig. 1 and additional calculations without plotting.')
args = parser.parse_args()
scripts = ['additional_calculations.py', 'verify_figure_1.py']
if not args.calculations_only:
    scripts += ['make_figures_1_to_4.py', 'make_figure_5.py', 'make_figure_6_cftr.py']
for script in scripts:
    print(f'Running {script}', flush=True)
    subprocess.run([sys.executable, str(ROOT/'code'/script)], cwd=ROOT, check=True)
