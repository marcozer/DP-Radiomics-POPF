# Environment & Dependency Capture

1. **Create an isolated interpreter**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip wheel setuptools
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r publication_export/requirements.txt
   ```
   - Optional extras: `pip install neurocombat-sklearn optuna` if discovery harmonization and LR C tuning are required.
   - Install STABL from git (required): `pip install git+https://github.com/gregbellan/Stabl.git@<commit>`

3. **Verify STABL installation**
   - Verify installation with `python -c "import stabl; print(stabl.__version__)"`.

4. **Record system information**
   - Capture `python --version`, `pip list`, and `uname -a` outputs in `results_reference/dev_run/system_info.txt` for reproducibility.

5. **Optional GPU/BLAS tuning**
   - If using xgboost/lightgbm GPU builds, document CUDA/cuDNN versions separately.

6. **Testing sanity**
   - Run `pytest -q` (future home for targeted unit tests under `tests/`).
   - Execute `python scripts/check_alignment.py --radiomics-path data/... --matches-path data/POPF-SCANNER.csv --out results/alignment_audit` to ensure IO dependencies are satisfied.
