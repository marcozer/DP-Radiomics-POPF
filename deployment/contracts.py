"""
Deployment-time CLI contracts for sibling repos.

These contracts are intentionally "help-text" checks (not imports) so that the
deployment surface stays stable even when internal code changes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScriptContract:
    repo: str  # "radiomics" | "analysis"
    rel_path: str
    required_tokens: tuple[str, ...]


CONTRACTS: tuple[ScriptContract, ...] = (
    ScriptContract(
        repo="radiomics",
        rel_path="code/extract_pancreatic_head_from_viewer_coordinates.py",
        required_tokens=("--ct-dir", "--seg-dir", "--coordinates-file", "--output-dir", "--patient-id"),
    ),
    ScriptContract(
        repo="radiomics",
        rel_path="code/extract_ct_from_head_segmentations.py",
        required_tokens=("--ct-dir", "--head-dir", "--output-dir"),
    ),
    ScriptContract(
        repo="radiomics",
        rel_path="code/extract_radiomics_yaml.py",
        required_tokens=("--config", "--input-dir", "--output-dir"),
    ),
    ScriptContract(
        repo="radiomics",
        rel_path="code/combat_apply.py",
        required_tokens=("--features-csv", "--estimates-pkl", "--output-csv"),
    ),
    ScriptContract(
        repo="analysis",
        rel_path="code/predict_popf_risk.py",
        required_tokens=("--model-pkl", "--features-csv", "--output-csv"),
    ),
)
