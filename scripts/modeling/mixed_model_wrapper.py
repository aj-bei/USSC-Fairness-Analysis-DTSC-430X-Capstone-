from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd


@dataclass
class MixedModelResult:
    success: bool
    formula: str
    family: str
    link: Optional[str]
    engine: str
    n_input: int
    n_used: Optional[int]
    n_dropped: Optional[int]
    converged: Optional[bool]
    singular: Optional[bool]
    fixed_effects: pd.DataFrame
    random_effects_variance: pd.DataFrame
    random_effects: pd.DataFrame
    fit_statistics: dict[str, Any]
    diagnostics: dict[str, Any]
    warnings: list[str]
    errors: list[str]
    raw_summary: Optional[str]
    stdout: Optional[str]
    stderr: Optional[str]
    config: dict[str, Any]


class MixedModelError(Exception):
    """Raised when the mixed model running fails."""


def fit_mixed_model(
    data: pd.DataFrame,
    formula: str,
    family: str = "gaussian",
    link: Optional[str] = None,
    categorical_cols: Optional[list[str]] = None,
    reference_levels: Optional[dict[str, str]] = None,
    return_confint: bool = True,
    return_random_effects_variance: bool = True,
    return_random_effects: bool = False,
    return_fitted: bool = False,
    return_residuals: bool = False,
    keep_raw_summary: bool = True,
    optimizer: Optional[str] = None,
    nAGQ: int = 1,
    r_script_path: str | Path = "scripts/r/fit_mixed_model.R",
    r_executable: str = "Rscript",
) -> MixedModelResult:
    _validate_inputs(
        data=data,
        formula=formula,
        family=family,
        link=link,
        categorical_cols=categorical_cols,
        reference_levels=reference_levels,
        nAGQ=nAGQ,
        r_script_path=r_script_path,
        r_executable=r_executable,
    )

    categorical_cols = categorical_cols or []
    reference_levels = reference_levels or {}

    with tempfile.TemporaryDirectory(prefix="mixed_model_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        data_path = tmpdir_path / "model_data.csv"
        config_path = tmpdir_path / "config.json"
        result_path = tmpdir_path / "result.json"

        data.to_csv(data_path, index=False)

        config = {
            "formula": formula,
            "family": family,
            "link": link,
            "engine": "lme4",
            "data_path": str(data_path.resolve()),
            "result_path": str(result_path.resolve()),
            "categorical_cols": categorical_cols,
            "reference_levels": reference_levels,
            "return_confint": return_confint,
            "return_random_effects_variance": return_random_effects_variance,
            "return_random_effects": return_random_effects,
            "return_fitted": return_fitted,
            "return_residuals": return_residuals,
            "keep_raw_summary": keep_raw_summary,
            "optimizer": optimizer,
            "nAGQ": nAGQ,
            "n_input": int(len(data)),
        }

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print("Fitting mixed model with formula:", formula, "and family:", family, "using R backend...")

        completed = _run_r_backend(
            r_executable=r_executable,
            r_script_path=r_script_path,
            config_path=config_path,
        )

        if not result_path.exists():
            raise MixedModelError(
                "R script finished but no result.json was produced.\n"
                f"STDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
            )

        # load results generated from R into json
        with open(result_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        warnings_list = _ensure_list(payload.get("warnings"))
        errors_list = _ensure_list(payload.get("errors"))

        if not payload.get("success", False):
            raise MixedModelError(
                "Model fitting failed in R.\n"
                f"Errors: {errors_list}\n"
                f"Warnings: {warnings_list}\n"
                f"STDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
            )

        return MixedModelResult(
            success=payload["success"],
            formula=payload["formula"],
            family=payload["family"],
            link=payload.get("link"),
            engine=payload["engine"],
            n_input=payload["n_input"],
            n_used=payload.get("n_used"),
            n_dropped=payload.get("n_dropped"),
            converged=payload.get("converged"),
            singular=payload.get("singular"),
            fixed_effects=pd.DataFrame.from_records(payload.get("fixed_effects", [])),
            random_effects_variance=pd.DataFrame.from_records(
                payload.get("random_effects_variance", [])
            ),
            random_effects=pd.DataFrame.from_records(
                payload.get("random_effects", [])
            ),
            fit_statistics=payload.get("fit_statistics", {}),
            diagnostics=payload.get("diagnostics", {}),
            warnings=warnings_list,
            errors=errors_list,
            raw_summary=payload.get("raw_summary"),
            stdout=completed.stdout,
            stderr=completed.stderr,
            config=config,
        )


def _ensure_list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(item) for item in x]
    return [str(x)]


def _validate_inputs(
    data: pd.DataFrame,
    formula: str,
    family: str,
    link: Optional[str],
    categorical_cols: Optional[list[str]],
    reference_levels: Optional[dict[str, str]],
    nAGQ: int,
    r_script_path: str | Path,
    r_executable: str,
) -> None:

    if data.empty:
        raise ValueError("`data` is empty.")

    allowed_families = {"gaussian", "binomial"}
    if family not in allowed_families:
        raise ValueError(f"`family` must be one of {allowed_families}, got {family!r}.")

    allowed_binomial_links = {"logit", "probit", "cloglog", "cauchit", "log"}
    if family == "gaussian" and link is not None:
        raise ValueError("For now, `link` must be None when family='gaussian'.")
    if family == "binomial" and link is not None and link not in allowed_binomial_links:
        raise ValueError(
            f"For binomial models, `link` must be one of {allowed_binomial_links}, got {link!r}."
        )

    categorical_cols = categorical_cols or []
    missing_cat = [col for col in categorical_cols if col not in data.columns]
    if missing_cat:
        raise ValueError(f"Categorical columns not found in data: {missing_cat}")

    reference_levels = reference_levels or {}
    for col, ref in reference_levels.items():
        if col not in data.columns:
            raise ValueError(f"Reference level column {col!r} not found in dataframe.")
        if ref not in set(data[col].dropna().astype(str)):
            raise ValueError(f"Reference level {ref!r} not found in column {col!r}.")

    if not isinstance(nAGQ, int) or nAGQ < 0:
        raise ValueError("`nAGQ` must be a nonnegative integer.")

    r_script_path = Path(r_script_path)

    if shutil.which(r_executable) is None:
        raise FileNotFoundError(f"Could not find R executable {r_executable!r} on PATH.")

    if not r_script_path.exists():
        raise FileNotFoundError(f"R script not found at: {r_script_path}")


def _run_r_backend(
    r_executable: str,
    r_script_path: str | Path,
    config_path: Path,
) -> subprocess.CompletedProcess:
    
    # build command to run R script with command line args
    cmd = [
        r_executable,
        str(Path(r_script_path).resolve()),
        str(config_path.resolve()),
    ]

    # run the R script and capture output
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise MixedModelError(
            "R backend returned a non-zero exit code.\n"
            f"Exit code: {completed.returncode}\n"
            f"STDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
        )

    return completed