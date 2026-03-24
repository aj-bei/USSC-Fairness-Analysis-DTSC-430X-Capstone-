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
    random_effects_covariance_matrices: dict[str, pd.DataFrame]
    fit_statistics: dict[str, Any]
    diagnostics: dict[str, Any]
    warnings: list[str]
    errors: list[str]
    raw_summary: Optional[str]
    stdout: Optional[str]
    stderr: Optional[str]
    config: dict[str, Any]


class MixedModelError(Exception):
    """Raised when the mixed model pipeline fails."""


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
    return_random_effects_covariance: bool = False,
    return_fitted: bool = False,
    return_residuals: bool = False,
    keep_raw_summary: bool = True,
    optimizer: Optional[str] = None,
    nAGQ: int = 1,
    r_script_path: str | Path = "scripts/r/fit_mixed_model.R",
    r_executable: str = "Rscript",
) -> MixedModelResult:
    """
    Fit a mixed-effects model in R via lme4/lmerTest and return structured output.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    formula : str
        R-style model formula.
    family : str
        Supported: 'gaussian', 'binomial'
    link : str | None
        Binomial link, e.g. 'logit', 'probit', 'cloglog'.
    categorical_cols : list[str] | None
        Columns to coerce to factors in R.
    reference_levels : dict[str, str] | None
        Mapping from factor column to reference level.
    return_confint : bool
        Included in config for future extension.
    return_random_effects_variance : bool
        Whether to return variance component output from broom.mixed::tidy(..., effects="ran_pars").
    return_random_effects : bool
        Whether to return BLUPs / conditional modes from ranef(model).
    return_random_effects_covariance : bool
        Whether to return covariance matrices for random effects, exposed as a dict of DataFrames.
    return_fitted : bool
        Whether to return the first few fitted values in diagnostics.
    return_residuals : bool
        Whether to return the first few residuals in diagnostics.
    keep_raw_summary : bool
        Whether to return the plain-text R summary.
    optimizer : str | None
        Optional optimizer passed to lme4 control.
    nAGQ : int
        Number of adaptive Gauss-Hermite quadrature points for glmer.
    r_script_path : str | Path
        Path to the R backend script.
    r_executable : str
        Name or full path of the Rscript executable.

    Returns
    -------
    MixedModelResult
    """
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
            "return_random_effects_covariance": return_random_effects_covariance,
            "return_fitted": return_fitted,
            "return_residuals": return_residuals,
            "keep_raw_summary": keep_raw_summary,
            "optimizer": optimizer,
            "nAGQ": nAGQ,
            "n_input": int(len(data)),
        }

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

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

        fixed_effects_df = _records_to_df(
            payload.get("fixed_effects", []),
            expected_cols=["effect", "term", "estimate", "std.error", "statistic", "p.value"],
        )

        random_effects_variance_df = _records_to_df(
            payload.get("random_effects_variance", []),
        )

        random_effects_df = _records_to_df(
            payload.get("random_effects", []),
            expected_cols=["group", "level", "term", "estimate", "conditional_estimate"],
        )

        covariance_records = payload.get("random_effects_covariance", [])
        covariance_long_df = pd.DataFrame.from_records(covariance_records or [])
        covariance_matrices = _covariance_df_to_matrices(covariance_long_df)

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
            fixed_effects=fixed_effects_df,
            random_effects_variance=random_effects_variance_df,
            random_effects=random_effects_df,
            random_effects_covariance_matrices=covariance_matrices,
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


def _records_to_df(records: Any, expected_cols: Optional[list[str]] = None) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records or [])
    if expected_cols:
        for col in expected_cols:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[expected_cols]
    return df


def _covariance_df_to_matrices(cov_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Convert long-format covariance rows into a dictionary of covariance matrices.

    Expected input columns:
        - group
        - term1
        - term2
        - covariance
    """
    if cov_df.empty:
        return {}

    required_cols = {"group", "term1", "term2", "covariance"}
    missing = required_cols - set(cov_df.columns)
    if missing:
        raise ValueError(
            f"Covariance DataFrame is missing required columns: {sorted(missing)}"
        )

    matrices: dict[str, pd.DataFrame] = {}

    for group_name, sub_df in cov_df.groupby("group", dropna=False):
        # preserve natural order from the long table
        terms = list(dict.fromkeys(sub_df["term1"].tolist() + sub_df["term2"].tolist()))

        mat = sub_df.pivot(index="term1", columns="term2", values="covariance")
        mat = mat.reindex(index=terms, columns=terms)

        matrices[str(group_name)] = mat

    return matrices


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
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")

    if data.empty:
        raise ValueError("`data` is empty.")

    if not isinstance(formula, str) or "~" not in formula:
        raise ValueError("`formula` must be a valid R-style formula string containing '~'.")

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
    cmd = [
        r_executable,
        str(Path(r_script_path).resolve()),
        str(config_path.resolve()),
    ]

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