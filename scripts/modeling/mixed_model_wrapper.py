from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import time

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


@dataclass
class MixedModelResult:
    success: bool
    formula: str
    family: str
    link: Optional[str]
    engine: str
    n_input: int
    n_written: int
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

    def print_summary(self) -> None:
        if self.raw_summary:
            print(self.raw_summary)
        else:
            print("No summary available.")


@dataclass
class MixedModelANOVAResult:
    success: bool
    formula_null: str
    formula_alt: str
    family: str
    link: Optional[str]
    test_type: str
    reml_used: Optional[bool]
    model_null_fit: dict[str, Any]
    model_alt_fit: dict[str, Any]
    comparison_table: pd.DataFrame
    test: dict[str, Any]
    warnings: list[str]
    errors: list[str]
    raw_anova: Optional[str]
    stdout: Optional[str]
    stderr: Optional[str]
    config: dict[str, Any]

    def print_anova(self) -> None:
        if self.raw_anova:
            print(self.raw_anova)
        else:
            print("No ANOVA output available.")


class MixedModelError(Exception):
    """Raised when the mixed model pipeline fails."""

def plot_diagnostics(result: MixedModelResult) -> None:

    """
    Plot basic diagnostics for a fitted mixed model, if available in the result.diagnostics.
        - Residuals vs Fitted: to check for non-linearity, heteroscedasticity, and outliers.
        - Histogram of Residuals: to check for normality of residuals.
        - QQ Plot of Residuals: to check for normality of residuals and identify deviations in the tails.
    """

    if "residuals" in result.diagnostics and "fitted" in result.diagnostics:
        resid = result.diagnostics["residuals"]
        fitted = result.diagnostics["fitted"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        sns.scatterplot(x=fitted, y=resid, ax=axes[0], alpha=0.4)
        axes[0].axhline(0, color="red", linestyle="--")
        axes[0].set_title("Residuals vs Fitted")
        axes[0].set_xlabel("Fitted values")
        axes[0].set_ylabel("Residuals")

        sns.histplot(resid, kde=True, ax=axes[1])
        axes[1].set_title("Histogram of Residuals")
        axes[1].set_xlabel("Residual value")

        sm.qqplot(resid, line="45", ax=axes[2])
        axes[2].set_title("QQ Plot of Residuals")

        plt.tight_layout()
        plt.show()
    else:
        print("No residuals or fitted values available for diagnostics. Please ensure `return_fitted=True` and `return_residuals=True` when fitting the model.")

def fit_mixed_model(
    data: pd.DataFrame,
    formula: str,
    family: str = "gaussian",
    link: Optional[str] = None,
    categorical_cols: Optional[list[str]] = None,
    reference_levels: Optional[dict[str, str]] = None,
    return_confint: bool = False,
    return_random_effects_variance: bool = True,
    return_random_effects: bool = False,
    return_random_effects_covariance: bool = False,
    return_fitted: bool = False,
    return_residuals: bool = False,
    keep_raw_summary: bool = True,
    optimizer: Optional[str] = None,
    nAGQ: int = 1,
    reml: bool = True,
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
            "mode": "fit",
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
            "n_written": int(len(data.columns)),
            "reml": bool(reml) if family == "gaussian" else None,
        }

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print(f"Fitting mixed model with formula: {formula}, family: {family}, and optimizer: {optimizer}...")
        time1 = time.time()

        completed = _run_r_backend(
            r_executable=r_executable,
            r_script_path=r_script_path,
            config_path=config_path,
        )

        time2 = time.time()
        print(f"Seconds taken to fit mixed model: {round(time2 - time1, 2)}")

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

        diagnostics = payload.get("diagnostics", {})
        if "residuals" in diagnostics:
            diagnostics["residuals"] = pd.Series(diagnostics["residuals"], name="residual")
        if "fitted" in diagnostics:
            diagnostics["fitted"] = pd.Series(diagnostics["fitted"], name="fitted")

        return MixedModelResult(
            success=payload["success"],
            formula=payload["formula"],
            family=payload["family"],
            link=payload.get("link"),
            engine=payload["engine"],
            n_input=payload["n_input"],
            n_written=payload.get("n_written", len(data.columns)),
            n_used=payload.get("n_used"),
            n_dropped=payload.get("n_dropped"),
            converged=payload.get("converged"),
            singular=payload.get("singular"),
            fixed_effects=fixed_effects_df,
            random_effects_variance=random_effects_variance_df,
            random_effects=random_effects_df,
            random_effects_covariance_matrices=covariance_matrices,
            fit_statistics=payload.get("fit_statistics", {}),
            diagnostics=diagnostics,
            warnings=warnings_list,
            errors=errors_list,
            raw_summary=payload.get("raw_summary"),
            stdout=completed.stdout,
            stderr=completed.stderr,
            config=config,
        )


def anova_mixed_models(
    data: pd.DataFrame,
    null_formula: str,
    alt_formula: str,
    family: str = "gaussian",
    link: Optional[str] = None,
    categorical_cols: Optional[list[str]] = None,
    reference_levels: Optional[dict[str, str]] = None,
    optimizer: Optional[str] = None,
    nAGQ: int = 1,
    test_type: str = "ml",
    r_script_path: str | Path = "scripts/r/fit_mixed_model.R",
    r_executable: str = "Rscript",
) -> MixedModelANOVAResult:
    _validate_inputs(
        data=data,
        formula=null_formula,
        family=family,
        link=link,
        categorical_cols=categorical_cols,
        reference_levels=reference_levels,
        nAGQ=nAGQ,
        r_script_path=r_script_path,
        r_executable=r_executable,
    )
    _validate_formula(alt_formula)
    _validate_test_type(test_type=test_type, family=family)

    categorical_cols = categorical_cols or []
    reference_levels = reference_levels or {}

    with tempfile.TemporaryDirectory(prefix="mixed_model_anova_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        data_path = tmpdir_path / "model_data.csv"
        config_path = tmpdir_path / "config.json"
        result_path = tmpdir_path / "result.json"

        data.to_csv(data_path, index=False)

        config = {
            "mode": "anova",
            "formula_null": null_formula,
            "formula_alt": alt_formula,
            "family": family,
            "link": link,
            "engine": "lme4",
            "data_path": str(data_path.resolve()),
            "result_path": str(result_path.resolve()),
            "categorical_cols": categorical_cols,
            "reference_levels": reference_levels,
            "optimizer": optimizer,
            "nAGQ": nAGQ,
            "n_input": int(len(data)),
            "n_written": int(len(data.columns)),
            "test_type": test_type.lower(),
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
                "Mixed-model ANOVA failed in R.\n"
                f"Errors: {errors_list}\n"
                f"Warnings: {warnings_list}\n"
                f"STDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
            )

        comparison_table = _records_to_df(payload.get("comparison_table", []))

        return MixedModelANOVAResult(
            success=payload["success"],
            formula_null=payload["formula_null"],
            formula_alt=payload["formula_alt"],
            family=payload["family"],
            link=payload.get("link"),
            test_type=payload.get("test_type", test_type.lower()),
            reml_used=payload.get("reml_used"),
            model_null_fit=payload.get("model_null_fit", {}),
            model_alt_fit=payload.get("model_alt_fit", {}),
            comparison_table=comparison_table,
            test=payload.get("test", {}),
            warnings=warnings_list,
            errors=errors_list,
            raw_anova=payload.get("raw_anova"),
            stdout=completed.stdout,
            stderr=completed.stderr,
            config=config,
        )


def _validate_formula(formula: str) -> None:
    if not isinstance(formula, str) or "~" not in formula:
        raise ValueError("Formula must be a valid R-style formula string containing '~'.")


def _validate_test_type(test_type: str, family: str) -> None:
    allowed = {"ml", "reml"}
    tt = test_type.lower()
    if tt not in allowed:
        raise ValueError(f"`test_type` must be one of {allowed}, got {test_type!r}.")
    if family != "gaussian" and tt == "reml":
        raise ValueError("`test_type='reml'` is only valid for gaussian mixed models.")


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

    _validate_formula(formula)

    allowed_families = {"gaussian", "binomial"}
    if family not in allowed_families:
        raise ValueError(f"`family` must be one of {allowed_families}, got {family!r}.")

    allowed_binomial_links = {"logit", "probit", "cloglog", "cauchit", "log"}
    if family == "gaussian" and link is not None:
        raise ValueError("For gaussian models, `link` must be None.")
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