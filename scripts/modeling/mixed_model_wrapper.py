from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import classification_report, roc_auc_score, roc_curve


@dataclass
class MixedModelResult:
    success: bool
    formula: str
    family: str
    link: Optional[str]
    offset: Optional[str]
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
    offset: Optional[str]
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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import classification_report, roc_auc_score, roc_curve


def _binned_quantiles(
    x: pd.Series,
    y: pd.Series,
    n_bins: int = 40,
    quantiles=(0.1, 0.5, 0.9)
) -> pd.DataFrame:
    """
    Bin x into quantile-based bins and compute requested quantiles of y in each bin.
    """
    df = pd.DataFrame({"x": x, "y": y}).dropna().copy()

    # qcut can fail if many duplicate x values; duplicates='drop' handles that
    df["bin"] = pd.qcut(df["x"], q=n_bins, duplicates="drop")

    out = (
        df.groupby("bin", observed=False)
          .apply(
              lambda g: pd.Series(
                  {
                      "x_mid": g["x"].median(),
                      **{f"q{int(q*100)}": g["y"].quantile(q) for q in quantiles},
                      "n": len(g),
                  }
              )
          )
          .reset_index(drop=True)
          .sort_values("x_mid")
    )
    return out


def plot_diagnostics(
    result,
    threshold: float = 0.5,
    max_scatter_points: int = 12000,
    n_bins: int = 40,
    random_state: int = 42,
) -> None:
    family = result.family

    sns.set_style("whitegrid")

    if family == "gaussian":
        if "residuals" not in result.diagnostics or "fitted" not in result.diagnostics:
            print(
                "No residuals or fitted values available. "
                "Fit with return_fitted=True and return_residuals=True."
            )
            return

        # resid = pd.Series(result.diagnostics["residuals"]).dropna().astype(float)
        # fitted = pd.Series(result.diagnostics["fitted"]).dropna().astype(float)

        # n = min(len(resid), len(fitted))
        # resid = resid.iloc[:n]
        # fitted = fitted.iloc[:n]

        # # subsample for visibility if needed
        # rng = np.random.default_rng(random_state)
        # if n > max_scatter_points:
        #     idx = rng.choice(n, size=max_scatter_points, replace=False)
        #     resid_scatter = resid.iloc[idx]
        #     fitted_scatter = fitted.iloc[idx]
        # else:
        #     resid_scatter = resid
        #     fitted_scatter = fitted

        # bq = _binned_quantiles(fitted, resid, n_bins=n_bins, quantiles=(0.1, 0.5, 0.9))

        # fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        # # --- Residuals vs fitted: hexbin + binned quantiles
        # hb = axes[0].hexbin(
        #     fitted,
        #     resid,
        #     gridsize=55,
        #     mincnt=1,
        #     cmap="viridis",
        # )
        # fig.colorbar(hb, ax=axes[0], label="Count")
        # axes[0].scatter(
        #     fitted_scatter,
        #     resid_scatter,
        #     s=6,
        #     alpha=0.08,
        #     edgecolors="none",
        #     rasterized=True,
        # )
        # axes[0].plot(bq["x_mid"], bq["q50"], linewidth=2, label="Median residual")
        # axes[0].plot(bq["x_mid"], bq["q10"], linestyle="--", linewidth=1.5, label="10th / 90th pct")
        # axes[0].plot(bq["x_mid"], bq["q90"], linestyle="--", linewidth=1.5)
        # axes[0].axhline(0, color="red", linestyle="--", linewidth=1.2)
        # axes[0].set_title("Residuals vs Fitted")
        # axes[0].set_xlabel("Fitted values")
        # axes[0].set_ylabel("Residuals")
        # axes[0].legend(frameon=True)

        # # --- Histogram
        # sns.histplot(resid, kde=True, ax=axes[1], bins=50, stat="density")
        # axes[1].set_title("Histogram of Residuals")
        # axes[1].set_xlabel("Residual value")

        # # --- QQ plot
        # sm.qqplot(resid, line="45", ax=axes[2])
        # axes[2].set_title("QQ Plot of Residuals")

        # plt.tight_layout()
        # plt.show()
        # return

        resid = result.diagnostics["residuals"]
        fitted = result.diagnostics["fitted"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        sns.scatterplot(x=fitted, y=resid, ax=axes[0], alpha=0.09)
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
  
    if family == "binomial":
        required = {"predicted_prob", "observed_response"}
        if not required.issubset(result.diagnostics):
            print(
                "No predicted probabilities / observed responses available. "
                "Fit with return_fitted=True."
            )
            return

        y_true = pd.Series(result.diagnostics["observed_response"]).dropna().astype(int)
        y_prob = pd.Series(result.diagnostics["predicted_prob"]).dropna().astype(float)

        n = min(len(y_true), len(y_prob))
        y_true = y_true.iloc[:n]
        y_prob = y_prob.iloc[:n]
        y_pred = (y_prob >= threshold).astype(int)

        print("Classification Report")
        print(classification_report(y_true, y_pred, digits=4))

        try:
            auc = roc_auc_score(y_true, y_prob)
            fpr, tpr, _ = roc_curve(y_true, y_prob)

            plt.figure(figsize=(7, 5))
            plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.4f})")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.show()
        except Exception as exc:
            print(f"Could not compute ROC/AUC: {exc}")
        return

    if family in {"gamma", "negative_binomial"}:
        required = {"dharma_scaled_residuals", "dharma_fitted_predicted"}
        if not required.issubset(result.diagnostics):
            print(
                "No DHARMa simulated residuals available. "
                "Fit with return_fitted=True or return_residuals=True and ensure DHARMa is installed in R."
            )
            return

        sim_resid = pd.Series(result.diagnostics["dharma_scaled_residuals"]).dropna().astype(float)
        fitted = pd.Series(result.diagnostics["dharma_fitted_predicted"]).dropna().astype(float)

        n = min(len(sim_resid), len(fitted))
        sim_resid = sim_resid.iloc[:n]
        fitted = fitted.iloc[:n]

        family_title = "Gamma" if family == "gamma" else "Negative Binomial"

        rng = np.random.default_rng(random_state)
        if n > max_scatter_points:
            idx = rng.choice(n, size=max_scatter_points, replace=False)
            sim_resid_scatter = sim_resid.iloc[idx]
            fitted_scatter = fitted.iloc[idx]
        else:
            sim_resid_scatter = sim_resid
            fitted_scatter = fitted

        # For DHARMa, quantile bands are more useful than raw scatter
        bq = _binned_quantiles(fitted, sim_resid, n_bins=n_bins, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        # --- Residuals vs fitted: hexbin + sample scatter + quantile bands
        hb = axes[0].hexbin(
            fitted,
            sim_resid,
            gridsize=55,
            mincnt=1,
            cmap="viridis",
        )
        fig.colorbar(hb, ax=axes[0], label="Count")

        axes[0].scatter(
            fitted_scatter,
            sim_resid_scatter,
            s=5,
            alpha=0.03,
            edgecolors="none",
            rasterized=True,
        )

        axes[0].plot(bq["x_mid"], bq["q50"], linewidth=2, label="Median")
        axes[0].plot(bq["x_mid"], bq["q25"], linestyle="--", linewidth=1.5, label="25th / 75th pct")
        axes[0].plot(bq["x_mid"], bq["q75"], linestyle="--", linewidth=1.5)
        axes[0].plot(bq["x_mid"], bq["q10"], linestyle=":", linewidth=1.3, label="10th / 90th pct")
        axes[0].plot(bq["x_mid"], bq["q90"], linestyle=":", linewidth=1.3)

        axes[0].axhline(0.5, color="red", linestyle="--", linewidth=1.2)
        axes[0].axhline(0.25, color="gray", linestyle=":", linewidth=0.9, alpha=0.8)
        axes[0].axhline(0.75, color="gray", linestyle=":", linewidth=0.9, alpha=0.8)
        axes[0].set_ylim(-0.02, 1.02)
        axes[0].set_title(f"{family_title} DHARMa Residuals vs Fitted")
        axes[0].set_xlabel("Predicted values")
        axes[0].set_ylabel("Scaled residuals")
        axes[0].legend(frameon=True, loc="best")

        # --- Histogram with uniform reference line
        sns.histplot(sim_resid, kde=False, stat="density", bins=30, ax=axes[1])
        axes[1].axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="Uniform density")
        axes[1].set_xlim(0, 1)
        axes[1].set_title(f"Histogram of {family_title} DHARMa Residuals")
        axes[1].set_xlabel("Scaled residual")
        axes[1].set_ylabel("Density")
        axes[1].legend(frameon=True)

        # --- QQ plot against Uniform(0,1)
        sm.qqplot(sim_resid, dist=stats.uniform, line="45", ax=axes[2])
        axes[2].set_title(f"QQ Plot of {family_title} DHARMa Residuals")
        axes[2].set_xlabel("Theoretical Quantiles")
        axes[2].set_ylabel("Sample Quantiles")

        plt.tight_layout()
        plt.show()

        dharma_tests = result.diagnostics.get("dharma_tests")
        if dharma_tests:
            print("DHARMa Tests")
            for test_name, test_info in dharma_tests.items():
                print(f"{test_name}: {test_info}")
        return

    print(f"No diagnostics plotting implemented for family={result.family!r}.")

def fit_mixed_model(
    data: pd.DataFrame,
    formula: str,
    family: str = "gaussian",
    offset: str = None,
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
        offset=offset,
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
            "offset": offset,
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

        print(f"Fitting mixed model with formula: {formula}, family: {family}, optimizer: {optimizer}...")
        start = time.time()

        completed = _run_r_backend(
            r_executable=r_executable,
            r_script_path=r_script_path,
            config_path=config_path,
        )

        print(f"Seconds taken to fit mixed model: {round(time.time() - start, 2)}")

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

        random_effects_variance_df = _records_to_df(payload.get("random_effects_variance", []))

        random_effects_df = _records_to_df(
            payload.get("random_effects", []),
            expected_cols=["group", "level", "term", "estimate", "conditional_estimate"],
        )

        covariance_records = payload.get("random_effects_covariance", [])
        covariance_long_df = pd.DataFrame.from_records(covariance_records or [])
        covariance_matrices = _covariance_df_to_matrices(covariance_long_df)

        diagnostics = payload.get("diagnostics", {})

        for key in [
            "residuals",
            "fitted",
            "predicted_prob",
            "observed_response",
            "dharma_scaled_residuals",
            "dharma_fitted_predicted",
        ]:
            if key in diagnostics:
                diagnostics[key] = pd.Series(diagnostics[key], name=key)

        return MixedModelResult(
            success=payload["success"],
            formula=payload["formula"],
            family=payload["family"],
            link=payload.get("link"),
            offset=payload.get("offset"),
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
    offset: Optional[str],
) -> None:
    _validate_formula(formula)

    allowed_families = {"gaussian", "binomial", "gamma", "negative_binomial"}
    if family not in allowed_families:
        raise ValueError(f"`family` must be one of {allowed_families}, got {family!r}.")

    allowed_binomial_links = {"logit", "probit", "cloglog", "cauchit", "log"}
    allowed_gamma_links = {"inverse", "identity", "log"}

    if family == "gaussian" and link is not None:
        raise ValueError("For gaussian models, `link` must be None.")

    if family == "binomial":
        if link is not None and link not in allowed_binomial_links:
            raise ValueError(
                f"For binomial models, `link` must be one of {allowed_binomial_links}, got {link!r}."
            )

    if family == "gamma":
        if link is not None and link not in allowed_gamma_links:
            raise ValueError(
                f"For gamma models, `link` must be one of {allowed_gamma_links}, got {link!r}."
            )

    if family == "negative_binomial" and link is not None:
        raise ValueError(
            "Negative binomial mixed models in this wrapper use lme4::glmer.nb(), "
            "so `link` must be None."
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
    
    if offset is not None and offset not in data.columns:
        raise ValueError(f"Offset column {offset!r} not found in dataframe!")


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