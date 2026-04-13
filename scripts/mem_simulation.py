import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning


"""
This script simulates grouped data with a known true effect of a "race" variable on an outcome, 
and then tests the type I error of three different modeling approaches for testing. This simulation
is used as evidence supporting the choice of a mixed model in the final report.

Note, this code was created through meticulous prompts with ChatGPT.
"""

# -----------------------------
# USER-CONTROLLABLE PARAMETERS
# -----------------------------
N_SIMS = 100
N_GROUPS = 90
OBS_PER_GROUP = 80          # can change this
SIGMA_EPSILON = 10.0        # residual SD
ALPHA = 0.05
SEED = 65

# True fixed effects
BETA_0 = 0.0               # fixed intercept
BETA_SEVERITY = 2.0        # true positive effect
BETA_RACE = 0.5            # true null effect

# Random effects covariance matrix:
# [random intercept variance, covariance;
#  covariance, random race-slope variance]
RANEF_COV = np.array([
    [100.0, 0.0],
    [0.0, 25.0]
])

# Race probability (Black = 1)
P_BLACK = 0.35

# Severity distribution (positive continuous)
SEVERITY_SHAPE = 2.0
SEVERITY_SCALE = 2.0

rng = np.random.default_rng(SEED)


def simulate_one_dataset(
    n_groups=N_GROUPS,
    obs_per_group=OBS_PER_GROUP,
    beta_0=BETA_0,
    beta_severity=BETA_SEVERITY,
    beta_race=BETA_RACE,
    sigma_epsilon=SIGMA_EPSILON,
    ranef_cov=RANEF_COV,
    p_black=P_BLACK,
    severity_shape=SEVERITY_SHAPE,
    severity_scale=SEVERITY_SCALE,
    rng=None
):
    """
    Simulate grouped sentencing-like data:
      Y = beta_0 + beta_severity*severity + beta_race*race
          + b0_j + b1_j*race + epsilon

    where:
      b0_j = random intercept for group j
      b1_j = random slope for race for group j
      epsilon ~ N(0, sigma_epsilon^2)
    """
    if rng is None:
        rng = np.random.default_rng()

    # One random intercept and random race slope per group
    ranefs = rng.multivariate_normal(
        mean=[0.0, 0.0],
        cov=ranef_cov,
        size=n_groups
    )

    rows = []

    for g in range(n_groups):
        b0_j, b1_j = ranefs[g]

        severity = rng.gamma(shape=severity_shape, scale=severity_scale, size=obs_per_group)
        race = rng.binomial(1, p_black, size=obs_per_group)
        epsilon = rng.normal(0.0, sigma_epsilon, size=obs_per_group)

        y = (
            beta_0
            + beta_severity * severity
            + beta_race * race
            + b0_j
            + b1_j * race
            + epsilon
        )

        group_df = pd.DataFrame({
            "group": g,
            "severity": severity,
            "race": race,
            "Y": y
        })
        rows.append(group_df)

    df = pd.concat(rows, ignore_index=True)
    df["group"] = df["group"].astype("category")
    return df


def fit_mixed_model_race_pvalue(df):
    """
    Mixed model with random intercept and random slope for race by group.
    Tests the fixed effect for race using the model's Wald p-value.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)

        model = smf.mixedlm(
            "Y ~ severity + race",
            data=df,
            groups=df["group"],
            re_formula="1 + race"
        )
        result = model.fit(reml=False, method="lbfgs", disp=False)

    return result.pvalues["race"]


def fit_ols_race_pvalue(df):
    """
    Plain OLS ignoring grouping.
    Tests the coefficient on race.
    """
    model = smf.ols("Y ~ severity + race", data=df).fit()
    return model.pvalues["race"]


def fit_ols_interaction_joint_pvalue(df):
    """
    OLS with group fixed effects and group-specific race interactions:
        Y ~ severity + C(group) + race + C(group):race

    Since there is no single common race effect here once interactions are added,
    test the JOINT null that all race-related coefficients are zero:
        race = 0
        and every group-by-race interaction = 0
    """
    model = smf.ols("Y ~ severity + C(group) + race + C(group):race", data=df).fit()

    param_names = model.params.index.tolist()
    race_terms = [name for name in param_names if name == "race" or ":race" in name]

    if not race_terms:
        raise ValueError("No race-related terms found in OLS interaction model.")

    restriction = " = 0, ".join(race_terms) + " = 0"
    ftest = model.f_test(restriction)

    return float(ftest.pvalue)


def run_simulation(
    n_sims=N_SIMS,
    alpha=ALPHA,
    seed=SEED
):
    sim_rng = np.random.default_rng(seed)

    mixed_rejections = 0
    ols_rejections = 0
    ols_interaction_rejections = 0

    mixed_failures = 0
    ols_failures = 0
    ols_interaction_failures = 0

    completed_sims = 0

    for sim in range(n_sims):
        df = simulate_one_dataset(
            rng=sim_rng
        )

        sim_success = True

        # Mixed model
        try:
            p_mixed = fit_mixed_model_race_pvalue(df)
            if p_mixed > alpha:
                mixed_rejections += 1
        except Exception:
            mixed_failures += 1
            sim_success = False

        # Plain OLS
        try:
            p_ols = fit_ols_race_pvalue(df)
            if p_ols > alpha:
                ols_rejections += 1
        except Exception:
            ols_failures += 1
            sim_success = False

        # OLS with group interactions
        try:
            p_ols_int = fit_ols_interaction_joint_pvalue(df)
            if p_ols_int > alpha:
                ols_interaction_rejections += 1
        except Exception:
            ols_interaction_failures += 1
            sim_success = False

        if sim_success:
            completed_sims += 1

    print("\n===== SIMULATION SETTINGS =====")
    print(f"Requested simulations: {n_sims}")
    print(f"Groups per simulation: {N_GROUPS}")
    print(f"Observations per group: {OBS_PER_GROUP}")
    print(f"Total observations per simulation: {N_GROUPS * OBS_PER_GROUP}")
    print(f"True fixed intercept: {BETA_0}")
    print(f"True fixed effect of severity: {BETA_SEVERITY}")
    print(f"True fixed effect of race: {BETA_RACE}")
    print(f"Residual SD (sigma_epsilon): {SIGMA_EPSILON}")
    print(f"Random effects covariance matrix:\n{RANEF_COV}")
    print(f"Alpha level: {alpha}")

    print("\n===== MODEL FAILURES =====")
    print(f"Mixed model failures: {mixed_failures}")
    print(f"OLS failures: {ols_failures}")
    print(f"OLS + interaction failures: {ols_interaction_failures}")

    print("\n===== TYPE I ERROR RESULTS =====")
    print(f"Simulations completed with all 3 models fit: {completed_sims}")

    if n_sims > 0:
        print(
            f"Mixed model: rejected H0 in {mixed_rejections} / {n_sims} "
            f"simulations ({mixed_rejections / n_sims:.3f})"
        )
        print(
            f"Plain OLS: rejected H0 in {ols_rejections} / {n_sims} "
            f"simulations ({ols_rejections / n_sims:.3f})"
        )
        print(
            f"OLS + interactions: rejected H0 in {ols_interaction_rejections} / {n_sims} "
            f"simulations ({ols_interaction_rejections / n_sims:.3f})"
        )


if __name__ == "__main__":
    run_simulation()