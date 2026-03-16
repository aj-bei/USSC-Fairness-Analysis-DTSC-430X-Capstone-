import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pointbiserialr

def cramers_v_correction(chi2: float, table: pd.DataFrame) -> float:
    """
    Cramers v has a tendency to overestimate the strength of association, so an adjustment
    is made to correct for the bias. The correction uses the number of categories for each variable
    and the total sample size to adjust the phi2 statistic before calculating Cramer's V.

    These formulas come from the wikipedia article on Cramer's V.
    """
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan

    n = table.to_numpy().sum()

    r, k = table.shape
    phi2 = chi2 / n

    # Bias correction (formula adapted from wikipedia article on Cramer's V)
    phi2_corr = max(0, phi2 - ((k - 1) / (n - 1)))
    r_corr = 2 - (1 / (n - 1))
    k_corr = k - ((k - 1) ** 2) / (n - 1)

    denom = min(k_corr - 1, r_corr - 1)
    if denom <= 0:
        return np.nan

    return np.sqrt(phi2_corr / denom)

def is_categorical(s: pd.Series, num_cats_thshld: int = 20) -> bool:
    """
    Treat a predictor as categorical if it is boolean or if it is numeric with a small number of unique values.
    This is useful for variables like FISCAL_YR, which are technically numeric, but for identifying missingness 
    patterns, it is more useful to treat them as categorical.

    Args:
    s: pd.Series - the variable to check
    num_cats_thshld: int - the maximum number of unique values for a numeric variable to be considered categorical

    Returns:
    bool - True if the variable should be treated as categorical, False otherwise
    """
    if pd.api.types.is_bool_dtype(s):
        return True
    if pd.api.types.is_numeric_dtype(s):
        return s.nunique(dropna=True) <= num_cats_thshld
    return True

def missingness_association_scan(
    df: pd.DataFrame,
    exclude_cols: list[str] | None = None,
    num_cats_thshld: int = 20,
) -> pd.DataFrame:
    """
    Build a long-format dataframe measuring pairwise association between the
    missingness of one variable and every other predictor variable.

    Output columns:
    - missing_variable
    - predictor_variable
    - association (either cramer's v or pearson r, depending on the result of is_categorical)
    - most common category among missing values of the target variable (only for categorical predictors, else NaN)
    - metric
    - p_value
    - n_used
    - n_missing_target
    """

    exclude_cols = set(exclude_cols)
    target_cols = [c for c in df.columns if c not in exclude_cols]

    results = []

    for target in target_cols:

        # missing indicator of current col
        m = df[target].isna().astype(int)

        if m.sum() == 0:
            continue  # skip cols with no missing values
        
        for predictor in target_cols:

            # dont use predictor to predict missingness of itself
            if predictor == target:
                continue

            s = df[predictor]

            try:

                # CATEGORICAL PREDICTORS 
                if is_cat:=is_categorical(s, num_cats_thshld=num_cats_thshld):

                    # add missing as a separate category, so we can see if missigness of the predictor 
                    # is associated with missingness of the target
                    s_cat = s.astype("string").fillna("MISSING_PRED")
                    m_cat = m.astype("string")

                    table = pd.crosstab(m_cat, s_cat, dropna=False)

                    if table.shape[0] < 2 or table.shape[1] < 2:
                        assoc = np.nan
                        p_value = np.nan
                        n_used = int(table.to_numpy().sum())
                    else:
                        chi2, p_value, _, _ = chi2_contingency(table, correction=False)
                        assoc = cramers_v_correction(chi2, table)
                        n_used = int(table.to_numpy().sum())

                    metric = "cramers_v"

                # NUMERIC PREDICTORS 
                else:
                    s_num = pd.to_numeric(s, errors="coerce")
                    valid = s_num.notna() & m.notna()
                    s_num = s_num.loc[valid]
                    m_use = m.loc[valid]

                    n_used = int(valid.sum())

                    if n_used < 3 or s_num.nunique() < 2 or m_use.nunique() < 2:
                        assoc = np.nan
                        p_value = np.nan
                    else:
                        # point biserial corr is the same as pearson corr when one variable is binary, just optimized
                        assoc, p_value = pointbiserialr(m_use, s_num)  

                    metric = "pearson_r"

                most_common_cats = s[m].value_counts(normalize=True, sort=True, ascending=False).to_dict() if is_cat else np.nan
                most_common_cats = {cat: f"{prop:.2%}" for cat, prop in most_common_cats.items()} if is_cat else np.nan

                results.append({
                    "missing_variable": target,
                    "predictor_variable": predictor,
                    "association": assoc,
                    "metric": metric,
                    "p_value": p_value,
                    "most_common_category_among_missing": most_common_cats,
                    "n_used": n_used,
                    "n_missing": int(m.sum()),
                })

            except Exception as e:
                results.append({
                    "missing_variable": target,
                    "predictor_variable": predictor,
                    "association": np.nan,
                    "metric": "error",
                    "p_value": np.nan,
                    "most_common_category_among_missing": most_common_cats,
                    "n_used": 0,
                    "n_missing": int(m.sum()),
                    "error": f"{type(e).__name__}: {e}",
                })

    out = pd.DataFrame(results)
    out["abs_association"] = out["association"].abs()
    out = out.sort_values(
        ["missing_variable", "abs_association"],
        ascending=[True, False],
        na_position="last"
    ).reset_index(drop=True)

    return out