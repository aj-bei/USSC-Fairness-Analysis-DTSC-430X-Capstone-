#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(lme4)
  library(broom.mixed)
})

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  stop("Usage: Rscript fit_mixed_model.R <config_path>")
}

config_path <- args[1]
config <- jsonlite::fromJSON(config_path, simplifyVector = TRUE)

normalize_family_name <- function(family_name) {
  fam <- tolower(family_name)

  if (fam %in% c("gaussian", "normal")) return("gaussian")
  if (fam %in% c("binomial", "logistic")) return("binomial")
  if (fam %in% c("gamma")) return("gamma")
  if (fam %in% c("negative_binomial", "negative-binomial", "negbin", "nb", "nbinom")) {
    return("negative_binomial")
  }

  fam
}

config$family <- normalize_family_name(config$family)

null_to_na <- function(x) {
  if (is.null(x)) return(NA)
  x
}

safe_capture_summary <- function(model) {
  paste(capture.output(summary(model)), collapse = "\n")
}

safe_capture_print <- function(obj) {
  paste(capture.output(print(obj)), collapse = "\n")
}

safe_write_result <- function(result_obj, path) {
  jsonlite::write_json(
    x = result_obj,
    path = path,
    pretty = TRUE,
    auto_unbox = TRUE,
    na = "null"
  )
}

df_to_records <- function(df) {
  if (is.null(df) || nrow(df) == 0) {
    return(list())
  }

  records <- vector("list", nrow(df))

  for (i in seq_len(nrow(df))) {
    row_list <- as.list(df[i, , drop = FALSE])
    records[[i]] <- lapply(row_list, function(x) {
      if (length(x) == 0) {
        return(NULL)
      }
      x[[1]]
    })
  }

  records
}

build_control <- function(family_name, optimizer, n_iter) {
  if (family_name == "gaussian") {
    if (!is.null(optimizer) && !is.na(optimizer) && optimizer != "") {
      return(lmerControl(optimizer = optimizer, optCtrl = list(maxfun = n_iter)))
    }
    return(lmerControl(optCtrl = list(maxfun = n_iter)))
  }

  if (!is.null(optimizer) && !is.na(optimizer) && optimizer != "") {
    return(glmerControl(optimizer = optimizer, optCtrl = list(maxfun = n_iter)))
  }
  return(glmerControl(optCtrl = list(maxfun = n_iter)))
}

make_binomial_family <- function(link_name) {
  if (is.null(link_name) || is.na(link_name) || link_name == "") {
    stats::binomial(link = "logit")
  } else {
    stats::binomial(link = link_name)
  }
}

make_gamma_family <- function(link_name) {
  if (is.null(link_name) || is.na(link_name) || link_name == "") {
    stats::Gamma(link = "log")
  } else {
    stats::Gamma(link = link_name)
  }
}

coerce_model_data <- function(data, categorical_cols, reference_levels) {
  if (!is.null(categorical_cols) && length(categorical_cols) > 0) {
    for (col in categorical_cols) {
      if (!col %in% names(data)) {
        stop(sprintf("Categorical column '%s' not found in data.", col))
      }
      data[[col]] <- as.factor(data[[col]])
    }
  }

  if (!is.null(reference_levels) && length(reference_levels) > 0) {
    for (col in names(reference_levels)) {
      ref <- reference_levels[[col]]

      if (!col %in% names(data)) {
        stop(sprintf("Reference-level column '%s' not found in data.", col))
      }

      if (!is.factor(data[[col]])) {
        data[[col]] <- as.factor(data[[col]])
      }

      if (!ref %in% levels(data[[col]])) {
        stop(sprintf(
          "Reference level '%s' not found in levels of column '%s'.",
          ref, col
        ))
      }

      data[[col]] <- relevel(data[[col]], ref = ref)
    }
  }

  data
}

fit_model_from_formula <- function(formula_str, data, family_name, link_name, optimizer, n_iter, nAGQ, reml, offset) {
  fml <- as.formula(formula_str)
  ctrl <- build_control(family_name, optimizer, n_iter)

  if (family_name == "gaussian") {
    if (requireNamespace("lmerTest", quietly = TRUE)) {
      return(lmerTest::lmer(
        formula = fml,
        data = data,
        control = ctrl,
        REML = isTRUE(reml),
        offset = if (!is.null(offset)) data[[offset]] else NULL
      ))
    } else {
      return(lme4::lmer(
        formula = fml,
        data = data,
        control = ctrl,
        REML = isTRUE(reml),
        offset = if (!is.null(offset)) data[[offset]] else NULL
      ))
    }
  }

  if (family_name == "binomial") {
    fam <- make_binomial_family(link_name)
    return(lme4::glmer(
      formula = fml,
      data = data,
      family = fam,
      control = ctrl,
      nAGQ = nAGQ,
      offset = if (!is.null(offset)) data[[offset]] else NULL
    ))
  }

  if (family_name == "gamma") {
    fam <- make_gamma_family(link_name)
    return(lme4::glmer(
      formula = fml,
      data = data,
      family = fam,
      control = ctrl,
      nAGQ = nAGQ,
      offset = if (!is.null(offset)) data[[offset]] else NULL
    ))
  }

  if (family_name == "negative_binomial") {
    return(lme4::glmer.nb(
      formula = fml,
      data = data,
      control = ctrl,
      offset = if (!is.null(offset)) data[[offset]] else NULL
    ))
  }

  stop(sprintf("Unsupported family: %s", family_name))
}

extract_fit_stats <- function(model) {
  out <- list(
    AIC = tryCatch(AIC(model), error = function(e) NA_real_),
    BIC = tryCatch(BIC(model), error = function(e) NA_real_),
    logLik = tryCatch(as.numeric(logLik(model)), error = function(e) NA_real_),
    deviance = tryCatch(
      {
        if (inherits(model, "lmerMod")) {
          deviance(model, REML = FALSE)
        } else {
          deviance(model)
        }
      },
      error = function(e) NA_real_
    ),
    nobs = tryCatch(stats::nobs(model), error = function(e) NA_integer_),
    df_residual = tryCatch(df.residual(model), error = function(e) NA_real_)
  )

  if (inherits(model, "lmerMod")) {
    out$sigma <- tryCatch(sigma(model), error = function(e) NA_real_)
    out$REMLcrit <- tryCatch(REMLcrit(model), error = function(e) NA_real_)
  }

  if (inherits(model, "glmerMod")) {
    out$theta_nb <- tryCatch({
      if ("glmer.nb.theta" %in% getNamespaceExports("lme4")) {
        lme4::getME(model, "glmer.nb.theta")
      } else {
        NA_real_
      }
    }, error = function(e) {
      tryCatch(model@resp$theta, error = function(e2) NA_real_)
    })
  }

  out
}

extract_diagnostics <- function(model) {
  optinfo <- tryCatch(model@optinfo, error = function(e) NULL)

  conv_messages <- tryCatch({
    msgs <- optinfo$conv$lme4$messages
    if (is.null(msgs)) character(0) else as.character(msgs)
  }, error = function(e) character(0))

  singular <- tryCatch(isSingular(model, tol = 1e-4), error = function(e) NA)
  converged <- length(conv_messages) == 0

  list(
    converged = converged,
    singular = singular,
    convergence_messages = as.list(conv_messages),
    optimizer = tryCatch(
      {
        opt <- optinfo$optimizer
        if (is.null(opt)) NA_character_ else as.character(opt)
      },
      error = function(e) NA_character_
    )
  )
}

extract_random_effects <- function(model) {
  re <- ranef(model)

  if (length(re) == 0) {
    return(data.frame())
  }

  out_list <- list()

  for (grp in names(re)) {
    df <- as.data.frame(re[[grp]])
    df$level <- rownames(df)
    rownames(df) <- NULL
    df$group <- grp

    value_cols <- setdiff(names(df), c("level", "group"))

    if (length(value_cols) == 0) {
      next
    }

    long_rows <- list()
    idx <- 1

    for (col_name in value_cols) {
      tmp <- data.frame(
        group = df$group,
        level = df$level,
        term = col_name,
        estimate = df[[col_name]],
        stringsAsFactors = FALSE
      )
      long_rows[[idx]] <- tmp
      idx <- idx + 1
    }

    out_list[[grp]] <- do.call(rbind, long_rows)
  }

  if (length(out_list) == 0) {
    return(data.frame())
  }

  combined <- do.call(rbind, out_list)

  fixef_vals <- tryCatch(fixef(model), error = function(e) NULL)
  if (!is.null(fixef_vals) && "(Intercept)" %in% names(fixef_vals)) {
    combined$conditional_estimate <- NA_real_
    intercept_rows <- combined$term == "(Intercept)"
    combined$conditional_estimate[intercept_rows] <-
      unname(fixef_vals["(Intercept)"]) + combined$estimate[intercept_rows]
  }

  rownames(combined) <- NULL
  combined
}

extract_random_effects_covariance <- function(model) {
  vc <- VarCorr(model)

  if (length(vc) == 0) {
    return(data.frame())
  }

  out_list <- list()

  for (grp in names(vc)) {
    mat <- as.matrix(vc[[grp]])

    if (is.null(mat) || length(mat) == 0) {
      next
    }

    df <- as.data.frame(as.table(mat), stringsAsFactors = FALSE)
    names(df) <- c("term1", "term2", "covariance")
    df$group <- grp

    df <- df[, c("group", "term1", "term2", "covariance")]
    out_list[[grp]] <- df
  }

  if (length(out_list) == 0) {
    return(data.frame())
  }

  combined <- do.call(rbind, out_list)
  rownames(combined) <- NULL
  combined
}

extract_test_info <- function(anova_df) {
  if (is.null(anova_df) || nrow(anova_df) == 0) {
    return(list(statistic = NA_real_, p_value = NA_real_, test_label = NA_character_))
  }

  last_row <- anova_df[nrow(anova_df), , drop = FALSE]

  stat_col_candidates <- c("Chisq", "L.Ratio", "F value", "F", "Chi Df")
  p_col_candidates <- c("Pr(>Chisq)", "Pr(>Chi)", "Pr(>F)")
  stat_col <- stat_col_candidates[stat_col_candidates %in% names(anova_df)]
  p_col <- p_col_candidates[p_col_candidates %in% names(anova_df)]

  statistic <- if (length(stat_col) > 0) suppressWarnings(as.numeric(last_row[[stat_col[1]]])) else NA_real_
  p_value <- if (length(p_col) > 0) suppressWarnings(as.numeric(last_row[[p_col[1]]])) else NA_real_
  test_label <- if (length(stat_col) > 0) stat_col[1] else NA_character_

  list(
    statistic = statistic,
    p_value = p_value,
    test_label = test_label
  )
}

extract_binomial_diagnostics <- function(model) {
  diagnostics <- list()

  prob <- tryCatch(as.numeric(fitted(model)), error = function(e) NULL)
  mf <- tryCatch(model.frame(model), error = function(e) NULL)
  response <- tryCatch(model.response(mf), error = function(e) NULL)

  if (!is.null(prob)) {
    diagnostics$predicted_prob <- unname(as.list(prob))
  }

  if (!is.null(response)) {
    if (is.factor(response)) {
      if (nlevels(response) != 2) {
        stop("Binomial diagnostics currently require a binary response.")
      }
      diagnostics$observed_response <- unname(as.list(as.integer(response == levels(response)[2])))
      diagnostics$positive_class <- levels(response)[2]
    } else {
      diagnostics$observed_response <- unname(as.list(as.integer(as.numeric(response))))
      diagnostics$positive_class <- 1
    }
  }

  diagnostics
}

extract_dharma_diagnostics <- function(model) {
  diagnostics <- list()

  if (!requireNamespace("DHARMa", quietly = TRUE)) {
    diagnostics$dharma_available <- FALSE
    diagnostics$dharma_message <- "Package 'DHARMa' is not installed; simulated residual diagnostics unavailable."
    return(diagnostics)
  }

  sim_out <- DHARMa::simulateResiduals(fittedModel = model, plot = FALSE)

  diagnostics$dharma_available <- TRUE
  diagnostics$dharma_scaled_residuals <- unname(as.list(as.numeric(sim_out$scaledResiduals)))
  diagnostics$dharma_fitted_predicted <- unname(as.list(as.numeric(sim_out$fittedPredictedResponse)))

  uniformity <- DHARMa::testUniformity(sim_out)
  dispersion <- DHARMa::testDispersion(sim_out)
  outliers <- DHARMa::testOutliers(sim_out)

  diagnostics$dharma_tests <- list(
    uniformity = list(
      statistic = unname(uniformity$statistic),
      p_value = uniformity$p.value,
      method = uniformity$method
    ),
    dispersion = list(
      statistic = unname(dispersion$statistic),
      p_value = dispersion$p.value,
      method = dispersion$method
    ),
    outliers = list(
      statistic = unname(outliers$statistic),
      p_value = outliers$p.value,
      method = outliers$method
    )
  )

  diagnostics
}

run_fit_mode <- function(config) {
  result <- list(
    success = FALSE,
    formula = config$formula,
    family = config$family,
    link = null_to_na(config$link),
    offset = config$offset,
    engine = "lme4",
    n_input = config$n_input,
    n_written = config$n_written,
    n_used = NA_integer_,
    n_dropped = NA_integer_,
    converged = NA,
    singular = NA,
    fixed_effects = list(),
    random_effects_variance = list(),
    random_effects = list(),
    random_effects_covariance = list(),
    fit_statistics = list(),
    diagnostics = list(),
    warnings = list(),
    errors = list(),
    raw_summary = NA_character_
  )

  warnings_collected <- character(0)

  tryCatch(
    withCallingHandlers(
      {
        data <- read.csv(config$data_path, stringsAsFactors = FALSE)
        data <- coerce_model_data(data, config$categorical_cols, config$reference_levels)
        
        model <- fit_model_from_formula(
          formula_str = config$formula,
          data = data,
          family_name = config$family,
          link_name = config$link,
          optimizer = config$optimizer,
          n_iter = config$n_iter,
          nAGQ = config$nAGQ,
          reml = config$reml,
          offset = config$offset
        )

        mf <- model.frame(model)
        n_used <- nrow(mf)

        result$n_used <- n_used
        result$n_dropped <- result$n_input - n_used

        fixed_tidy <- broom.mixed::tidy(model, effects = "fixed")
        result$fixed_effects <- df_to_records(fixed_tidy)

        if (isTRUE(config$return_random_effects_variance)) {
          ran_pars <- broom.mixed::tidy(model, effects = "ran_pars")
          result$random_effects_variance <- df_to_records(ran_pars)
        }

        if (isTRUE(config$return_random_effects)) {
          re_df <- extract_random_effects(model)
          result$random_effects <- df_to_records(re_df)
        }

        if (isTRUE(config$return_random_effects_covariance)) {
          cov_df <- extract_random_effects_covariance(model)
          result$random_effects_covariance <- df_to_records(cov_df)
        }

        result$fit_statistics <- extract_fit_stats(model)

        diag_out <- extract_diagnostics(model)
        result$diagnostics <- diag_out
        result$converged <- diag_out$converged
        result$singular <- diag_out$singular

        if (config$family == "gaussian") {
          if (isTRUE(config$return_fitted)) {
            result$diagnostics$fitted <- unname(as.list(as.numeric(fitted(model))))
          }

          if (isTRUE(config$return_residuals)) {
            resid_vec <- tryCatch(residuals(model), error = function(e) NULL)
            if (!is.null(resid_vec)) {
              result$diagnostics$residuals <- unname(as.list(as.numeric(resid_vec)))
            }
          }
        }

        if (config$family == "binomial") {
          if (isTRUE(config$return_fitted)) {
            binom_diag <- extract_binomial_diagnostics(model)
            result$diagnostics <- c(result$diagnostics, binom_diag)
          }

          if (isTRUE(config$return_residuals)) {
            resid_vec <- tryCatch(residuals(model, type = "pearson"), error = function(e) NULL)
            if (!is.null(resid_vec)) {
              result$diagnostics$residuals <- unname(as.list(as.numeric(resid_vec)))
            }
          }
        }

        if (config$family %in% c("gamma", "negative_binomial", "binomial")) {
          if (isTRUE(config$return_fitted) || isTRUE(config$return_residuals)) {
            dharma_diag <- extract_dharma_diagnostics(model)
            result$diagnostics <- c(result$diagnostics, dharma_diag)
          }

          if (isTRUE(config$return_fitted)) {
            fit_vec <- tryCatch(as.numeric(fitted(model)), error = function(e) NULL)
            if (!is.null(fit_vec)) {
              result$diagnostics$fitted <- unname(as.list(fit_vec))
            }
          }

          if (isTRUE(config$return_residuals)) {
            resid_vec <- tryCatch(residuals(model, type = "pearson"), error = function(e) NULL)
            if (!is.null(resid_vec)) {
              result$diagnostics$residuals <- unname(as.list(as.numeric(resid_vec)))
            }
          }
        }

        if (isTRUE(config$keep_raw_summary)) {
          result$raw_summary <- safe_capture_summary(model)
        }

        result$warnings <- as.list(unique(c(
          warnings_collected,
          unlist(diag_out$convergence_messages, use.names = FALSE)
        )))
        result$errors <- list()
        result$success <- TRUE
      },
      warning = function(w) {
        warnings_collected <<- c(warnings_collected, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) {
      result$errors <<- as.list(c(unlist(result$errors, use.names = FALSE), conditionMessage(e)))
      result$warnings <<- as.list(unique(warnings_collected))
      result$success <<- FALSE
    }
  )

  result
}

run_anova_mode <- function(config) {
  result <- list(
    success = FALSE,
    formula_null = config$formula_null,
    formula_alt = config$formula_alt,
    family = config$family,
    link = null_to_na(config$link),
    test_type = config$test_type,
    offset = config$offset,
    reml_used = NA,
    model_null_fit = list(),
    model_alt_fit = list(),
    comparison_table = list(),
    test = list(statistic = NA_real_, p_value = NA_real_, test_label = NA_character_),
    warnings = list(),
    errors = list(),
    raw_anova = NA_character_
  )

  warnings_collected <- character(0)

  tryCatch(
    withCallingHandlers(
      {
        data <- read.csv(config$data_path, stringsAsFactors = FALSE)
        data <- coerce_model_data(data, config$categorical_cols, config$reference_levels)

        reml_to_use <- FALSE
        if (config$family == "gaussian") {
          reml_to_use <- identical(tolower(config$test_type), "reml")
        }

        model_null <- fit_model_from_formula(
          formula_str = config$formula_null,
          data = data,
          family_name = config$family,
          link_name = config$link,
          optimizer = config$optimizer,
          n_iter = config$n_iter,
          nAGQ = config$nAGQ,
          reml = reml_to_use,
          offset = config$offset
        )

        model_alt <- fit_model_from_formula(
          formula_str = config$formula_alt,
          data = data,
          family_name = config$family,
          link_name = config$link,
          optimizer = config$optimizer,
          n_iter = config$n_iter,
          nAGQ = config$nAGQ,
          reml = reml_to_use,
          offset = config$offset
        )

        result$reml_used <- reml_to_use
        result$model_null_fit <- extract_fit_stats(model_null)
        result$model_alt_fit <- extract_fit_stats(model_alt)

        if (config$family == "gaussian" && reml_to_use) {
          anova_obj <- anova(model_null, model_alt)
        } else {
          anova_obj <- anova(model_null, model_alt, test = "Chisq")
        }

        anova_df <- as.data.frame(anova_obj)
        anova_df$model <- rownames(anova_df)
        rownames(anova_df) <- NULL

        result$comparison_table <- df_to_records(anova_df)
        result$test <- extract_test_info(anova_df)
        result$raw_anova <- safe_capture_print(anova_obj)
        result$warnings <- as.list(unique(warnings_collected))
        result$errors <- list()
        result$success <- TRUE
      },
      warning = function(w) {
        warnings_collected <<- c(warnings_collected, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) {
      result$errors <<- as.list(c(unlist(result$errors, use.names = FALSE), conditionMessage(e)))
      result$warnings <<- as.list(unique(warnings_collected))
      result$success <<- FALSE
    }
  )

  result
}

mode <- if (!is.null(config$mode)) config$mode else "fit"

result <- switch(
  mode,
  fit = run_fit_mode(config),
  anova = run_anova_mode(config),
  stop(sprintf("Unsupported mode: %s", mode))
)

safe_write_result(result, config$result_path)