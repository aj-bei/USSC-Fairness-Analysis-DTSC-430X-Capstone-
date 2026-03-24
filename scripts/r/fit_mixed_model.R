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

null_to_na <- function(x) {
  if (is.null(x)) return(NA)
  x
}

safe_capture_summary <- function(model) {
  paste(capture.output(summary(model)), collapse = "\n")
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

extract_fit_stats <- function(model) {
  out <- list(
    AIC = tryCatch(AIC(model), error = function(e) NA_real_),
    BIC = tryCatch(BIC(model), error = function(e) NA_real_),
    logLik = tryCatch(as.numeric(logLik(model)), error = function(e) NA_real_),
    deviance = tryCatch(
      {
        if (inherits(model, "lmerMod")) {
          NA_real_
        } else {
          deviance(model)
        }
      },
      error = function(e) NA_real_
    )
  )

  if (inherits(model, "lmerMod")) {
    out$sigma <- tryCatch(sigma(model), error = function(e) NA_real_)
    out$REMLcrit <- tryCatch(REMLcrit(model), error = function(e) NA_real_)
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

build_control <- function(family_name, optimizer) {
  if (family_name == "gaussian") {
    if (!is.null(optimizer) && !is.na(optimizer) && optimizer != "") {
      return(lmerControl(optimizer = optimizer))
    }
    return(lmerControl())
  }

  if (!is.null(optimizer) && !is.na(optimizer) && optimizer != "") {
    return(glmerControl(optimizer = optimizer))
  }
  return(glmerControl())
}

make_binomial_family <- function(link_name) {
  if (is.null(link_name) || is.na(link_name) || link_name == "") {
    stats::binomial(link = "logit")
  } else {
    stats::binomial(link = link_name)
  }
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

result <- list(
  success = FALSE,
  formula = config$formula,
  family = config$family,
  link = null_to_na(config$link),
  engine = "lme4",
  n_input = config$n_input,
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

      if (!is.null(config$categorical_cols) && length(config$categorical_cols) > 0) {
        for (col in config$categorical_cols) {
          if (!col %in% names(data)) {
            stop(sprintf("Categorical column '%s' not found in data.", col))
          }
          data[[col]] <- as.factor(data[[col]])
        }
      }

      if (!is.null(config$reference_levels) && length(config$reference_levels) > 0) {
        for (col in names(config$reference_levels)) {
          ref <- config$reference_levels[[col]]

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

      fml <- as.formula(config$formula)
      ctrl <- build_control(config$family, config$optimizer)

      model <- NULL

      if (config$family == "gaussian") {
        if (requireNamespace("lmerTest", quietly = TRUE)) {
          model <- lmerTest::lmer(
            formula = fml,
            data = data,
            control = ctrl,
            REML = TRUE
          )
        } else {
          model <- lme4::lmer(
            formula = fml,
            data = data,
            control = ctrl,
            REML = TRUE
          )
          warnings_collected <- c(
            warnings_collected,
            "Package 'lmerTest' not installed; gaussian-model p-values may be unavailable."
          )
        }
      } else if (config$family == "binomial") {
        fam <- make_binomial_family(config$link)
        model <- lme4::glmer(
          formula = fml,
          data = data,
          family = fam,
          control = ctrl,
          nAGQ = config$nAGQ
        )
      } else {
        stop(sprintf("Unsupported family: %s", config$family))
      }

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

      if (isTRUE(config$return_fitted)) {
        result$diagnostics$fitted_head <- unname(as.list(head(fitted(model), 10)))
      }

      if (isTRUE(config$return_residuals)) {
        resid_vec <- tryCatch(residuals(model), error = function(e) NULL)
        if (!is.null(resid_vec)) {
          result$diagnostics$residuals_head <- unname(as.list(head(resid_vec, 10)))
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

safe_write_result(result, config$result_path)