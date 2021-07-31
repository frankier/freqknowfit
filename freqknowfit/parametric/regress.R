library(here)
library(arrow)

CHUNK_SIZE <- 65536

printStacktrace <- function() {
  calls <- sys.calls()
  if (length(calls) >= 2L) {
    cat("Backtrace:\n")
    calls <- rev(calls[-length(calls)])
    for (i in seq_along(calls)) {
      cat(i, ": ", deparse(calls[[i]], nlines = 1L), "\n", sep = "")
    }
  }
}


options(error = function() {
  calls <- sys.calls()
  if (length(calls) >= 2L) {
    sink(stderr())
    on.exit(sink(NULL))
    cat("Backtrace:\n")
    calls <- rev(calls[-length(calls)])
    for (i in seq_along(calls)) {
      cat(i, ": ", deparse(calls[[i]], nlines = 1L), "\n", sep = "")
    }
  }
  if (nzchar(Sys.getenv("DEBUG_R_ERRORS"))) {
    recover()
  }
  if (!interactive()) {
    q(status = 1)
  }
})

maybePrintSummary <- function(fit) {
  if (nzchar(Sys.getenv("PRINT_SUMMARIES"))) {
    print(summary(fit))
  }
}

add_computed <- function(res) {
  res["theta"] <- -res["const_coef"] / res["zipf_coef"]
  res
}

glmFit <- function(df, link) {
  fit <- glm(known ~ zipf, data=df, family=binomial(link=link))
  maybePrintSummary(fit)
  coefs  <- coef(summary(fit))
  if (dim(coefs)[[1]] < 2) {
    c(
      const_coef = NaN,
      zipf_coef = NaN,
      const_err = NaN,
      zipf_err = NaN,
      aic = NaN
    )
  } else {
    c(
      const_coef = coefs[[1,1]],
      zipf_coef = coefs[[2,1]],
      const_err = coefs[[1,2]],
      zipf_err = coefs[[2,2]],
      aic = AIC(fit)
    )
  }
}

betaBinFit <- function(df, link) {
  library(aod)
  fit <- betabin(formula = cbind(known, !known) ~ zipf, random = ~ 1, data=as.data.frame(df), link=link)
  maybePrintSummary(fit)
  tryCatch(
    {
      sumFit <- summary(fit)
      coefs <- sumFit@Coef
      phi <- sumFit@Phi
      aic <- AIC(fit)
      c(
        const_coef = coefs[[1,1]],
        zipf_coef = coefs[[2,1]],
        phi_coef = phi[[1,1]],
        const_err = coefs[[1,2]],
        zipf_err = coefs[[2,2]],
        phi_err = phi[[1,2]],
        aic = aic@istats$AIC[[1]],
        aic_c = aic@istats$AICc[[1]]
      )
    },
    error=function(cond) {
      sink(stderr())
      message("Error while getting results from of betabin(...)")
      message(cond)
      printStacktrace()
      sink(NULL)
      c(
        const_coef = NaN,
        zipf_coef = NaN,
        phi_coef = NaN,
        const_err = NaN,
        zipf_err = NaN,
        phi_err = NaN,
        aic = NaN,
        aic_c = NaN
      )
    }
  )
}

glmmTmbFit <- function(df, link) {
  library(glmmTMB)
  tryCatch(
    {
      fit <- glmmTMB(
        cbind(unk, known) ~ zipf,
        data=df,
        family=binomial(link=link),
        ziformula=~1,
      )
      maybePrintSummary(fit)
      coefs  <- coef(summary(fit))
      c(
        const_coef = coefs$cond[[1,1]],
        zipf_coef = coefs$cond[[2,1]],
        zi_coef = coefs$zi[[1,1]],
        const_err = coefs$cond[[1,2]],
        zipf_err = coefs$cond[[2,2]],
        zi_err = coefs$zi[[1,2]],
        aic = AIC(fit)
      )
    },
    error=function(cond) {
      sink(stderr())
      message("Error while getting results from glmmTMB(...)")
      message(cond)
      printStacktrace()
      sink(NULL)
      c(
        const_coef = NaN,
        zipf_coef = NaN,
        zi_coef = NaN,
        const_err = NaN,
        zipf_err = NaN,
        zi_err = NaN,
        aic = NaN
      )
    }
  )
}

regressors <- c(
  glmLogit = function(df) {
    glmFit(df, "logit")
  },
  glmProbit = function(df) {
    glmFit(df, "probit")
  },
  glmCloglog = function(df) {
    glmFit(df, "cloglog")
  },
  aodBetabinLogit = function(df) {
    betaBinFit(df, "logit")
  },
  aodBetabinCloglog = function(df) {
    betaBinFit(df, "cloglog")
  },
  glmmadmb = function(df) {
    library(glmmADMB)

    fit <- glmmadmb(
      known ~ zipf,
      data=df,
      zeroInflation=TRUE,
      family="binomial",
      debug=TRUE,
      save.dir="glmmadmb"
    )
    maybePrintSummary(fit)
    coefs  <- coef(summary(fit))
    c(
      const_coef = coefs[[1,1]],
      zipf_coef = coefs[[2,1]],
      const_err = coefs[[1,2]],
      zipf_err = coefs[[2,2]],
      aic = AIC(fit)
    )
  },
  glmmTmbLogit = function(df) {
    glmmTmbFit(df, "logit")
  },
  glmmTmbProbit = function(df) {
    glmmTmbFit(df, "probit")
  },
  glmmTmbCloglog = function(df) {
    glmmTmbFit(df, "cloglog")
  },
  vglm = function(df) {
    # Broken
    library(VGAM)
    tryCatch(
      {
        fit <- vglm(cbind(known, unk) ~ zipf, zibinomialff, data = df,
                    trace = TRUE)
        maybePrintSummary(fit)
        coefs  <- coef(summary(fit))
        c(
          const_coef = coefs[[1,1]],
          zipf_coef = coefs[[2,1]],
          const_err = coefs[[1,2]],
          zipf_err = coefs[[2,2]],
          aic = AIC(fit)
        )
      },
      error=function(cond) {
        sink(stderr())
        message("Error while fitting vglm(...)")
        message(cond)
        printStacktrace()
        sink(NULL)
        c(
          const_coef = NaN,
          zipf_coef = NaN,
          const_err = NaN,
          zipf_err = NaN,
          aic = NaN
        )
      }
    )
  },
  tmbOi = function(df) {
    if (!exists("tmb_oi_init")) {
      source(here("freqknowfit/parametric/tmb_oi/reg.R"), chdir=TRUE)
    }
    fit <- glm(known ~ zipf, data=df, family=binomial(link="logit"))
    maybePrintSummary(fit)
    coefs  <- coef(summary(fit))
    if (dim(coefs)[[1]] < 2) {
      c(
        const_coef = NaN,
        zipf_coef = NaN,
        phi_coef = NaN,
        aic = NaN
      )
    } else {
      initial_const_coef <- coefs[[1,1]]
      initial_zipf_coef <- coefs[[2,1]]
      # y = mx + c, max entropy when y = 0 => x = -c / m
      theta <- -initial_const_coef / initial_zipf_coef
      if (initial_zipf_coef > 0) {
        mask <- df$zipf < theta
      } else {
        mask <- df$zipf > theta
      }
      num_samples <- sum(mask)
      initial_oi_prob <- mean(df$known[mask])
      if (num_samples < 5 || initial_oi_prob > 0.75) {
        initial_oi_prob <- 0.5
        initial_oi <- 0
      } else {
        initial_oi <- qlogis(initial_oi_prob)
      }
      print(c(
        initial_const_coef=initial_const_coef,
        initial_zipf_coef=initial_zipf_coef,
        initial_oi_prob=initial_oi_prob,
        initial_oi=initial_oi
      ))
      opt <- fit_tmb_oi(df$zipf, df$known, initial_oi, initial_const_coef, initial_zipf_coef)
      coef <- opt$par
      c(
        const_coef = coef[["reg_const_coef"]],
        zipf_coef = coef[["reg_zipf_coef"]],
        phi_coef = coef[["inflate_coef"]],
        aic = 2 * 3 + 2 * opt$value
      )
    }
  }
)

main <- function() {
  # First canonicalise all arguments relative to the original working directory
  args <- commandArgs(trailingOnly = TRUE)
  regressorName <- args[1]
  inParaquet <- R.utils::getAbsolutePath(args[2])
  outParquetsDir <- R.utils::getAbsolutePath(args[3])

  # Now change the working directory to the to the project root to set up the `here` library
  freqknowfit_base <- Sys.getenv("FREQKNOWFIT_BASE")
  if (nzchar(freqknowfit_base)) {
    setwd(freqknowfit_base)
  }
  i_am("freqknowfit/parametric/regress.R")

  print("Loading dataframe")
  df <- read_parquet(inParaquet)
  resp.dfs <- split(df, df$respondent)
  print("Loaded!")

  dir.create(outParquetsDir, recursive = TRUE)
  regressor <- regressors[[regressorName]]
  chunk.idx <- 1

  flush <- function(last) {
    if (last) {
      trueIdx = idx - 1
    } else {
      trueIdx = idx
    }
    chunkSize = trueIdx %% CHUNK_SIZE
    shouldFlush <- (last && (chunkSize > 0)) || chunkSize == 0
    if (!shouldFlush) {
      return()
    }
    print("outParquetsDir")
    print(outParquetsDir)
    outPath <- file.path(outParquetsDir, sprintf("%08d.parquet", chunk.idx))
    outDf <- do.call(data.frame, cols)
    if (chunkSize > 0) {
      outDf <- head(outDf, chunkSize)
    }
    write_parquet(x=outDf, sink=outPath)
    chunk.idx <<- chunk.idx + 1
  }

  cols <- NULL
  idx <- 1
  for (resp.id in names(resp.dfs)) {
    if (nzchar(Sys.getenv("PRINT_PROGRESS"))) {
      cat("Regressing respondent ", resp.id, " [", idx, " / ", length(resp.dfs), "]\n")
    }
    resp.df <- resp.dfs[[resp.id]]
    row <- regressor(resp.df)
    row <- add_computed(row)
    if (is.null(cols)) {
      cols = list(respondent=rep("", CHUNK_SIZE))
      for (col.name in names(row)) {
        cols[[col.name]] = rep(row[[col.name]], CHUNK_SIZE)
      }
    }
    colIdx = ((idx - 1) %% CHUNK_SIZE) + 1
    cols$respondent[colIdx] = resp.id;
    for (col.name in names(row)) {
      cols[[col.name]][colIdx] <- row[[col.name]]
    }
    flush(FALSE)
    idx <- idx + 1
  }
  flush(TRUE)
}

main()
