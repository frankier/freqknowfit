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
  fit <- glmmTMB(
    known ~ zipf,
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
    library(VGAM)
    tryCatch(
      {
        fit <- vglm(cbind(known, unknown) ~ zipf, zibinomialff, data = df, trace = TRUE)
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
  }
)

library(arrow)

print("Loading dataframe")
args <- commandArgs(trailingOnly = TRUE)
df <- read_parquet(args[2])
resp.dfs <- split(df, df$respondent)
print("Loaded!")

outParquetsDir <- args[3]
dir.create(outParquetsDir, recursive = TRUE)
regressor <- regressors[[args[1]]]
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
