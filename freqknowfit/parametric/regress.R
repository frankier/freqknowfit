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
  c(
    coef = fit$coef
  )
}

betaBinFit <- function(df, link) {
  library(aod)
  fit <- betabin(formula = cbind(known, !known) ~ zipf, random = ~ 1, data=df, link=link)
  maybePrintSummary(fit)
  c(
    coef = fit$coef
  )
}

glmmTmbFit <- function(df, link) {
  library(glmmTMB)
  fit <- glmmTMB(
    unknown ~ zipf,
    data=df,
    family=binomial(link=link),
    ziformula=~1,
  )
  maybePrintSummary(fit)
  c(
    coef = fit$coef
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
      unknown ~ zipf,
      data=df,
      zeroInflation=TRUE,
      family="binomial",
      debug=TRUE,
      save.dir="glmmadmb"
    )
    maybePrintSummary(fit)
    c(
      coef = fit$coef
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
    fit <- vglm(cbind(unknown, known) ~ zipf, zibinomialff, data = df, trace = TRUE)
    maybePrintSummary(fit)
    c(
      coef = fit$coef
    )
  }
)

library(arrow)

print("Loading dataframe")
args <- commandArgs(trailingOnly = TRUE)
df <- read_parquet(args[2])
resp.dfs <- split(df, df$respondent)
print("Loaded!")

regressor <- regressors[[args[1]]]

cols <- NULL
idx <- 1
for (resp.id in names(resp.dfs)) {
  cat("Regressing respondent ", resp.id, " [", idx, " / ", length(resp.dfs), "]\n")
  resp.df <- resp.dfs[[resp.id]]
  row <- regressor(df)
  if (is.null(cols)) {
    cols <- rep(rep(NULL, length(resp.dfs)), length(row))
    names(cols) <- names(row)
    cols$respondent <- rep(NULL, length(resp.dfs));
  }
  for (col.name in names(row)) {
    cols[[col.name]][[idx]] <- row[[col.name]]
  }
  cols$respondent[[idx]] = resp.id;
  idx <- idx + 1
}
df <- data.frame(cols)
print("Writing dataframe")
write_parquet(df, args[3])
print("Written")
