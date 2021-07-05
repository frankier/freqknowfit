library(arrow)
library(glmmADMB)

args <- commandArgs(trailingOnly = TRUE)
df <- read_feather(args[1])
df <- as.data.frame(df)
df.resp <- df[df$respondent == 1, ]

fit_zipoiss <- glmmadmb(
  unknown ~ zipf,
  data=df.resp,
  zeroInflation=TRUE,
  family="binomial",
  debug=TRUE,
  save.dir="glmmadmb"
)
