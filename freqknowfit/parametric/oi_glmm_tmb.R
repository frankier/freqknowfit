library(arrow)
library(glmmTMB)

args <- commandArgs(trailingOnly = TRUE)
df <- read_feather(args[1])
df <- as.data.frame(df)
df.resp <- df[df$respondent == 1, ]

ziLogitReg <- glmmTMB(
  unknown ~ zipf,
  data=df.resp,
  family=binomial(link="logit"),
  ziformula=~1,
)
summary(ziLogitReg)

ziProbitReg <- glmmTMB(
  unknown ~ zipf,
  data=df.resp,
  family=binomial(link="probit"),
  ziformula=~1,
)
summary(ziProbitReg)

ziCloglogReg <- glmmTMB(
  unknown ~ zipf,
  data=df.resp,
  family=binomial(link="cloglog"),
  ziformula=~1,
)
summary(ziCloglogReg)
