library(arrow)
library(aod)


args <- commandArgs(trailingOnly = TRUE)
df <- read_feather(args[1])
df <- as.data.frame(df)
df.resp <- df[df$respondent == 1, ]

fixedRandLogit <- betabin(formula = cbind(known, !known) ~ zipf, random = ~ 1, data=df.resp, link="logit")
summary(fixedRandLogit)

fixedRandCLogLog <- betabin(formula = cbind(known, !known) ~ zipf, random = ~ 1, data=df.resp, link="cloglog")
summary(fixedRandCLogLog)
