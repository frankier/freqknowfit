library(arrow)
library(VGAM)

args <- commandArgs(trailingOnly = TRUE)
df <- read_feather(args[1])
df <- as.data.frame(df)
df.resp <- df[df$respondent == 1, ]

fit <- vglm(cbind(unknown, known) ~ zipf, zibinomialff, data = df.resp, trace = TRUE)
