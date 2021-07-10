library(arrow)

args <- commandArgs(trailingOnly = TRUE)
df <- read_feather(args[1])
df.resp <- df[df$respondent == 1, ]

logitReg = glm(known ~ zipf, data=df.resp, family=binomial(link="logit"))
summary(logitReg)

probitReg = glm(known ~ zipf, data=df.resp, family=binomial(link="probit"))
summary(probitReg)

cloglogReg = glm(known ~ zipf, data=df.resp, family=binomial(link="cloglog"))
summary(cloglogReg)
