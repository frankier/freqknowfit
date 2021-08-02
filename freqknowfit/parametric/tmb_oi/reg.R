library(here)
library(TMB)

tmb_oi_init <- FALSE

init_tmb_oi <- function() {
    compile(here("freqknowfit/parametric/tmb_oi/reg.cpp"))
    dyn.load(dynlib(here("freqknowfit/parametric/tmb_oi/reg")))
    tmb_oi_init <<- TRUE
}

fit_tmb_oi <- function(x, Y, inflate_coef, reg_const_coef, reg_zipf_coef) {
    if (!tmb_oi_init) {
        init_tmb_oi()
    }
    data <- list(Y = Y, x=x)
    parameters <- list(inflate_coef=inflate_coef, reg_const_coef=reg_const_coef, reg_zipf_coef=reg_zipf_coef)
    obj <- MakeADFun(data, parameters, DLL="reg", silent=TRUE)
    obj$control <- list(
        maxit = 30000,
        reltol = 1e-10
    )
    opt <- do.call("optim", obj)
    opt
}
