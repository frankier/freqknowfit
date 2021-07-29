library(here)
library(TMB)

tmb_oi_init <- FALSE

init_tmb_oi <- function() {
    print(getwd())
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
    obj <- MakeADFun(data, parameters, DLL="reg")
    #print("nlls")
    #options(digits=20)
    #obj2 <- MakeADFun(data, list(inflate_coef=-16, reg_const_coef=reg_const_coef, reg_zipf_coef=reg_zipf_coef), DLL="reg")
    #print(obj2$fn())
    #obj2 <- MakeADFun(data, list(inflate_coef=0, reg_const_coef=reg_const_coef, reg_zipf_coef=reg_zipf_coef), DLL="reg")
    #print(obj2$fn())
    #print(obj$fn(inflate_coef=-15, reg_const_coef=initial_const_coef, reg_zipf_coef=initial_zipf_coef))
    #print(obj$fn(inflate_coef=-8, reg_const_coef=initial_const_coef, reg_zipf_coef=initial_zipf_coef))
    #print(obj$fn(inflate_coef=-1, reg_const_coef=initial_const_coef, reg_zipf_coef=initial_zipf_coef))
    #print(obj$fn(inflate_coef=0, reg_const_coef=initial_const_coef, reg_zipf_coef=initial_zipf_coef))
    #print(obj$fn(inflate_coef=4, reg_const_coef=initial_const_coef, reg_zipf_coef=initial_zipf_coef))
    #print(names(obj))
    #print("obj")
    obj$control <- list(
        maxit = 30000,
        trace = TRUE,
        reltol = 1e-10
    )
    #obj$hessian <- TRUE
    #obj$method <- "BFGS"
    #obj$reltol <- 1e-10
    #obj$factr <- 1e10
    print("opt")
    opt <- do.call("optim", obj)
    print(opt)
    opt
}
