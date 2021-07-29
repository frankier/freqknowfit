import os
import numpy
import click
from functools import partial
import pandas
from os.path import join as pjoin
import warnings
from math import nan
from os.path import exists
from string import Template


STATSMODELS_NANS = {
    "const_coef": nan,
    "zipf_coef": nan,
    "const_err": nan,
    "zipf_err": nan,
    "aic": nan,
    "aic_c": nan,
    "bic_deviance": nan,
    "bic_llf": nan,
}


def maybe_print_summary(fit):
    if os.environ.get("PRINT_SUMMARIES"):
        print(fit.summary())


def fit_statsmodels(df_resp, link):
    from statsmodels.genmod.families import links as L
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.generalized_linear_model import GLM
    from statsmodels.tools.tools import add_constant
    from statsmodels.tools.sm_exceptions import PerfectSeparationError
    from statsmodels.tools.eval_measures import aicc

    if link == "logit":
        link_func = L.logit()
    elif link == "probit":
        link_func = L.probit()
    elif link == "cloglog":
        link_func = L.cloglog()
    else:
        assert False

    with warnings.catch_warnings():
        # TODO: Fixed in next statsmodels release
        # FutureWarning: In a future version of pandas all arguments of concat
        # except for the argument 'objs' will be keyword-only
        warnings.simplefilter("ignore")
        try:
            model = GLM(
                df_resp["known"],
                add_constant(df_resp["zipf"]),
                family=Binomial(link=link_func)
            ).fit()
        except PerfectSeparationError:
            return STATSMODELS_NANS

    maybe_print_summary(model)
    if len(model.params) < 2:
        return STATSMODELS_NANS

    return {
        "const_coef": model.params[0],
        "zipf_coef": model.params[1],
        "const_err": model.bse[0],
        "zipf_err": model.bse[1],
        "aic": model.aic,
        # XXX: Does df_model include the constant? Is it the same as df_modelwc?
        "aic_c": aicc(model.llf, model.nobs, model.df_model),
        "bic_deviance": model.bic_deviance,
        "bic_llf": model.bic_llf,
    }


STAN_MODEL = Template("""
functions {
    int num_zeros(int[] y) {
        int sum = 0;
        for (n in 1:size(y))
            sum += (y[n] == 0);
        return sum;
    }
}

data {
    int<lower=0> N;
    int<lower=0> K;
    matrix[N, K] x;
    int<lower=0> y[N];
}

transformed data {
    int<lower = 0> N_zero = num_zeros(y);
    int<lower = 0> N_ones = N - N_zero;
}

parameters {
    real inflate_coef;
    vector[K] reg_coef;
}

transformed parameters {
    real inflate_mags;
    vector[N] reg_mags;
    inflate_mags = inv_logit(inflate_coef);
    reg_mags =  $REG_LINK(x * reg_coef);
}

model {
    inflate_coef ~ normal(0, 1);
    reg_coef ~ normal(0, 1);
    for (n in 1:N){
        if (y[n] == 1) {
            target += log_sum_exp(
                bernoulli_lpmf(1 | inflate_mags),
                bernoulli_lpmf(0 | inflate_mags)
                + bernoulli_lpmf(1 | reg_mags[n])
            );
        } else {
            target += (
                bernoulli_lpmf(0 | inflate_mags)
                + bernoulli_lpmf(0 | reg_mags[n])
            );
        }
    }
}

generated quantities {
    real log_lik[N];
    for (n in 1:N){
        if (y[n] == 1) {
            log_lik[n] = log_sum_exp(
                bernoulli_lpmf(1 | inflate_mags),
                bernoulli_lpmf(0 | inflate_mags)
                + bernoulli_lpmf(1 | reg_mags[n])
            );
        } else {
            log_lik[n] = (
                bernoulli_lpmf(0 | inflate_mags)
                + bernoulli_lpmf(0 | reg_mags[n])
            );
        }
    }
}
""")


STAN_LOGIT_MODEL = STAN_MODEL.substitute(REG_LINK="inv_logit")
STAN_PROBIT_MODEL = STAN_MODEL.substitute(REG_LINK="Phi")
STAN_CLOGLOG_MODEL = STAN_MODEL.substitute(REG_LINK="inv_cloglog")


def get_model_path():
    return os.environ.get("STAN_PROG_DIR", os.environ.get("TMPDIR", "/tmp"))


def ensure_model_at_path(path, contents):
    if exists(path):
        with open(path, "r") as inf:
            if inf.read() == contents:
                return
    with open(path, "w") as outf:
        outf.write(contents)


class SliceTaker:
    def __init__(self, arr):
        self.arr = arr
        self.cursor = 0

    def take(self, length):
        res = self.arr[self.cursor: self.cursor + length]
        self.cursor += length
        return res

    def drop(self, length):
        self.cursor += length


def fit_stan(df_resp, link):
    from statsmodels.tools.tools import add_constant
    from cmdstanpy import CmdStanModel

    if link == "logit":
        stan_code = STAN_LOGIT_MODEL
    elif link == "probit":
        stan_code = STAN_PROBIT_MODEL
    elif link == "cloglog":
        stan_code = STAN_CLOGLOG_MODEL
    else:
        assert False
    full_path = pjoin(get_model_path(), link + ".stan")
    ensure_model_at_path(full_path, stan_code)
    model = CmdStanModel(link, full_path)
    n = len(df_resp)
    k = 2
    mle = model.optimize({
        "N": n,
        "K": k,
        "x": add_constant(df_resp["zipf"].to_numpy()),
        "y": df_resp["known"].to_numpy().astype(numpy.int32),
    })
    taker = SliceTaker(mle.optimized_params_np)
    lp = taker.take(1)[0]
    inflate_coef = taker.take(1)[0]
    reg_coef = taker.take(k)
    # Drop reg_mags
    taker.drop(n)
    log_lik = taker.take(n)
    sum_lok_lik = sum(log_lik)
    num_params = k + 1
    aic = 2 * num_params - 2 * sum_lok_lik

    return {
        "const_coef": reg_coef[0],
        "zipf_coef": reg_coef[1],
        "phi_coef": inflate_coef,
        "aic": aic,
    }


METHODS = {
    "statsmodelsGlmLogit": partial(fit_statsmodels, link="logit"),
    "statsmodelsGlmProbit": partial(fit_statsmodels, link="probit"),
    "statsmodelsGlmCloglog": partial(fit_statsmodels, link="cloglog"),
    "stanLogit": partial(fit_stan, link="logit"),
    "stanProbit": partial(fit_stan, link="probit"),
    "stanCloglog": partial(fit_stan, link="cloglog"),
}


@click.command()
@click.argument("method")
@click.argument("dfin", type=click.Path(exists=True))
@click.argument("dfout", type=click.Path())
def main(method, dfin, dfout):
    print("Loading dataframe")
    fit = METHODS[method]
    df = pandas.read_parquet(dfin)
    print("Loaded!")
    cols = None
    idx1 = 1
    grouped = df.groupby("respondent")
    for respondent, resp_df in grouped:
        if os.environ.get("PRINT_PROGRESS"):
            print(
                f"Regressing respondent {respondent} [{idx1} / {len(grouped)}]"
            )
        row = fit(resp_df)
        if cols is None:
            cols = {k: [] for k in row}
            cols["respondent"] = []
        for k, v in row.items():
            cols[k].append(v)
        cols["respondent"].append(respondent)
        idx1 += 1
    print("Writing dataframe")
    os.makedirs(dfout, exist_ok=True)
    full_out = pjoin(dfout, "00000001.parquet")
    pandas.DataFrame(cols).to_parquet(full_out)
    print("Written")


if __name__ == "__main__":
    main()
