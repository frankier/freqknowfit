import os
import click
from functools import partial
import pandas
from os.path import join as pjoin
import warnings
from math import nan


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
            return {
                "const_coef": nan,
                "zipf_coef": nan,
                "const_err": nan,
                "zipf_err": nan,
                "aic": nan,
                "aic_c": nan,
                "bic_deviance": nan,
                "bic_llf": nan,
            }

    maybe_print_summary(model)
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


METHODS = {
    "statsmodelsGlmLogit": partial(fit_statsmodels, link="logit"),
    "statsmodelsGlmProbit": partial(fit_statsmodels, link="probit"),
    "statsmodelsGlmCloglog": partial(fit_statsmodels, link="cloglog"),
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
            print(f"Regressing respondent {respondent} [{idx1} / {len(grouped)}]")
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
