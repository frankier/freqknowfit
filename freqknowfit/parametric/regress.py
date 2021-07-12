import click
from functools import partial
import pandas


def fit_statsmodels(df_resp, link):
    from statsmodels.genmod.families import links as L
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.generalized_linear_model import GLM
    from statsmodels.tools.tools import add_constant

    if link == "logit":
        link_func = L.logit
    elif link == "probit":
        link_func = L.probit
    elif link == "cloglog":
        link_func = L.cloglog
    else:
        assert False

    model = GLM(
        df_resp["known"],
        add_constant(df_resp["zipf"]),
        family=Binomial(link=link_func)
    ).fit()
    print(model.summary())
    return {
        "const_coef": model.params[0],
        "zipf_coef": model.params[1],
        "const_err": model.bse[0],
        "zipf_err": model.bse[1],
        "aic": model.aic,
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
    fit = METHODS[method]
    df = pandas.read_parquet(dfin)
    cols = None
    for respondent, resp_df in df.groupby("respondent"):
        row = fit(resp_df)
        if cols is None:
            cols = {k: [] for k in row}
            cols["respondent"] = []
        for k, v in row.items():
            cols[k].append(v)
        cols["respondent"].append(respondent)
    pandas.DataFrame(cols).to_parquet(dfout)


if __name__ == "__main__":
    main()
