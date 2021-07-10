from statsmodels.genmod.families import links as L


def fit_logit(df_resp, link="logit"):
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.generalized_linear_model import GLM
    from statsmodels.tools.tools import add_constant

    if link == "logit":
        link_func = L.logit
    elif link == "cloglog":
        link_func = L.cloglog

    model = GLM(
        df_resp["known"], add_constant(df_resp["zipf"]), family=Binomial(link=link_func)
    ).fit()
    print(model.summary())
    print(model.params)
