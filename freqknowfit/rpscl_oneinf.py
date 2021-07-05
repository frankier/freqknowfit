from contextlib import contextmanager

import rpy2.robjects as ro
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr


@contextmanager
def convert_r_exceptions():
    try:
        yield
    except RRuntimeError as e:
        try:
            e.context = {"r_traceback": "\n".join(ro.r("unlist(traceback())"))}
        except Exception as traceback_exc:
            e.context = {
                "r_traceback": "(an error occurred while getting traceback from R)",
                "r_traceback_err": traceback_exc,
            }

        raise


def fit_logit(df_resp, link="logit"):
    R_base = importr("base")
    R_vgam = importr("VGAM")
    df_resp["known"] = ~df_resp["known"]
    df_resp = df_resp.astype({"known": "float64"})
    with localconverter(
        ro.default_converter + pandas2ri.converter
    ), convert_r_exceptions():
        print(df_resp.head())
        print("zipf finite", all(R_base.is_finite(df_resp["zipf"])))
        print("known finite", all(R_base.is_finite(df_resp["known"])))
        fit = R_vgam.vglm(
            "cbind(known, 1 - known) ~ zipf",
            R_vgam.zibinomialff,
            data=df_resp,
            trace=True,
            stepsize=0.5,
        )
        R_base.summary(fit)
