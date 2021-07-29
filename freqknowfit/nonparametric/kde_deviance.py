import click
import pandas
import numpy
from scipy.integrate import trapezoid

from .nonparametric import NonParametricEstimator
from .utils import IterFittedResps


SAMPLES = numpy.linspace(0, 7, 1000)


def logistic(m, c):
    return 1 / (1 + numpy.exp(-(c + m * SAMPLES)))


@click.command()
@click.argument("dfin", type=click.Path(exists=True))
@click.argument("fitin", type=click.Path(exists=True))
def main(dfin, fitin):
    cols = {"respondent": [], "mae": [], "mse": [], "weighted_mae": [], "weighted_mse": []}
    for resp_idx, df_resp, fit_row in IterFittedResps(dfin, fitin):
        predictions = logistic(fit_row["zipf_coef"].to_numpy(), fit_row["const_coef"].to_numpy())
        nonparametric_est = NonParametricEstimator(df_resp, "zipf", "known")
        nonparametric_eval = nonparametric_est.evaluate(SAMPLES)
        trans = nonparametric_eval.transfer()
        supp = nonparametric_eval.support()
        cols["respondent"].append(resp_idx)
        cols["mae"].append(trapezoid(numpy.abs(trans - predictions), SAMPLES))
        cols["mse"].append(trapezoid((trans - predictions) ** 2, SAMPLES))
        cols["weighted_mae"].append(trapezoid(supp * numpy.abs(trans - predictions), SAMPLES))
        cols["weighted_mse"].append(trapezoid(supp * (trans - predictions) ** 2, SAMPLES))
    df_out = pandas.DataFrame(cols)
    print(df_out)
    print(df_out.mean())


if __name__ == "__main__":
    main()
