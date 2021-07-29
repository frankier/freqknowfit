from dataclasses import dataclass
from enum import Enum
import click
import numpy
import pandas
import seaborn as sns
from matplotlib import pyplot as plt, rcParams
from statsmodels.nonparametric.kde import KDEUnivariate
from scipy import stats as ss
from math import ceil

from .nonparametric import NonParametricEstimator
from .utils import IterFittedResps

# from statsmodels.nonparametric.smoothers_lowess import lowess


class LinkFunc(Enum):
    LOGIT = 1
    PROBIT = 2
    CLOGLOG = 3


LINKS = {
    LinkFunc.LOGIT: ss.logistic.cdf,  # lambda x: 1 / (1 + numpy.exp(-x)),
    LinkFunc.PROBIT: ss.norm.cdf,
    LinkFunc.CLOGLOG: lambda x: numpy.log(-numpy.log(1 - x))
}


@dataclass
class RegressionConfig:
    pred_unk: bool = False
    link: LinkFunc = LinkFunc.LOGIT
    zi: bool = False
    oi: bool = False


OI_CONF = RegressionConfig(oi=True)
NUM_SAMPLE_POINTS = 2048
ZIPF_X = numpy.linspace(0, 7.5, NUM_SAMPLE_POINTS)


def sample_reg_curve(fit_conf, fit_row, x):
    assert not fit_conf.pred_unk
    link = LINKS[fit_conf.link]
    print("fit_row", fit_row)
    theta = x * fit_row["zipf_coef"].iloc[0] + fit_row["const_coef"].iloc[0]
    y_uninflated = link(theta)
    if fit_conf.oi or fit_conf.zi:
        inflate_prob = LINKS[LinkFunc.LOGIT](fit_row["phi_coef"].iloc[0])
        if fit_conf.oi:
            assert not fit_conf.zi
            return inflate_prob + (1 - inflate_prob) * y_uninflated
        else:
            return (1 - inflate_prob) * y_uninflated
    else:
        return y_uninflated


def regplot(ax, x, y, data, fit_conf, fit_row):
    ax.scatter(data[x], data[y])
    ax.plot(ZIPF_X, sample_reg_curve(fit_conf, fit_row, ZIPF_X))

# sns.regplot(
#   x="zipf",
#   y="known",
#   data=df_resp,
#   logistic=True,
#   ax=ax,
#   ci=None
# )


def plot_nonparametric_fit(df_resp, ax):
    nonparametric_est = NonParametricEstimator(df_resp, "zipf", "known")
    est_result = nonparametric_est.evaluate(ZIPF_X)
    sns.lineplot(x=ZIPF_X, y=est_result.transfer(), ax=ax)
    sns.lineplot(x=ZIPF_X, y=est_result.support(), ax=ax)


def draw_plot_using_fit(df_resp, fit_conf, fit_row, ax):
    plot_nonparametric_fit(df_resp, ax)
    regplot(
        x="zipf", y="known", data=df_resp, ax=ax,
        fit_conf=fit_conf, fit_row=fit_row
    )


def draw_plot(df_resp, ax, create_fit=True):
    plot_nonparametric_fit(df_resp, ax)
    if create_fit:
        sns.regplot(x="zipf", y="known", data=df_resp, ax=ax, logistic=True, ci=None)
    else:
        ax.scatter(df_resp["zipf"], df_resp["known"])


def mk_subplots(num_plots):
    low_dim = int(num_plots ** 0.5)
    high_dim = int(ceil(num_plots / low_dim))
    rcParams["figure.figsize"] = 4 * high_dim, 4 * low_dim
    fig, ax = plt.subplots(high_dim, low_dim)
    return fig, ax.flatten()


@click.command()
@click.argument("dfin", type=click.Path(exists=True))
@click.argument("imgout")
@click.option("--fit", type=click.Path(exists=True))
@click.option("--no-add-fit/--add-fit")
def main(dfin, imgout, fit, no_add_fit):
    if fit is not None:
        fitted_resps = IterFittedResps(dfin, fit)
        fig, ax_flat = mk_subplots(len(fitted_resps))
        for resp_idx, (resp_ax, (resp_id, df_resp, fit_row)) in enumerate(zip(ax_flat, fitted_resps)):
            progress = f"{resp_idx + 1}/{len(fitted_resps)}"
            print(f"Plotting {progress} ({resp_id})")
            print(df_resp)
            draw_plot_using_fit(df_resp, OI_CONF, fit_row, resp_ax)
            print(f"Done {progress}")
    else:
        df = pandas.read_parquet(dfin)
        resp_grouped = df.groupby("respondent")
        fig, ax_flat = mk_subplots(len(resp_grouped))
        print(len(ax_flat), len(resp_grouped))
        for resp_idx, (resp_ax, (resp_id, df_resp)) in enumerate(zip(ax_flat, resp_grouped)):
            progress = f"{resp_idx + 1}/{len(resp_grouped)}"
            print(f"Plotting {progress} ({resp_id})")
            print(df_resp)
            draw_plot(df_resp, resp_ax, create_fit=not no_add_fit)
            print(f"Done {progress}")
    fig.tight_layout()
    plt.savefig(imgout)


if __name__ == "__main__":
    main()
