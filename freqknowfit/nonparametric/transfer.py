from dataclasses import dataclass
from enum import Enum
import click
import numpy
import pandas
import seaborn as sns
from matplotlib import pyplot as plt, rcParams
from scipy import stats as ss
from math import ceil

from .nonparametric import NonParametricEstimator
from .utils import zip_fits

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


def plot_nonparametric_fit(df_resp, ax, ordinal=False, bw="normal_reference", add_support=False):
    if ordinal:
        for score in sorted(df_resp["score"].unique())[1:]:
            nonparametric_est = NonParametricEstimator(df_resp["zipf"], df_resp["score"] >= score, bw=bw)
            est_result = nonparametric_est.evaluate(ZIPF_X)
            sns.lineplot(x=ZIPF_X, y=est_result.transfer(), ax=ax)
    else:
        nonparametric_est = NonParametricEstimator.from_df(
            df_resp,
            "zipf",
            "known",
            bw=bw
        )
        est_result = nonparametric_est.evaluate(ZIPF_X)
        sns.lineplot(x=ZIPF_X, y=est_result.transfer(), ax=ax)
    if add_support:
        sns.lineplot(x=ZIPF_X, y=est_result.support(), ax=ax)


def draw_plot_using_fit(df_resp, fit_conf, fit_row, ax, ordinal=False, bw="normal_reference", add_support=False, add_kde=True):
    assert not ordinal
    if add_kde:
        plot_nonparametric_fit(df_resp, ax, ordinal=ordinal, bw=bw, add_support=add_support)
    regplot(
        x="zipf", y="known", data=df_resp, ax=ax,
        fit_conf=fit_conf, fit_row=fit_row
    )


def draw_plot(df_resp, ax, create_fit=True, ordinal=False, datapoints="none", bw="normal_reference", add_support=False, add_kde=True):
    if add_kde:
        plot_nonparametric_fit(df_resp, ax, ordinal=ordinal, bw=bw, add_support=add_support)
    if create_fit:
        assert not ordinal
        sns.regplot(x="zipf", y="known", data=df_resp, ax=ax, logistic=True, ci=None)
    else:
        if ordinal:
            min_score = df_resp["score"].min()
            max_score = df_resp["score"].max()
            y = (df_resp["score"] - min_score) / (max_score - min_score)
        else:
            y = df_resp["known"]
        if datapoints == "stripplot":
            sns.stripplot(x=df_resp["zipf"], y=y, size=1, jitter=5000)
        elif datapoints == "scatterplot":
            ax.scatter(x=df_resp["zipf"], y=y)


def mk_subplots(num_plots, size_inches=4):
    low_dim = int(num_plots ** 0.5)
    high_dim = int(ceil(num_plots / low_dim))
    fig, ax = plt.subplots(high_dim, low_dim)
    fig.set_size_inches(size_inches * low_dim * 2, size_inches * high_dim)
    return fig, (ax.flatten() if isinstance(ax, numpy.ndarray) else [ax])


@click.command()
@click.argument("dfin", type=click.Path(exists=True))
@click.argument("imgout")
@click.option("--fit", type=click.Path(exists=True))
@click.option("--no-add-fit/--add-fit")
@click.option("--ordinal/--add-fit")
@click.option("--datapoints", type=click.Choice(["stripplot", "scatterplot", "none"]))
@click.option("--add-support/--no-support")
@click.option("--no-add-kde/--add-kde")
@click.option("--respondent", multiple=True)
@click.option("--bw", default="normal_reference")
@click.option("--size-inches", type=int, default=4)
def main(
    dfin,
    imgout,
    fit,
    no_add_fit,
    ordinal,
    datapoints,
    add_support,
    no_add_kde,
    respondent,
    bw,
    size_inches
):
    if bw[0].isnumeric():
        bw = float(bw)
    df = pandas.read_parquet(dfin)
    if respondent:
        if respondent[0].isnumeric():
            respondent = [int(r) for r in respondent]
        for r in respondent:
            if not (df["respondent"] == r).any():
                raise click.ClickException(f"Respondent not found: {r}")
        df = df[df["respondent"].isin(respondent)]
    resp_grouped = df.groupby("respondent")
    fig, ax_flat = mk_subplots(len(resp_grouped), size_inches)
    if fit is not None:
        fit_df = pandas.read_parquet(fit)
        if respondent:
            fit_df = fit_df[fit_df["respondent"].isin(respondent)]
        for resp_idx, (resp_ax, (resp_id, df_resp, fit_row)) in \
                enumerate(zip(ax_flat, zip_fits(resp_grouped, fit_df))):
            progress = f"{resp_idx + 1}/{len(fitted_resps)}"
            print(f"Plotting {progress} ({resp_id})")
            print(df_resp)
            draw_plot_using_fit(df_resp, OI_CONF, fit_row, resp_ax, ordinal=ordinal, bw=bw, add_support=add_support, add_kde=not no_add_kde)
            print(f"Done {progress}")
    else:
        for resp_idx, (resp_ax, (resp_id, df_resp)) in \
                enumerate(zip(ax_flat, resp_grouped)):
            progress = f"{resp_idx + 1}/{len(resp_grouped)}"
            print(f"Plotting {progress} ({resp_id})")
            print(df_resp)
            draw_plot(df_resp, resp_ax, create_fit=not no_add_fit, ordinal=ordinal, datapoints=datapoints, bw=bw, add_support=add_support, add_kde=not no_add_kde)
            print(f"Done {progress}")
    fig.tight_layout()
    plt.savefig(imgout)


if __name__ == "__main__":
    main()
