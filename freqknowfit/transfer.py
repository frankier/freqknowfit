import click
import numpy
import pandas
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.nonparametric.kde import KDEUnivariate

from .nonparametric import NonParametricEstimator

# from statsmodels.nonparametric.smoothers_lowess import lowess


def print_model(model):
    print("Weight", model.linear.weight.detach())
    print("Bias", model.linear.bias.detach())
    print("Background", model.sigmoid(model.offset.detach()))


def draw_plot(df_resp, ax):
    df_resp = df_resp[df_resp["zipf"] >= 3]
    nonparametric_est = NonParametricEstimator(df_resp, "zipf", "known")
    samples = numpy.linspace(0, 7, 1000)
    est_result = nonparametric_est.evaluate(samples)
    sns.lineplot(x=samples, y=est_result.transfer(), ax=ax)
    sns.lineplot(x=samples, y=est_result.support(), ax=ax)
    sns.regplot(
        x="zipf",
        y="known",
        data=df_resp,
        logistic=True,
        ax=ax,
        ci=None
    )


@click.command()
@click.argument("dfin", type=click.Path(exists=True))
def main(dfin):
    df = pandas.read_parquet(dfin)
    df["known"] = df["score"] >= 5
    fig, ax = plt.subplots(5, 3)
    ax_flat = ax.flatten()

    groups = df.groupby("respondent")
    for resp_ax, (resp_idx, df_resp) in zip(ax_flat, groups):
        progress = f"{resp_idx}/{len(groups)}"
        print(f"Plotting {progress}")
        print(df_resp)
        draw_plot(df_resp, resp_ax)
        print(f"Done {progress}")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
