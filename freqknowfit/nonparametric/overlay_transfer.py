import click
import scipy
import datashader as ds
import datashader.transfer_functions as tf
import pandas as pd
import numpy as np
from .nonparametric import NonParametricEstimator
from .transfer_curves import inv_cloglog
from matplotlib import pyplot as plt


NUM_SAMPLE_POINTS = 2048


def resample_nonparameteric(df, x_in, x_out):
    groups = df.groupby("respondent")
    x_series = pd.Series(x_in)
    curves = []
    for resp_idx, df_resp in groups:
        est = NonParametricEstimator(df_resp, "zipf", "known")
        transformed_curve = get_overlay(x_in, x_out, inv_cloglog, est)
        #curve = est.evaluate(x).transfer()
        #curves.append(curve)
        if transformed_curve is None:
            continue
        curves.append(transformed_curve)
    return pd.DataFrame(curves, columns=x_series)


def mk_curve_transformer(estimator):
    def transformed_curve(x, x_shift, x_scale, y_offset):
        return (
            y_offset
            + (1 - y_offset)
            * estimator.evaluate(x_scale * (x - x_shift)).transfer()
        )
    return transformed_curve


def get_overlay(x_in, x_out, trans_func, estimator):
    transformed_curve = mk_curve_transformer(estimator)
    support = estimator.evaluate(x_in).support()
    try:
        popt, pcov = scipy.optimize.curve_fit(
            mk_curve_transformer(estimator),
            x_in,
            trans_func(x_in),
            (0, 1, 0),
            sigma=1 / np.sqrt(support),
            absolute_sigma=True
        )
        return transformed_curve(x_out, *popt)
    except RuntimeError:
        import traceback
        traceback.print_exc()
        return None


def plot_datashader(std_x, resampled_df, imgout):
    cvs = ds.Canvas(plot_height=2048, plot_width=2048)
    y_list = list(resampled_df.columns.to_numpy())
    agg = cvs.line(resampled_df, x=std_x, y=y_list, agg=ds.count(), axis=1)
    img = tf.shade(agg, how='eq_hist')
    ds.utils.export_image(
        img=img,
        filename=imgout,
        fmt=".png",
        background="white"
    )


def plot_mpl(std_x, resampled_df, imgout):
    plt.plot(std_x, resampled_df.to_numpy().T)
    plt.savefig(imgout)


@click.command()
@click.argument("dfin", type=click.Path(exists=True))
@click.argument("imgout")
def main(dfin, imgout):
    df = pd.read_parquet(dfin)
    zipf_x = np.linspace(0, 7, NUM_SAMPLE_POINTS)
    std_x = np.linspace(0, 1, NUM_SAMPLE_POINTS)
    resampled_df = resample_nonparameteric(df, zipf_x, std_x)
    #plot_datashader(std_x, resampled_df, imgout)
    plot_mpl(std_x, resampled_df, imgout)


if __name__ == "__main__":
    main()
