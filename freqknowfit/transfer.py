import sys

import numpy
import pandas
import seaborn as sns
import torch
import torch.nn
from matplotlib import pyplot as plt
from statsmodels.nonparametric.kde import KDEUnivariate

from vocabmodel.transfer.rpscl_oneinf import fit_logit
from vocabmodel.utils.freq import add_zipfs

# from statsmodels.nonparametric.smoothers_lowess import lowess


def print_model(model):
    print("Weight", model.linear.weight.detach())
    print("Bias", model.linear.bias.detach())
    print("Background", model.sigmoid(model.offset.detach()))


df = pandas.read_parquet(sys.argv[1])
add_zipfs(df)
df["known"] = df["score"] >= 5

fig, ax = plt.subplots(5, 3)
ax_flat = ax.flatten()
# ax = sns.regplot(x="zipf", y="known", data=df_resp, lowess=True)
# plt.show()


def draw_plot(df_resp, ax):
    df_resp = df_resp[df_resp["zipf"] >= 3]
    known_mask = df_resp["known"]
    known_count = known_mask.sum()
    known_zipfs = df_resp[known_mask]["zipf"].to_numpy()
    kde_known = KDEUnivariate(known_zipfs)
    kde_known.fit()
    unknown_mask = ~known_mask
    unknown_count = unknown_mask.sum()
    unknown_zipfs = df_resp[unknown_mask]["zipf"].to_numpy()
    kde_unknown = KDEUnivariate(unknown_zipfs)
    kde_unknown.fit()
    total_count = known_count + unknown_count

    """
    model = fit_offset_logit(
        torch.as_tensor(df_resp["zipf"].to_numpy()).unsqueeze(-1).float(),
        torch.as_tensor(df_resp["known"].to_numpy()).float(),
    )
    print_model(model)
    """

    samples = numpy.linspace(0, 7, 1000)
    known_y = kde_known.evaluate(samples) * known_count / total_count
    unknown_y = kde_unknown.evaluate(samples) * unknown_count / total_count
    support = known_y + unknown_y
    sns.lineplot(x=samples, y=known_y / support, ax=ax)
    sns.lineplot(x=samples, y=support, ax=ax)
    sns.regplot(x="zipf", y="known", data=df_resp, logistic=True, ax=ax, ci=None)
    fit_logit(df_resp)
    """
    with torch.no_grad():
        offset_logit_y = model.forward(
            torch.as_tensor(samples).unsqueeze(-1).float()
        ).numpy()[:, 0]
        print(offset_logit_y)
    sns.lineplot(
        x=samples, y=offset_logit_y, ax=ax,
    )
    """


for resp_idx in range(1, 16):  # 16
    print(f"Plotting {resp_idx}/15")
    resp_ax = ax_flat[resp_idx - 1]
    df_resp = df[df["respondent"] == resp_idx]
    draw_plot(df_resp, resp_ax)
    print(f"Done {resp_idx}/15")
fig.tight_layout()
plt.show()
