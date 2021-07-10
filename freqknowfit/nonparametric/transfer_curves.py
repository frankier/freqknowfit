import click
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, logistic


FLOAT_EPS = np.finfo(float).eps


def inv_cloglog(z):
    return 1 - np.exp(-np.exp(z))
    #theta = np.clip(theta, FLOAT_EPS, 1. - FLOAT_EPS)
    #return np.log(-np.log(1 - theta))


@click.command()
@click.argument("imgout")
def main(imgout):
    x = np.linspace(-10, 10, 1024)
    plt.plot(x, inv_cloglog(x), label="cloglog")
    plt.plot(x, norm.cdf(x), label="probit")
    plt.plot(x, logistic.cdf(x), label="logit")
    plt.legend()
    plt.savefig(imgout)


if __name__ == "__main__":
    main()
