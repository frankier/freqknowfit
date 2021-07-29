import click
import pandas
from matplotlib import pyplot as plt, rcParams


@click.command()
@click.argument("fitin", type=click.Path(exists=True))
@click.argument("imgout")
def main(fitin, imgout):
    rcParams["figure.figsize"] = 20, 20
    df = pandas.read_parquet(fitin)
    plt.scatter(df["theta"], df["phi_coef"])
    plt.tight_layout()
    plt.savefig(imgout)


if __name__ == "__main__":
    main()
