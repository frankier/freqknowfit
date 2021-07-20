import click
import pandas


@click.command()
@click.argument("measure")
@click.argument("treat1", type=click.Path(exists=True))
@click.argument("treat2", type=click.Path(exists=True))
def main(measure, treat1, treat2):
    df1 = pandas.read_parquet(treat1)
    df2 = pandas.read_parquet(treat2)
    aics1 = df1[measure].to_numpy()
    aics2 = df2[measure].to_numpy()
    if len(aics1) != len(aics2):
        print(f"WARNING: Lengths of treat1 and treat2 differ ({len(aics1)} vs {len(aics2)})")
        print("Truncating to shortest")
        trunk_len = min(len(aics1), len(aics2))
        aics1 = aics1[:trunk_len]
        aics2 = aics2[:trunk_len]
    print("Treatment 1 better {:.2f}".format(100 * (aics1 > aics2).mean()))
    print("Treatment 2 better {:.2f}".format(100 * (aics2 > aics1).mean()))


if __name__ == "__main__":
    main()
