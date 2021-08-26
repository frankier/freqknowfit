Fitting various linear-ish and non-parametric models for vocabulary inventory
data for fun and profit.

## Getting started

To start with you need to get your vocabulary data in the right format. The
format assumed here is the "enriched" format which has word frequencies
embedded and is produced by
[vocabaqdata](https://github.com/frankier/vocabaqdata/). These files are named
e.g. `$DATASET.enriched.parquet`. As an example in the following we will use
`svl12k.enriched.parquet`. So go to vocabaqdata and get one of those files first.

Then install using Poetry:

    $ poetry install

## Nonparametric models --- visualisation

You can plot nonparametric transfer curves regressing using frequency like so:

    $ poetry run python -m freqknowfit.nonparametric.transfer \
        /path/to/svl12k.enriched.parquet \
	svl12k_transfer.png

## Parametric models --- goodness of fit

The Snakefile contains workflows to fit various models to the vocabulary data
and produce statistics about degree of fit AIC and so on. You can run it
(single-threaded) like so:

    $ poetry run snakemake -j1
