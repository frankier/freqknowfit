import pandas


class IterFittedResps:
    def __init__(self, dfin, fitin):
        self.dfin = dfin
        self.fitin = fitin
        df = pandas.read_parquet(dfin)
        self.fit_df = pandas.read_parquet(fitin)
        self.groups = df.groupby("respondent")

    def __len__(self):
        return len(self.groups)

    def __iter__(self):
        for resp_idx, df_resp in self.groups:
            fit_row = self.fit_df[self.fit_df["respondent"] == str(resp_idx)]
            if len(fit_row) != 1:
                raise ValueError(f"Couldn't get fitted model for {resp_idx}")
            yield resp_idx, df_resp, fit_row
