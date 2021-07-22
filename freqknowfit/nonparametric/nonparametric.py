from statsmodels.nonparametric.kde import KDEUnivariate


class NonParametricResult:
    def __init__(self, known_y, unknown_y):
        self.known_y = known_y
        self.unknown_y = unknown_y
        self._support = known_y + unknown_y

    def transfer(self):
        return self.known_y / self._support

    def support(self):
        return self._support


class NonParametricEstimator:
    def __init__(self, df, x, y):
        known_mask = df[y]
        self.known_count = known_mask.sum()
        known_zipfs = df[known_mask][x].to_numpy()
        self.kde_known = KDEUnivariate(known_zipfs)
        self.kde_known.fit()
        unknown_mask = ~known_mask
        self.unknown_count = unknown_mask.sum()
        unknown_zipfs = df[unknown_mask][x].to_numpy()
        self.kde_unknown = KDEUnivariate(unknown_zipfs)
        self.kde_unknown.fit()
        self.total_count = self.known_count + self.unknown_count

    def evaluate(self, samples):
        known_y = (
            self.kde_known.evaluate(samples)
            * self.known_count / self.total_count
        )
        unknown_y = (
            self.kde_unknown.evaluate(samples)
            * self.unknown_count / self.total_count
        )
        return NonParametricResult(known_y, unknown_y)
