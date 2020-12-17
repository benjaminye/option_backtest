from typing import Literal

import pandas as pd

from .options import Loader


class Volatility:
    def __init__(self, loader: Loader):
        self.loader = loader

    def get_vol_surface(
        self,
        ticker,
        date,
        tenors,
        x_axis=Literal["moneyness", "delta", "strike"],
    ):
        chain = self.loader.get_chains_by_date(date)[ticker]

        out = pd.DataFrame(columns=tenors)

        for tenor in tenors:
            expiration = date + tenor

            sub_chain = chain.get_approx_expiry(expiration).chain

            options = list(sub_chain.values())
            options = [option for sublist in options for option in sublist]

            for option in options:
                if option.greeks["vega"] != 0:
                    out.loc[option.strike, tenor] = option.iv

        return out
