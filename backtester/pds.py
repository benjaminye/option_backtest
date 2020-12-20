from typing import Literal, Union

import numpy as np
import pandas as pd

from .options import Loader
from .strategy import InvalidCloseDateError


class PDSStrategy:
    def __init__(self, legs: list):
        self.legs = legs

    def get_stats(self):
        df = self.__construct_df(self.legs[0])

        if len(self.legs) > 1:
            for leg in self.legs[1:]:
                df += self.__construct_df(leg)

        df["drawdown"] = self.__get_drawdown(df["capital"])
        return df

    def __construct_df(self, leg):
        unrealized, capital = leg.get_unrealized_pnl()
        delta = leg.get_greek("delta")
        gamma = leg.get_greek("gamma")
        theta = leg.get_greek("theta")
        vega = leg.get_greek("vega")
        rho = leg.get_greek("rho")

        df = pd.concat([unrealized, delta, gamma, theta, vega, rho, capital], axis=1)

        return df

    def __get_drawdown(self, series):
        series = series.to_numpy()
        return (series - np.fmax.accumulate(series)) / np.fmax.accumulate(series)


class PDS:
    def __init__(self, loader: Loader, start_date: np.datetime64):
        self.loader = loader
        self.start_date = start_date
        self.dates = self.__get_dates()

        self.ticker: str = None

        self.moneyness: tuple = (1.0, 0.9)  # (short_leg, long_leg)
        self.initial_cap: float = None

        self.tenor: tuple = (7, 30)  # (short_leg, long_leg)

        self.open_px: Literal["bid", "ask", "mid"] = None
        self.close_px: Literal["bid", "ask", "mid"] = None

        self.trades = None

        self.capital = None
        self.decider = lambda x: True  # func(date) -> bool

    def __get_dates(self):
        dates = self.loader.list_dates()
        dates = pd.DataFrame(dates)
        dates = dates.where(dates >= self.start_date).dropna()
        dates = dates.values.astype("datetime64[D]").flatten()

        return dates

    @property
    def last_open(self):
        last_day = self.dates[-1]
        last_open = last_day - self.tenor[0]

        return last_open

    def generate_trades(self):
        open_dates = []
        trades_s = []
        trades_l = []
        close_dates = []

        open_date = self.dates[0]

        while open_date <= self.last_open:
            target_expiration_s = open_date + self.tenor[0]
            target_expiration_l = open_date + self.tenor[1]
            open_dates.append(open_date)

            option_s = self.loader.find_option_by_desc(
                open_date, self.ticker, target_expiration_s, "put", self.moneyness[0]
            )

            option_l = self.loader.find_option_by_desc(
                open_date, self.ticker, target_expiration_l, "put", self.moneyness[1]
            )

            if option_s.expiration not in self.dates:
                open_dates.pop()
                break

            trades_s.append(option_s)
            trades_l.append(option_l)

            target_close_date = self.dates[self.check_date(option_s.expiration)]

            close_dates.append(target_close_date)

            open_date = option_s.expiration

            if not (self.decider(open_date)):
                open_dates.pop()
                trades_s.pop()
                trades_l.pop()
                close_dates.pop()

        return list(zip(open_dates, close_dates, trades_s, trades_l))

    def get_unrealized_pnl(self):
        if not (self.trades):
            self.trades = self.generate_trades()

        df = pd.DataFrame(
            np.zeros(len(self.dates)), index=self.dates, columns=["unrealized_pnl"]
        )

        nans = np.empty(len(self.dates))
        nans[:] = np.nan

        df_capital = pd.DataFrame(nans, index=self.dates, columns=["capital"])
        df_capital.iloc[0] = self.initial_cap

        for open_date, close_date, option_s, option_l in self.trades:
            curr_date = open_date
            open_px_s = getattr(option_s, self.open_px)
            open_px_l = getattr(option_l, self.open_px)

            while curr_date <= close_date:
                curr_option_s = self.loader.find_option_exact(curr_date, option_s)
                curr_px_s = getattr(curr_option_s, self.close_px)

                curr_option_l = self.loader.find_option_exact(curr_date, option_l)
                curr_px_l = getattr(curr_option_l, self.close_px)

                if df_capital.loc[open_date].isnull()[0]:
                    curr_capital = 0
                else:
                    curr_capital = df_capital.loc[open_date][0]

                curr_margin = self.get_margin(option_s, option_l)

                df.loc[curr_date] += self.get_multiplier(
                    "s", curr_capital, curr_margin
                ) * (curr_px_s - open_px_s)

                df.loc[curr_date] += self.get_multiplier(
                    "l", curr_capital, curr_margin
                ) * (curr_px_l - open_px_l)

                df_capital.loc[curr_date] = max(
                    self.initial_cap + df.cumsum().iloc[-1][0], 0
                )

                if df_capital.loc[curr_date][0] == 0:
                    break

                open_px_s = curr_px_s
                open_px_l = curr_px_l

                date_idx = self.check_date(curr_date)
                curr_date = self.dates[date_idx + 1]

        df_capital = df_capital.fillna(method="ffill")
        self.capital = df_capital
        return df, df_capital

    def get_greek(self, greek: Literal["delta", "gamma", "theta", "vega", "rho"]):
        if not (self.trades):
            self.trades = self.generate_trades()

        if self.capital is None:
            raise CapitalNotInitializedError

        df = pd.DataFrame(np.zeros(len(self.dates)), index=self.dates, columns=[greek])

        for open_date, close_date, option_s, option_l in self.trades:
            curr_date = open_date

            while curr_date <= close_date:
                curr_option_s = self.loader.find_option_exact(curr_date, option_s)
                curr_option_l = self.loader.find_option_exact(curr_date, option_l)

                if self.capital.loc[open_date].isnull()[0]:
                    curr_capital = 0
                else:
                    curr_capital = self.capital.loc[open_date][0]

                curr_margin = self.get_margin(option_s, option_l)

                df.loc[curr_date] += (
                    self.get_multiplier("s", curr_capital, curr_margin)
                    * curr_option_s.greeks[greek]
                )
                df.loc[curr_date] += (
                    self.get_multiplier("l", curr_capital, curr_margin)
                    * curr_option_l.greeks[greek]
                )

                date_idx = self.check_date(curr_date)
                curr_date = self.dates[date_idx + 1]

        return df

    def get_multiplier(self, side, capital, margin):
        # store this as attribute so that it doesn't get run over and over again...
        if side == "l":
            multiplier = capital // margin * 100
        elif side == "s":
            multiplier = capital // margin * -100

        return multiplier

    def get_margin(self, option_s, option_l):
        return max(option_s.strike - option_l.strike, 0) * 100

    def get_date(self, date, days_delta):
        target_date = date + days_delta
        while not (self.check_date(target_date)):
            days_delta += 1
            target_date = date + days_delta

            if days_delta > 30:
                return False
        return target_date

    def check_date(self, date):
        if np.where(self.dates == date)[0].size == 0:
            return False

        return int(np.where(self.dates == date)[0])


class CapitalNotInitializedError(Exception):
    pass