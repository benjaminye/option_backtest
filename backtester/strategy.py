from typing import Literal, Union

import numpy as np
import pandas as pd

from .options import Loader


class Leg:
    def __init__(self, loader: Loader, start_date: np.datetime64):
        self.loader = loader
        self.start_date = start_date
        self.dates = self.__get_dates()

        self.ticker: str = None

        self.side: Literal["l", "s"] = None  # L/S
        self.quantity: int = None

        self.opt_type: Literal["call", "put"] = None
        self.tenor_day: int = 0
        self.moneyness: float = None

        self.open_frequency_d: Union[int, Literal["roll"]] = None  # 7D / 30D... etc
        self.close_n_tday_before: int = None

        self.open_px: Literal["bid", "ask", "mid"] = None
        self.close_px: Literal["bid", "ask", "mid"] = None

        self.trades = None
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
        last_open = last_day - self.tenor_day

        return last_open

    def generate_trades(self):
        open_dates = []
        trades = []
        close_dates = []

        open_date = self.dates[0]

        while open_date <= self.last_open:
            target_expiration = open_date + self.tenor_day
            open_dates.append(open_date)

            option = self.loader.find_option_by_desc(
                open_date, self.ticker, target_expiration, self.opt_type, self.moneyness
            )

            if option.expiration not in self.dates:
                open_dates.pop()
                break

            trades.append(option)

            target_close_date = self.dates[
                self.check_date(option.expiration) - self.close_n_tday_before
            ]

            if target_close_date < open_date or target_close_date > option.expiration:
                raise InvalidCloseDateError

            close_dates.append(target_close_date)

            if self.open_frequency_d == "roll":
                open_date = option.expiration
            else:
                open_date = self.get_date(open_date, self.open_frequency_d)

                if not (open_date):
                    break

            if not (self.decider(open_date)):
                open_dates.pop()
                trades.pop()
                close_dates.pop()

        return list(zip(open_dates, close_dates, trades))

    def get_realized_pnl(self):
        if not (self.trades):
            self.trades = self.generate_trades()

        df = pd.DataFrame(
            np.zeros(len(self.dates)), index=self.dates, columns=["realized_pnl"]
        )

        for open_date, close_date, option in self.trades:
            open_px = getattr(option, self.open_px)

            closing_option = self.loader.find_option_exact(close_date, option)
            close_px = getattr(closing_option, self.close_px)

            df.loc[close_date] += self.__get_multiplier() * (close_px - open_px)

        return df

    def get_unrealized_pnl(self):
        if not (self.trades):
            self.trades = self.generate_trades()

        df = pd.DataFrame(
            np.zeros(len(self.dates)), index=self.dates, columns=["unrealized_pnl"]
        )

        for open_date, close_date, option in self.trades:
            curr_date = open_date
            open_px = getattr(option, self.open_px)

            while curr_date <= close_date:

                curr_option = self.loader.find_option_exact(curr_date, option)
                curr_px = getattr(curr_option, self.close_px)

                df.loc[curr_date] += self.__get_multiplier() * (curr_px - open_px)

                open_px = curr_px
                date_idx = self.check_date(curr_date)
                curr_date = self.dates[date_idx + 1]

        return df

    def get_greek(self, greek: Literal["delta", "gamma", "theta", "vega", "rho"]):
        if not (self.trades):
            self.trades = self.generate_trades()

        df = pd.DataFrame(np.zeros(len(self.dates)), index=self.dates, columns=[greek])

        for open_date, close_date, option in self.trades:
            curr_date = open_date

            while curr_date <= close_date:

                curr_option = self.loader.find_option_exact(curr_date, option)

                df.loc[curr_date] += self.__get_multiplier() * curr_option.greeks[greek]

                date_idx = self.check_date(curr_date)
                curr_date = self.dates[date_idx + 1]

        return df

    def __get_multiplier(self):
        # store this as attribute so that it doesn't get run over and over again...
        if self.side == "l":
            multiplier = self.quantity * 100
        else:
            multiplier = self.quantity * -100

        return multiplier

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


class InvalidCloseDateError(Exception):
    pass


class Strategy:
    def __init__(self, legs: list):
        self.legs = legs

    def get_stats(self):
        df = self.__construct_df(self.legs[0])

        if len(self.legs) > 1:
            for leg in self.legs[1:]:
                df += self.__construct_df(leg)

        df["cumulative_real_pnl"] = df["realized_pnl"].cumsum()
        df["cumulative_unreal_pnl"] = df["unrealized_pnl"].cumsum()
        df["drawdown"] = self.__get_drawdown(df["cumulative_unreal_pnl"])
        return df

    def __construct_df(self, leg):
        realized = leg.get_realized_pnl()
        unrealized = leg.get_unrealized_pnl()
        delta = leg.get_greek("delta")
        gamma = leg.get_greek("gamma")
        theta = leg.get_greek("theta")
        vega = leg.get_greek("vega")
        rho = leg.get_greek("rho")

        df = pd.concat([realized, unrealized, delta, gamma, theta, vega, rho], axis=1)

        return df

    def __get_drawdown(self, series):
        series = series.to_numpy()
        return series - np.fmax.accumulate(series)
