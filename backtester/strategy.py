from typing import Literal, Union

import numpy as np
import pandas as pd

from .options import Loader
from .utils import find_option_by_desc, find_option_exact


class Leg:
    def __init__(self, dates: np.ndarray, loader: Loader, last_open: np.datetime64):
        self.dates = dates
        self.loader = loader
        self.last_open = last_open

        self.ticker: str = None

        self.side: Literal["l", "s"] = None  # L/S
        self.quantity: int = None

        self.opt_type: Literal["call", "put"] = None
        self.tenor_day: int = None
        self.moneyness: float = None

        self.open_frequency_d: Union[int, Literal["roll"]] = None  # 7D / 30D... etc
        self.close_n_tday_before: int = None

        self.open_px: Literal["bid", "ask", "mid"] = None
        self.close_px: Literal["bid", "ask", "mid"] = None

        self.trades = None
        self.decider = None  # func(date) -> bool

    def generate_trade_dates(self):

        open_dates = []
        options = []
        close_dates = []

        open_dates.append(self.dates[0])

        while open_dates[-1] <= self.last_open:
            expiration = self.get_date(open_dates[-1], self.tenor_day)
            if not (expiration):
                break

            option = find_option_by_desc(
                self.loader,
                open_dates[-1],
                self.ticker,
                expiration,
                self.opt_type,
                self.moneyness,
            )

            options.append(option)

            close_date = self.dates[
                self.check_date(option.expiration) - self.close_n_tday_before
            ]

            if close_date <= open_dates[-1]:
                raise InvalidCloseDateError

            close_dates.append(close_date)

            if self.open_frequency_d == "roll":
                open_dates.append(option.expiration)
            else:
                open_date = self.get_date(open_dates[-1], self.open_frequency_d)
                if open_date:
                    open_dates.append(open_date)
                else:
                    break

        if self.open_frequency_d == "roll":
            open_dates.pop()

        return list(zip(open_dates, close_dates, options))

    def get_realized_pnl(self):
        if not (self.trades):
            self.trades = self.generate_trade_dates()

        df = pd.DataFrame(
            np.zeros(len(self.dates)), index=self.dates, columns=["realized_pnl"]
        )

        decider = self.decider
        for open_date, close_date, option in self.trades:
            if decider:
                if decider(open_date):
                    pass
                else:
                    continue

            # can just get_attr... but I've already loaded the data which took like 5min... TODO!
            if self.open_px == "mid":
                open_px = option.mid()
            else:
                open_px = getattr(option, self.open_px)

            closing_option = find_option_exact(self.loader, close_date, option)
            if self.close_px == "mid":
                close_px = closing_option.mid()
            else:
                close_px = getattr(closing_option, self.close_px)

            if self.side == "l":
                multiplier = self.quantity * 100
            else:
                multiplier = -1 * self.quantity * 100

            df.loc[close_date] += self.get_multiplier() * (close_px - open_px)

        return df

    def get_unrealized_pnl(self):
        if not (self.trades):
            self.trades = self.generate_trade_dates()

        df = pd.DataFrame(
            np.zeros(len(self.dates)), index=self.dates, columns=["unrealized_pnl"]
        )

        decider = self.decider

        for open_date, close_date, option in self.trades:
            if decider:
                if decider(open_date):
                    pass
                else:
                    continue
            # can just get_attr... but I've already loaded the data which took like 5min... TODO!
            curr_date = open_date

            if self.open_px == "mid":
                open_px = option.mid()
            else:
                open_px = getattr(option, self.open_px)

            while curr_date <= close_date:
                curr_option = find_option_exact(self.loader, curr_date, option)

                if self.close_px == "mid":
                    curr_px = curr_option.mid()
                else:
                    curr_px = getattr(curr_option, self.close_px)

                df.loc[curr_date] += self.get_multiplier() * (curr_px - open_px)

                open_px = curr_px

                date_idx = self.check_date(curr_date)

                if date_idx < len(self.dates) - 1:
                    curr_date = self.dates[date_idx + 1]
                else:
                    break

        return df

    def get_greek(self, greek: Literal["delta", "gamma", "theta", "vega", "rho"]):

        if not (self.trades):
            self.trades = self.generate_trade_dates()

        df = pd.DataFrame(np.zeros(len(self.dates)), index=self.dates, columns=[greek])

        decider = self.decider

        for open_date, close_date, option in self.trades:
            curr_date = open_date

            if decider:
                if decider(open_date):
                    pass
                else:
                    continue

            while curr_date <= close_date:
                curr_option = find_option_exact(self.loader, curr_date, option)

                df.loc[curr_date] += self.get_multiplier() * curr_option.greeks[greek]

                date_idx = self.check_date(curr_date)

                if date_idx < len(self.dates) - 1:
                    curr_date = self.dates[date_idx + 1]
                else:
                    break

        return df

    def get_multiplier(self):
        # store this as attribute so that it doesn't get run over and over again...
        if self.side == "l":
            multiplier = self.quantity * 100
        else:
            multiplier = -1 * self.quantity * 100

        return multiplier

    def get_date(self, date, days_delta):
        delta = np.timedelta64(0, "D")
        target_date = date + days_delta

        while not (self.check_date(target_date + delta)) or not (
            self.check_date(target_date - delta)
        ):
            delta += np.timedelta64(1, "D")

            if delta > np.timedelta64(5, "D"):
                return False

        return (
            target_date + delta
            if self.check_date(target_date + delta)
            else target_date - delta
        )

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

        df["cummulative_unreal_pnl"] = df["unrealized_pnl"].cumsum()
        df["cummulative_real_pnl"] = df["realized_pnl"].cumsum()
        return df

    def __construct_df(self, leg):
        unrealized = leg.get_unrealized_pnl()
        realized = leg.get_realized_pnl()
        delta = leg.get_greek("delta")
        gamma = leg.get_greek("gamma")
        theta = leg.get_greek("theta")
        vega = leg.get_greek("vega")
        rho = leg.get_greek("rho")

        df = pd.concat([unrealized, realized, delta, gamma, theta, vega, rho], axis=1)

        return df
