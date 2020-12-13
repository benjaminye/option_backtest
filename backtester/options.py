from typing import Literal
import copy
import csv

import numpy as np


class Option:
    """
    Defines an option
    """

    def __init__(
        self,
        ticker: str,
        quote_date: np.datetime64,
        expiration: np.datetime64,
        strike: float,
        opt_type: Literal["call", "put"],
        price: dict,
        bid: float,
        ask: float,
        underlying_px: float,
        iv: float,
        greeks: dict,
    ):
        self.ticker = ticker.upper()
        self.quote_date = quote_date
        self.expiration = expiration
        self.strike = strike
        self.opt_type = opt_type
        self.price = price
        self.bid = bid
        self.ask = ask
        self.underlying_px = underlying_px
        self.iv = iv
        self.greeks = greeks

    def __repr__(self):
        return f"""
        Quote Date: {self.quote_date}
        -----------------------------
        Ticker: {self.ticker}
        Underlying Price: {self.underlying_px}
        -----------------------------
        Type: {self.opt_type.upper()}
        Expiry: {self.expiration}
        Strike: {self.strike}
        -----------------------------
        Bid: {self.bid} 
        Ask: {self.ask}
        IV: {self.iv}
        -----------------------------
        Open:{self.price['open']}
        High:{self.price['high']}
        Low:{self.price['low']}
        Close: {self.price['close']}
        -----------------------------
        Delta: {self.greeks['delta']}
        Gamma: {self.greeks['gamma']}
        Theta: {self.greeks['theta']}
        Vega: {self.greeks['vega']}
        Rho: {self.greeks['rho']}
        """

    def moneyness(self):
        return self.strike / self.underlying_px

    def dte(self):
        return (self.expiration - self.quote_date).astype(int)

    def spread(self):
        return self.ask - self.bid

    def mid(self):
        return (self.bid + self.ask) / 2


class OptionChain:
    def __init__(self, ticker: str, quote_date: np.datetime64):
        self.ticker = ticker.upper()
        self.quote_date = quote_date
        self.chain = {}

    def add_option(self, option: Option):
        if option.ticker != self.ticker:
            raise TickerMismatchException
        if option.quote_date != self.quote_date:
            raise DateMismatchException

        if option.expiration not in self.chain:
            self.chain[option.expiration] = []

        self.chain[option.expiration].append(option)

    def get_approx_expiry(self, expiry: np.datetime64):
        dates = list(self.chain.keys())
        days_delta = [abs(date - expiry).astype(int) for date in dates]

        match = dates[days_delta.index(min(days_delta))]

        new_chain = OptionChain(self.ticker, self.quote_date)

        for option in self.chain[match]:
            new_chain.add_option(option)

        return new_chain

    def get_type(self, opt_type: Literal["call", "put"]):
        options_list = list(self.chain.values())
        options_list = [option for sublist in options_list for option in sublist]

        new_chain = OptionChain(self.ticker, self.quote_date)

        for option in options_list:
            if option.opt_type.lower() == opt_type.lower():
                new_chain.add_option(option)

        return new_chain

    def get_approx_moneyness(self, tgt_moneyness: float):
        new_chain = OptionChain(self.ticker, self.quote_date)
        results = []

        chain = copy.deepcopy(self.chain)
        for _, option_list in chain.items():
            moneyness_list = [option.moneyness() for option in option_list]
            moneyness_delta = [
                abs(moneyness - tgt_moneyness) for moneyness in moneyness_list
            ]

            target = min(moneyness_delta)

            while len(option_list):
                try:
                    results.append(option_list.pop(moneyness_delta.index(target)))
                    moneyness_delta.pop(moneyness_delta.index(target))
                except ValueError:
                    break

        for option in results:
            new_chain.add_option(option)

        return new_chain

    def get_option(self, expiry, opt_type, moneyness) -> Option:
        result = (
            self.get_type(opt_type)
            .get_approx_expiry(expiry)
            .get_approx_moneyness(moneyness)
        )
        return list(result.chain.values())[0][0]


class TickerMismatchException(Exception):
    pass


class DateMismatchException(Exception):
    pass


class Loader:
    def __init__(self):
        self.column_idx = {
            "ticker": 0,
            "quote_date": 1,
            "expiration": 3,
            "strike": 4,
            "opt_type": 5,
            "open": 6,
            "high": 7,
            "low": 8,
            "close": 9,
            "bid": 12,
            "ask": 14,
            "underlying_px": 18,
            "iv": 19,
            "delta": 20,
            "gamma": 21,
            "theta": 22,
            "vega": 23,
            "rho": 24,
        }
        self.data = {}
        """
        data = {
            "AMZN":
                {
                    "2018-01-01": OptionChain
                }
        }
        """

    def load_data(self, file_paths: list, has_header: bool):

        self.data = {}
        for path in file_paths:
            with open(path, "r") as csv_file:
                reader = csv.reader(csv_file)

                if has_header:
                    next(reader, None)

                for row in reader:
                    parsed = self.__parse_row(row)

                    if parsed["ticker"] not in self.data:
                        self.data[parsed["ticker"]] = {}

                    if parsed["quote_date"] not in self.data[parsed["ticker"]]:
                        self.data[parsed["ticker"]][parsed["quote_date"]] = OptionChain(
                            parsed["ticker"], parsed["quote_date"]
                        )

                    option = Option(
                        parsed["ticker"],
                        parsed["quote_date"],
                        parsed["expiration"],
                        parsed["strike"],
                        parsed["opt_type"],
                        parsed["price"],
                        parsed["bid"],
                        parsed["ask"],
                        parsed["underlying_px"],
                        parsed["iv"],
                        parsed["greeks"],
                    )

                    self.data[parsed["ticker"]][parsed["quote_date"]].add_option(option)

    def __parse_row(self, row):
        parsed = {}

        parsed["ticker"] = row[self.column_idx["ticker"]]
        parsed["quote_date"] = np.datetime64(row[self.column_idx["quote_date"]])
        parsed["expiration"] = np.datetime64(row[self.column_idx["expiration"]])
        parsed["strike"] = float(row[self.column_idx["strike"]])
        parsed["bid"] = float(row[self.column_idx["bid"]])
        parsed["ask"] = float(row[self.column_idx["ask"]])
        parsed["underlying_px"] = float(row[self.column_idx["underlying_px"]])
        parsed["iv"] = float(row[self.column_idx["iv"]])

        parsed["price"] = {
            "open": float(row[self.column_idx["open"]]),
            "high": float(row[self.column_idx["high"]]),
            "low": float(row[self.column_idx["low"]]),
            "close": float(row[self.column_idx["close"]]),
        }

        parsed["greeks"] = {
            "delta": float(row[self.column_idx["delta"]]),
            "gamma": float(row[self.column_idx["gamma"]]),
            "theta": float(row[self.column_idx["theta"]]),
            "vega": float(row[self.column_idx["vega"]]),
            "rho": float(row[self.column_idx["rho"]]),
        }

        if row[self.column_idx["opt_type"]] == "C":
            parsed["opt_type"] = "call"
        elif row[self.column_idx["opt_type"]] == "P":
            parsed["opt_type"] = "put"
        else:
            raise ParseError

        return parsed

    def get_chains_by_date(self, date: np.datetime64):
        out = {}
        for ticker in self.data:
            out[ticker] = self.data[ticker][date]

        return out


class ParseError(Exception):
    pass
