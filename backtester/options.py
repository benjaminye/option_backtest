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
        DTE: {self.dte}
        Strike: {self.strike}
        Moneyness: {self.moneyness}
        -----------------------------
        Bid: {self.bid} 
        Mid: {self.mid}
        Ask: {self.ask}
        Spread: {self.spread}
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

    @property
    def moneyness(self):
        if self.opt_type == "call":
            return self.underlying_px / self.strike
        else:
            return self.strike / self.underlying_px

    @property
    def dte(self):
        return (self.expiration - self.quote_date).astype(int)

    @property
    def spread(self):
        return self.ask - self.bid

    @property
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

    def filter_expiry(self, expiry: np.datetime64):
        dates = list(self.chain.keys())
        days_delta = [abs(date - expiry).astype(int) for date in dates]

        match = dates[days_delta.index(min(days_delta))]

        new_chain = OptionChain(self.ticker, self.quote_date)

        for option in self.chain[match]:
            new_chain.add_option(option)

        return new_chain

    def filter_dte(self, dte: int):
        target_expiry = self.quote_date + dte

        return self.filter_expiry(target_expiry)

    def filter_type(self, opt_type: Literal["call", "put"]):
        options_list = list(self.chain.values())
        options_list = [option for sublist in options_list for option in sublist]

        new_chain = OptionChain(self.ticker, self.quote_date)

        for option in options_list:
            if option.opt_type.lower() == opt_type.lower():
                new_chain.add_option(option)

        return new_chain

    def filter_moneyness(self, tgt_moneyness: float):
        new_chain = OptionChain(self.ticker, self.quote_date)
        results = []

        chain = copy.deepcopy(self.chain)
        for _, option_list in chain.items():
            moneyness_list = [option.moneyness for option in option_list]
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
        result = self.filter_type(opt_type).filter_moneyness(moneyness)

        if isinstance(expiry, int):
            result = result.filter_dte(expiry)
        else:
            result = result.filter_expiry(expiry)

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
            self.__parse_file(path, has_header)

    def __parse_file(self, file_path, has_header):
        with open(file_path, "r") as csv_file:
            reader = csv.reader(csv_file)

            if has_header:
                next(reader, None)

            for row in reader:
                parsed = self.__parse_row(row)

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

                self.__add_to_chain(option)

    def __add_to_chain(self, option):
        ticker = option.ticker
        quote_date = option.quote_date

        if ticker not in self.data:
            self.data[ticker] = {}
        if quote_date not in self.data[ticker]:
            self.data[ticker][quote_date] = OptionChain(ticker, quote_date)

        self.data[ticker][quote_date].add_option(option)

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

    def filter_quote_date(self, date: np.datetime64):
        new_loader = Loader()

        for ticker in self.data:
            new_loader.data[ticker] = self.data[ticker][date]

        return new_loader

    def find_option_by_desc(self, quote_date, ticker, expiration, opt_type, mny):
        return self.data[ticker][quote_date].get_option(expiration, opt_type, mny)

    def find_option_exact(self, quote_date, target_option):
        ticker = target_option.ticker
        chain = self.data[ticker][quote_date].chain[target_option.expiration]

        for option in chain:
            if [option.strike, option.opt_type] == [
                target_option.strike,
                target_option.opt_type,
            ]:
                return option

    def list_dates(self):
        ticker = list(self.data)[0]
        date_list = list(self.data[ticker])

        return date_list


class ParseError(Exception):
    pass
