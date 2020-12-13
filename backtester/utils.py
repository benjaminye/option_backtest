def find_option_by_desc(loader, quote_date, ticker, expiration, opt_type, mny):
    return loader.data[ticker][quote_date].get_option(expiration, opt_type, mny)


def find_option_exact(loader, quote_date, target_option):
    ticker = target_option.ticker
    chain = loader.data[ticker][quote_date].chain[target_option.expiration]

    for option in chain:
        if [option.strike, option.opt_type] == [
            target_option.strike,
            target_option.opt_type,
        ]:
            return option
