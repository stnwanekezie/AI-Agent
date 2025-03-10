# %%
import os
import logging
import requests
import pandas as pd
import yfinance as yf
from io import StringIO
from pathlib import Path
import statsmodels.api as sm
from bs4 import BeautifulSoup
from datetime import datetime
from sklearn.model_selection import train_test_split

root = Path(__file__).parent.resolve()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# %%


class FamaFrenchModel:

    def __init__(self, **kwargs):
        self.raw_data = None
        self.add_constant = kwargs.pop("add_constant")
        self.stock_tickers = kwargs.pop("stock_tickers")
        self.train_test_split = kwargs.pop("train_test_split")
        self.cols_to_bump = self.parse_factor_adjustments(kwargs.pop("cols_to_bump"))
        self.monthly_ff = pd.DataFrame()
        self.end_train_indexes = {}
        self.stock_returns_list = []
        self.run_batch_process(**kwargs)

    @staticmethod
    def parse_factor_adjustments(cols_to_bump: list) -> dict:

        factor_dict = {}

        for col in cols_to_bump:
            if not isinstance(col, str) or col.count("-") != 2:
                logger.warning(f"Skipping invalid format: {col}")
                continue

            factor, value, adj_type = col.split("-")
            try:
                factor_dict[factor] = {"factor": float(value), "type": adj_type}
            except ValueError:
                logger.error(f"Could not convert {value} to float for {factor}")
                continue

        return factor_dict

    def load_and_transform_data(self, ticker, **kwargs):
        stock = yf.Ticker(ticker)
        stock_data = stock.history(period="max")
        close_prices = "Close"

        if stock_data.empty:
            # Alphavantage stock data
            api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={self.stock_ticker}&interval=5min&apikey={api_key}&outputsize=full"
            r = requests.get(url)
            stock_data = pd.DataFrame(r.json()["Time Series (Daily)"]).T
            close_prices = "4. close"
        if stock_data.empty:
            raise ValueError(f"No data available for ticker {ticker}")

        stock_return = (
            stock_data[[close_prices]]
            .rename(columns={close_prices: ticker})
            .pct_change()
            .dropna()
        )

        stock_return.index = pd.to_datetime(
            stock_return.index, format=r"%Y-%m-%d"
        ).tz_localize(None)
        stock_return[ticker] = stock_return[ticker].astype(float)
        stock_return = (1 + stock_return).resample("MS").prod() - 1
        stock_return.index = stock_return.index
        self.stock_returns_list.append(stock_return)

        if self.monthly_ff.empty:
            # URL for Fama-French 3-factor data
            url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
            ff_data = pd.read_csv(url, skiprows=3, index_col=0)
            # Find the end of the monthly data block (before annual data starts)
            end_idx = next(
                (i for i, x in enumerate(ff_data.index) if "Annual" in str(x)),
                len(ff_data),
            )  # alternatively, do: monthly_ff = ff_data[ff_data.index.map(lambda x: len(str(x).strip()) == 6)]
            monthly_ff = ff_data.iloc[:end_idx]

            monthly_ff.index = pd.to_datetime(monthly_ff.index, format="%Y%m")
            monthly_ff = monthly_ff.rename(
                columns={
                    "Mkt-RF": "excess_market_return",
                    "SMB": "size_factor",
                    "HML": "value_factor",
                    "RF": "risk_free_rate",
                }
            ).astype(float)
            monthly_ff = monthly_ff / 100
            self.monthly_ff = monthly_ff.copy(deep=True)
            monthly_ff["excess_market_return"] = (
                monthly_ff["excess_market_return"] + monthly_ff["risk_free_rate"]
            )

            params_to_drop = [
                k for k, v in kwargs.items() if isinstance(v, bool) and not v
            ]
            if params_to_drop:
                monthly_ff = monthly_ff.drop(params_to_drop, axis=1)
            params_to_flatten = {
                k: v for k, v in kwargs.items() if (isinstance(v, float) or v == "flat")
            }
            for k, v in params_to_flatten.items():
                if v == "flat":
                    monthly_ff.loc[:, k] = monthly_ff.loc[:, k].mean()
                else:
                    monthly_ff.loc[:, k] = v

            for k in self.cols_to_bump:
                tmp = self.cols_to_bump[k]
                if tmp["type"] == "additive":
                    monthly_ff.loc[:, k] = monthly_ff.loc[:, k] + tmp["factor"]
                else:
                    monthly_ff.loc[:, k] = monthly_ff.loc[:, k] * (1 + tmp["factor"])

            if "excess_market_return" in monthly_ff.columns:
                monthly_ff["excess_market_return"] = (
                    monthly_ff["excess_market_return"] - monthly_ff["risk_free_rate"]
                )
            if self.add_constant:
                monthly_ff = sm.add_constant(monthly_ff)

            self.monthly_ff_adj = monthly_ff

        df = pd.concat([stock_return, self.monthly_ff_adj], axis=1).dropna()
        df[ticker] = df[ticker] - df["risk_free_rate"]

        if self.train_test_split:
            train_dep, _, train_indeps, _ = train_test_split(
                df[ticker],
                df.drop([ticker, "risk_free_rate"], axis=1),
                test_size=self.train_test_split,
            )
        else:
            train_dep, train_indeps = df[ticker], df.drop(
                [ticker, "risk_free_rate"], axis=1
            )

        self.end_train_indexes[ticker] = train_dep.index[-1]

        return train_dep, train_indeps

    def predict(self, test_indeps=pd.DataFrame(), **kwargs):

        ticker = kwargs.get("ticker")
        predictions = []
        if ticker:
            model = next(
                (m for m in self.models if m.model.endog_names == ticker), None
            )
            if not model:
                raise ValueError(f"No model estimate for {ticker}")

        models = [model] if ticker else self.models
        monthly_ff_adj = self.monthly_ff_adj.drop("risk_free_rate", axis=1)
        for idx, model in enumerate(models):
            ticker = model.model.endog_names
            if not test_indeps.empty:
                test_indeps = (
                    test_indeps.assign(const=1) if self.add_constant else test_indeps
                )
                pred = model.predict(test_indeps)
            elif kwargs.get("simulation_horizon"):
                start_date, end_date = kwargs.get("simulation_horizon").split("-")
                start_date, end_date = [
                    datetime.strptime(dt.strip(), "%Y%m").date()
                    for dt in [start_date, end_date]
                ]
                pred = model.predict(monthly_ff_adj.loc[start_date:end_date])
            elif self.train_test_split:
                test_index = self.end_train_indexes[ticker]
                pred = model.predict(
                    monthly_ff_adj.loc[monthly_ff_adj.index > test_index]
                )
            else:
                if idx == 0:
                    logger.warning(
                        "No test data provided. Full insample prediction will be returned."
                    )
                df = pd.concat(
                    [self.stocks_data[ticker], monthly_ff_adj], axis=1
                ).dropna()
                pred = model.predict(df.drop(ticker, axis=1))

            pred.name = ticker
            predictions.append(pred)

        predictions = pd.concat(predictions, axis=1)
        return predictions

    def run_batch_process(self, **kwargs):
        models = []
        params_list = []
        for ticker in self.stock_tickers:
            train_dep, train_indeps = self.load_and_transform_data(ticker, **kwargs)
            model = sm.OLS(train_dep, train_indeps).fit()
            models.append(model)
            params = model.params.to_dict()
            params["rsq"] = model.rsquared
            params["rsq_adj"] = model.rsquared_adj
            params_df = (
                pd.DataFrame(params, index=[0]).transpose().rename(columns={0: ticker})
            )
            params_list.append(params_df)
        self.stocks_data = pd.concat(self.stock_returns_list, axis=1)
        del self.stock_returns_list
        self.models = models
        self.params_df = pd.concat(params_list, axis=1)


if __name__ == "__main__":
    args = {
        "stock_tickers": ["NVDA", "AAPL"],
        "risk_free_rate": True,
        "excess_market_return": True,
        "size_factor": True,
        "value_factor": True,
        "cols_to_bump": [],  # "size_factor-0.15-multiplicative"
        "train_test_split": 0.0,
        "add_constant": True,
    }
    model = FamaFrenchModel(**args).predict()
