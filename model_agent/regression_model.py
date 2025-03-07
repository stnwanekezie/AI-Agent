# %%
import os
import logging
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime
import statsmodels.api as sm
from pathlib import Path
from sklearn.model_selection import train_test_split

root = Path(__file__).parent.resolve()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# %%


class StockReturnsModel:

    def __init__(self, **kwargs):
        self.raw_data = None
        self.add_constant = kwargs.pop("add_constant")
        self.stock_ticker = kwargs.pop("stock_ticker")
        self.train_test_split = kwargs.pop("train_test_split")
        self.cols_to_bump = self.parse_factor_adjustments(kwargs.pop("cols_to_bump"))
        self.dataset = self.load_and_transform_data(**kwargs)

        train_dep, train_indeps, test_dep, test_indeps = self.split_data()
        self.train_dep = train_dep
        self.test_dep = test_dep
        self.train_indeps = (
            train_indeps.assign(const=1) if self.add_constant else train_indeps
        )
        self.test_indeps = (
            test_indeps.assign(const=1) if self.add_constant else test_indeps
        )

        # self.train_indeps = (
        #     sm.add_constant(train_indeps) if add_constant else train_indeps
        # )
        # self.test_indeps = sm.add_constant(test_indeps) if add_constant else test_indeps

        self.model = self.estimate_model()

    def load_and_transform_data(self, **kwargs):

        # Alphavantage stock data
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={self.stock_ticker}&interval=5min&apikey={api_key}&outputsize=full"
        r = requests.get(url)
        stock_data = pd.DataFrame(r.json()["Time Series (Daily)"]).T
        stock_data = stock_data[["4. close"]].rename(
            columns={"4. close": self.stock_ticker}
        )
        stock_data.index = pd.to_datetime(stock_data.index, format=r"%Y-%m-%d")
        stock_data[self.stock_ticker] = stock_data[self.stock_ticker].astype(float)
        stock_data = stock_data.resample("ME").last().pct_change()
        stock_data.index = stock_data.index + pd.offsets.MonthBegin()

        # URL for Fama-French 3-factor data
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
        ff_data = pd.read_csv(url, skiprows=3, index_col=0)
        monthly_ff = ff_data[ff_data.index.map(lambda x: len(str(x).strip()) == 6)]
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
        df = pd.concat([stock_data, monthly_ff], axis=1).dropna()
        df[self.ticker] = df[self.ticker] - df["risk_free_rate"]
        self.raw_data = df.copy(deep=True)

        params_to_drop = [k for k, v in kwargs.items() if isinstance(v, bool) and not v]
        if params_to_drop:
            df = df.drop(params_to_drop, axis=1)
        params_to_flatten = {
            k: v for k, v in kwargs.items() if (isinstance(v, float) or v == "flat")
        }
        for k, v in params_to_flatten.items():
            if v == "flat":
                df.loc[:, k] = df.loc[:, k].mean()
            else:
                df.loc[:, k] = v

        for k in self.cols_to_bump:
            tmp = self.cols_to_bump[k]
            if tmp["type"] == "additive":
                df.loc[:, k] = df.loc[:, k] + tmp["factor"]
            else:
                df.loc[:, k] = df.loc[:, k] * (1 + tmp["factor"])

        return df

    def split_data(self):

        df = self.dataset

        if self.train_test_split:
            train_dep, test_dep, train_indeps, test_indeps = train_test_split(
                df[self.stock_ticker],
                df.drop(self.stock_ticker, axis=1),
                test_size=self.train_test_split,
            )
        else:
            train_dep, train_indeps, test_dep, test_indeps = (
                df[self.stock_ticker],
                df.drop(self.stock_ticker, axis=1),
                pd.DataFrame(),
                pd.DataFrame(),
            )

        return train_dep, train_indeps, test_dep, test_indeps

    def estimate_model(self):
        return sm.OLS(self.train_dep, self.train_indeps).fit()

    def predict(self, test_indeps=pd.DataFrame(), **kwargs):

        if not test_indeps.empty:
            test_indeps = (
                test_indeps.assign(const=1) if self.add_constant else test_indeps
            )
            pred = self.model.predict(test_indeps)
        elif kwargs.get("simulation_horizon"):
            start_date, end_date = kwargs.get("simulation_horizon").split("-")
            start_date, end_date = [
                datetime.strptime(dt.strip(), "%Y%m").date()
                for dt in [start_date, end_date]
            ]
            pred = self.model.predict(self.train_indeps.loc[start_date:end_date])
        elif self.train_test_split and not self.test_indeps.empty:
            pred = self.model.predict(self.test_indeps)
        else:
            logger.warning(
                "No test data provided. Full insample prediction will be returned."
            )
            pred = self.model.predict(self.train_indeps)

        return pred

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


if __name__ == "__main__":
    args = {
        "stock_ticker": "NVDA",
        "risk_free_rate": True,
        "excess_market_return": True,
        "size_factor": True,
        "value_factor": True,
        "cols_to_bump": ["size_factor-0.15-multiplicative"],
        "train_test_split": 0.0,
        "add_constant": True,
    }
    model = StockReturnsModel(**args).predict()
