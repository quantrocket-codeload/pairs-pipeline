# Copyright 2019 QuantRocket LLC - All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from moonshot import Moonshot
from moonshot.commission import PerShareCommission

class USStockCommission(PerShareCommission):
    IB_COMMISSION_PER_SHARE = 0.005

class PairsStrategy(Moonshot):
    """
    Pairs trading strategy that uses the Johansen test to re-calculate
    hedge ratios daily and uses Bollinger Bands to time entries and exits.
    Buys (sells) the spread when it crosses below (above) its lower (upper)
    Bollinger Band and exits when it crosses its moving average.

    To use the strategy, subclass this base class and define the appropriate
    DB and CONIDS.
    """

    CODE = "pairs"
    DB = None
    DB_FIELDS = ["Close", "Open"]
    CONIDS = []
    LOOKBACK_WINDOW = 20 # Calculate hedge ratios and Bollinger Bands using this lookback
    BBAND_STD = 1 # Set Bollinger Bands this many standard deviations away from mean
    COMMISSION_CLASS = USStockCommission

    def get_hedge_ratio(self, pair_prices):
        """
        Helper function that uses the Johansen test to calculate hedge ratio. This is applied
        to the pair prices on a rolling basis in prices_to_signals.
        """
        pair_prices = pair_prices.dropna()

        # Skip if we don't have at least 75% of the expected observations
        if len(pair_prices) < self.LOOKBACK_WINDOW * 0.75:
            return pd.Series(0, index=pair_prices.columns)

        # The second and third parameters indicate constant term, with a lag of 1.
        # See Chan, Algorithmic Trading, chapter 2.
        result = coint_johansen(pair_prices, 0, 1)

        # The first column of eigenvectors contains the best weights
        weights = list(result.evec[0])

        return pd.Series(weights, index=pair_prices.columns)

    def prices_to_signals(self, prices):
        """
        Generates a DataFrame of signals indicating whether to long or short the
        spread.
        """
        closes = prices.loc["Close"]

        # Calculate hedge ratios on a rolling basis. Unfortunately, pandas
        # rolling apply() won't work here, so we have to loop through each day
        all_hedge_ratios = []
        for idx in range(len(closes)):
            start_idx = idx - self.LOOKBACK_WINDOW
            some_closes = closes.iloc[start_idx:idx]
            hedge_ratio = self.get_hedge_ratio(some_closes)
            hedge_ratio = pd.Series(hedge_ratio).to_frame().T
            all_hedge_ratios.append(hedge_ratio)

        hedge_ratios = pd.concat(all_hedge_ratios)
        hedge_ratios.index = closes.index

        # Compute spread and Bollinger Bands (spreads and everything derived
        # from it is a Series, which we later broadcast back to a DataFrame)
        spreads = (closes * hedge_ratios).sum(axis=1)
        means = spreads.fillna(method="ffill").rolling(self.LOOKBACK_WINDOW).mean()
        stds = spreads.fillna(method="ffill").rolling(self.LOOKBACK_WINDOW).std()
        upper_bands = means + self.BBAND_STD * stds
        lower_bands = means - self.BBAND_STD * stds

        # Long (short) the spread when it crosses below (above) the lower (upper)
        # band, then exit when it crosses the mean
        long_entries = spreads < lower_bands
        long_exits = spreads >= means
        short_entries = spreads > upper_bands
        short_exits = spreads <= means

        # Combine entries and exits
        ones = pd.Series(1, index=spreads.index)
        zeros = pd.Series(0, index=spreads.index)
        minus_ones = pd.Series(-1, index=spreads.index)
        long_signals = ones.where(long_entries).fillna(zeros.where(long_exits)).fillna(method="ffill")
        short_signals = minus_ones.where(short_entries).fillna(zeros.where(short_exits)).fillna(method="ffill")
        signals = long_signals + short_signals

        # Broadcast Series of signals to DataFrame
        signals = closes.apply(lambda x: signals)

        # Save hedge_ratios for signals_to_target_weights
        self.hedge_ratios = hedge_ratios

        return signals

    def signals_to_target_weights(self, signals, prices):
        """
        Converts the DataFrame of integer signals, indicating whether to long
        or short the spread, into the corresponding weight of each instrument
        to hold.
        """
        # hedge_ratios represents ratios of shares, multiply by price to get ratios
        # of weights
        hedge_ratio_weights = self.hedge_ratios * prices.loc["Close"]

        # Multiply weight ratios by signals to get target weights, then reduce to
        # 1X total allocation
        weights = signals * hedge_ratio_weights
        total_weights= weights.abs().sum(axis=1)
        weights = weights.div(total_weights, axis=0)
        return weights

    def target_weights_to_positions(self, weights, prices):
        # we'll enter in the period after the signal
        positions = weights.shift()
        return positions

    def positions_to_gross_returns(self, positions, prices):
        # Enter and exit on the open
        opens = prices.loc["Open"]
        gross_returns = opens.pct_change() * positions.shift()
        return gross_returns

class GDX_GLD_Pair(PairsStrategy):

    CODE = "pairs-gdx-gld"
    DB = "usa-etf-1d-p"
    CONIDS = [
        229726316, # GDX
        51529211, # GLD
    ]
