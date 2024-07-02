# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta

class Strategy_trading(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = '5m'

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 0.114, 
        "816": 0.049, 
        "1756": 0.015, 
        "2656": 0 
    }
    stoploss = -0.111

    # Trailing stoploss
    trailing_stop = True
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 400

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }
    

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return [
            ("BTC/USDT:USDT", "1h"),
             ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        
        if not self.dp:
            # Don't do anything if DataProvider is not available.
            return dataframe

        #### TREND ####
        inf_tf = '1h'
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=inf_tf)    
        # Get the 200 hour EMA
        informative['ma200'] = ta.MA(informative, timeperiod=200)

        # Support and Resistance for 1h
        supports_1h, resistances_1h = self.supports_and_resistances(informative,50, field_for_support='low', field_for_resistance='high')
        informative['support'] = None
        informative.loc[supports_1h.index, "support"] = supports_1h.values
        informative['resistance'] = None
        informative.loc[resistances_1h.index, "resistance"] = resistances_1h.values
        informative['date'] = pd.to_datetime(informative['date'])

        # Xét từng nến để tìm các Resistance/Support gần nhất
        informative['nearest_support_index'] = -1
        informative['nearest_support'] = -1
        informative['nearest_resistance_index'] = -1
        informative['nearest_resistance'] = -1

        # Duyệt qua từng hàng trong DataFrame
        for i in range(len(informative)):

            # Lấy giá đóng cửa của nến hiện tại
            close_price = informative.loc[i, 'close']

            nearest_support_1h = supports_1h[supports_1h.index < i]
            nearest_resistance_1h = resistances_1h[resistances_1h.index < i]

            # Tìm mức hỗ trợ gần nhất (nhỏ hơn giá đóng cửa của nến)
            nearest_support_1h_index = nearest_support_1h[nearest_support_1h.values < close_price].index.max()
            if not pd.isna(nearest_support_1h_index):
                informative.loc[i, 'nearest_support_index'] = nearest_support_1h_index 
                informative.loc[i, 'nearest_support'] = nearest_support_1h[nearest_support_1h_index]

            # Tìm mức kháng cự gần nhất (lớn hơn giá đóng cửa của nến)
            nearest_resistance_1h_index = nearest_resistance_1h[nearest_resistance_1h.values > close_price].index.max()
            if not pd.isna(nearest_resistance_1h_index):
                informative.loc[i, 'nearest_resistance_index'] = nearest_resistance_1h_index 
                informative.loc[i, 'nearest_resistance'] = nearest_resistance_1h[nearest_resistance_1h_index]    
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, inf_tf, ffill=True)

        ### AREA OF VALUE & ENTRY TRIGER ###
        # Price channel    
        dataframe = self.trendlines(dataframe=dataframe, slicing_window=60, distance=20, chart=True, field_for_supports='low', field_for_resistances='high', timeframe='5m')
        
        # Shift values of maxslope, minslope, max_y_intercept, min_y_intercept of previous hour to prior hour
        dataframe['previous_maxslope_5m'] = dataframe['maxslope_5m'].shift(1)
        dataframe['previous_minslope_5m'] = dataframe['minslope_5m'].shift(1)
        dataframe['previous_max_y_intercept_5m'] = dataframe['max_y_intercept_5m'].shift(1)
        dataframe['previous_min_y_intercept_5m'] = dataframe['min_y_intercept_5m'].shift(1)

        # Peak and Bottom for 5m
        bottoms_5m, peaks_5m = self.supports_and_resistances(dataframe, 50, field_for_support='low', field_for_resistance='high')
        dataframe['bottom_5m'] = None
        dataframe.loc[bottoms_5m.index, "bottom_5m"] = bottoms_5m.values
        dataframe['peak_5m'] = None
        dataframe.loc[peaks_5m.index, "peak_5m"] = peaks_5m.values
        dataframe['date'] = pd.to_datetime(dataframe['date'])

        # Xét từng nến để tìm các đỉnh/đáy gần nhất
        dataframe['nearest_peak_index_5m'] = -1
        dataframe['nearest_peak_5m'] = -1
        dataframe['nearest_bottom_index_5m'] = -1
        dataframe['nearest_bottom_5m'] = -1

        # Duyệt qua từng hàng trong DataFrame
        for i in range(len(dataframe)):

            # Lấy giá đóng cửa của nến hiện tại
            close_price = dataframe.loc[i, 'close']

            # Tìm chỉ số (indice) của mức đáy và mức đỉnh gần nhất
            nearest_bottom_5m = bottoms_5m[bottoms_5m.index < i]
            if len(nearest_bottom_5m) != 0:
                dataframe.loc[i, 'nearest_bottom_index_5m'] = nearest_bottom_5m.index.max()
                dataframe.loc[i, 'nearest_bottom_5m'] = nearest_bottom_5m[nearest_bottom_5m.index.max()]

            nearest_peak_5m = peaks_5m[peaks_5m.index < i]
            if len(nearest_peak_5m) != 0:
                dataframe.loc[i, 'nearest_peak_index_5m'] = nearest_peak_5m.index.max()
                dataframe.loc[i, 'nearest_peak_5m'] = nearest_peak_5m[nearest_peak_5m.index.max()]

        # Xét từng nến để tìm các đỉnh/đáy, vùng kháng cự/ hỗ trợ gần nhất
        dataframe['previous_nearest_support_1h'] = dataframe['nearest_support_1h']
        dataframe['previous_nearest_resistance_1h'] = dataframe['nearest_resistance_1h']
        dataframe['previous_nearest_peak_5m'] = dataframe['nearest_peak_5m'].shift(1)
        dataframe['previous_nearest_bottom_5m'] = dataframe['nearest_bottom_5m'].shift(1)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        dataframe.loc[
            (
                (
                    dataframe['close'] > dataframe['previous_maxslope_5m'] * dataframe['close'].index + dataframe['previous_max_y_intercept_5m']
                )
                &
                
                (   
                    dataframe['close_1h'] > dataframe['ma200_1h']
                ) 
            ),
            'enter_long',
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (
                    dataframe['previous_nearest_resistance_1h'] != -1
                ) &
                (
                    (
                        dataframe['close'] >= dataframe['previous_nearest_resistance_1h'] - 100
                    )  
                    &
                    (
                        dataframe['close'] <= dataframe['previous_nearest_resistance_1h'] + 100
                    ) 
                )
            ),
            'exit_long'] = 1
        return dataframe

    """
    Put other function for our own strategy in here
    """
    def trendlines(self, dataframe, slicing_window=100, distance=50, chart=True, field_for_supports='low', field_for_resistances='high', timeframe="1h"):
        """
        Return a Pandas dataframe with support and resistance lines.

        :param dataframe: incoming data matrix
        :param slicing_window: number of candles for slicing window
        :param distance: Number of candles between two maximum points and two minimum points
        :param chart: Boolean value saying whether to print chart on web
        :param field_for_supports: for which column would you like to generate the support lines
        :param field_for_resistances: for which column would you like to generate the resistance lines
        :param timeframe: tmieframe use to find trendline
        """

        # Step 1: Using rolling window to find 2 peaks and 2 bottoms in each 100 candles
        df_high = dataframe[field_for_resistances].copy()
        df_low = dataframe[field_for_supports].copy()

        dataframe['peak1_idx'] = df_high.rolling(window=slicing_window).apply(lambda x: x.idxmax())
        dataframe['bottom1_idx'] = df_low.rolling(window=slicing_window).apply(lambda x: x.idxmin())
        dataframe['peak2_idx'] = df_high.rolling(window=slicing_window).apply(self.find_second_peak, args=(distance, ))
        dataframe['bottom2_idx'] = df_low.rolling(window=slicing_window).apply(self.find_second_bottom, args=(distance, ))
        
        # Step 2: Find maxline through 2 peaks and minline through 2 bottoms in each 100 candles
        dataframe['maxslope_' + timeframe] = None
        dataframe.loc[slicing_window - 1:, 'maxslope_' + timeframe] = (np.array(dataframe[field_for_resistances].iloc[dataframe['peak1_idx'][slicing_window - 1:].astype(int)]) - 
                                                    np.array(dataframe[field_for_resistances].iloc[dataframe['peak2_idx'][slicing_window - 1:].astype(int)])) / (np.array(dataframe['peak1_idx'][slicing_window - 1:]) - 
                                                                                                                                            np.array(dataframe['peak2_idx'][slicing_window - 1:])) # Slope between max points
        
        dataframe['minslope_' + timeframe] = None
        dataframe.loc[slicing_window - 1:, 'minslope_' + timeframe] = (np.array(dataframe[field_for_supports].iloc[dataframe['bottom1_idx'][slicing_window - 1:].astype(int)]) - 
                                                    np.array(dataframe[field_for_supports].iloc[dataframe['bottom2_idx'][slicing_window - 1:].astype(int)])) / (np.array(dataframe['bottom1_idx'][slicing_window - 1:]) - 
                                                                                                                                            np.array(dataframe['bottom2_idx'][slicing_window - 1:])) # Slope between max points

        dataframe['max_y_intercept_' + timeframe] = None
        dataframe.loc[slicing_window - 1:, 'max_y_intercept_' + timeframe] = (np.array(dataframe[field_for_resistances].iloc[dataframe['peak1_idx'][slicing_window - 1:].astype(int)]) - 
                                                            np.array(dataframe['maxslope_' + timeframe][slicing_window - 1:]) * np.array(dataframe['peak1_idx'][slicing_window - 1:])) # y-intercept for max trendline
        dataframe['min_y_intercept_' + timeframe] = None
        dataframe.loc[slicing_window - 1:, 'min_y_intercept_' + timeframe] = (np.array(dataframe[field_for_supports].iloc[dataframe['bottom1_idx'][slicing_window - 1:].astype(int)]) - 
                                                            np.array(dataframe['minslope_' + timeframe][slicing_window - 1:]) * np.array(dataframe['bottom1_idx'][slicing_window - 1:])) # y-intercept for min trendline
            
        return dataframe


    def supports_and_resistances(self, dataframe, rollsize, field_for_support='low', field_for_resistance='high'): 
        diffs1 = abs(dataframe['high'].diff().abs().iloc[1:]) 

        diffs2 = abs(dataframe['low'].diff().abs().iloc[1:]) 

        mean_deviation_ressistance = diffs1.mean() 

        mean_deviation_support = diffs2.mean() 
        supports = dataframe[dataframe.low == dataframe[field_for_support].rolling(rollsize, center=True).min()].low
        resistances = dataframe[dataframe.high == dataframe[field_for_resistance].rolling(rollsize, center=True).max()].high
        supports = supports[abs(supports.diff()) > mean_deviation_support] 
        resistances = resistances[abs(resistances.diff()) > mean_deviation_ressistance] 
        return supports,resistances 
    
    def find_second_peak(self, window_data, distance):
        peak1_idx = window_data.idxmax()
        if peak1_idx + distance >= window_data.index[-1]:    
            peak2_idx = window_data.loc[: (peak1_idx - distance + 1)].idxmax()
        else:
            peak2_idx = window_data.loc[(peak1_idx + distance) :].idxmax()
        return peak2_idx


    def find_second_bottom(self, window_data, distance):
        bottom1_idx = window_data.idxmin()
        if bottom1_idx + distance >= window_data.index[-1]:    
            bottom2_idx = window_data.loc[: (bottom1_idx - distance + 1)].idxmin()
        else:
            bottom2_idx = window_data.loc[(bottom1_idx + distance) :].idxmin()
        return bottom2_idx