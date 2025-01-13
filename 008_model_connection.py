import os
import pandas as pd
from datetime import timedelta
from modules import indicators as indicators
from trading.utils.data import get_latest_open_time

symbol = 'ETHUSDT'
predicted_long_results = [1]
max_active_orders = 10
entry_confidence = 0.8
lookahead = 168
interval = 15

take_profit = 0.02
etp = 0.01
stop_loss = 0.2
# trailing_threshold = 1
# trailing_return_threshold = 0.4

def customized_specific_period_col(df):
    """
    Note: Calculate customized columns for the model
    """
    df = indicators.calculate_volatility(df, 20)
    df = indicators.calculate_VWAP(df, window=20)
    df = indicators.trend_min_max_price_indicator(df, look_back=192)
    df = indicators.calculate_candelstick_patterns(df)

    return df

def get_model_cols():
    """
    Note: Define the columns used in the input data of model
    :return: list of columns
    """
    cols = [
        'Open', 'High', 'Low', 'Close', 'Open_Volatility', 'High_Volatility', 'Low_Volatility', 'Close_Volatility',
        'fibonacci_0.382', 'fibonacci_0.5', 'fibonacci_0.618', 'fibonacci_1', 'fibonacci_0', 'Open_Close_pct', 
        'High_Low_pct', 'Up_Shadow_pct', 'Down_Shadow_pct', 'RSI', 'Lower Band', 'Cumulative_VP', 'Volume_log', 
        'Cumulative_Volume_log'
    ]
    return cols

def get_max_active_orders():
    """
    Note: Define the maximum number of active orders
    :return: max_active_orders as int
    """
    return max_active_orders

def check_entry_conditions(predicted_result, confidence, data=None, position_info={}):
    """
    :param predicted_result: result of prediction
    :param confidence: confidence of prediction
    :param data: market data, including prices, volumes and indicators
    :param position_info: position info such as entry price, active orders, take profit and stop loss
    :return: meet_entry_conditions as 'Long', 'Short' or ''
    """
    meet_entry_conditions = ''

    if predicted_result in predicted_long_results and confidence >= entry_confidence:
        meet_entry_conditions = 'Long'
    return meet_entry_conditions

def set_trade_configuration(position, current_price):
    """
    Note: Must set take_profit, stop_loss, tp_price and sl_price, otherwise the order cannot be sent correctly
    :param position: 'Long' or 'Short'
    :param current_price: current price of the symbol
    :return: trade_configuration as a dictionary
    """
    trade_configuration = {}
    if position == 'Long':
        trade_configuration['take_profit'] = take_profit
        trade_configuration['stop_loss'] = stop_loss
        trade_configuration['tp_price'] = current_price * (1 + take_profit)
        trade_configuration['sl_price'] = current_price * (1 - stop_loss)
        # trade_configuration['trailing_threshold'] = trailing_threshold
        # trade_configuration['trailing_return_threshold'] = trailing_return_threshold
    return trade_configuration

def check_exit_conditions(data, position, position_info={}, predicted_results=None, save_position_info=False):
    """
    Note: Determine exit conditions based on current market data and position info
    :param data: market data, including prices, volumes and indicators
    :param position: int, 1 for long position, 2 for short position
    :param position_info: DataFrame, including position info, such as entry price, take profit and stop loss
    :param predicted_results: Dataframe of model's predicted results
    :param save_position_info: bool, whether to save position info to csv
    :return: exit_condition as 'Stop Loss Long', 'Stop Loss Short', 'Take Profit Long', 'Take Profit Short' for record purpose
    :return: exit_price send current high or low price for tp or sl as their trigger price, use current close price for others
    """
    exit_condition = None
    exit_price = None

    # Read current market data and position info to determine exit conditions
    current_price = data.iloc[-1]['Close']
    current_high = data.iloc[-1]['High']
    current_low = data.iloc[-1]['Low']

    take_profit = position_info.iloc[-1]['take_profit']
    stop_loss = position_info.iloc[-1]['stop_loss']
    tp_price = position_info.iloc[-1]['tp_price']
    sl_price = position_info.iloc[-1]['sl_price']
    entry_price = position_info.iloc[-1]['entry_price']
    hold_count = position_info.iloc[-1]['hold_count']

    # Set exit long conditions
    if position == 1:
        if current_low <= sl_price:
            exit_condition = 'Stop Loss Long'
            exit_price = current_low
        elif current_high >= tp_price:
            exit_condition = 'Take Profit Long'
            exit_price = current_high
        elif hold_count >= lookahead:
            last_bar_open_time, _ = get_latest_open_time(interval)
            lookahead_start = last_bar_open_time - timedelta(minutes=lookahead * interval)
            predicted_results['datetime'] = pd.to_datetime(predicted_results['datetime'])
            predicted_results = predicted_results[predicted_results['datetime'] >= lookahead_start]
            if 1 not in predicted_results['predicted_result'].values:
                new_tp_price = entry_price * (1 + (take_profit - etp)) if take_profit - etp > 0 else tp_price
                new_sl_price = sl_price * 1.05
                print(f'Lookahead period reached and no signal found. New TP Price: {new_tp_price}, New SL Price: {new_sl_price}')

                if save_position_info == True:
                    position_info.loc[0, 'take_profit'] = take_profit - etp
                    position_info.loc[0, 'tp_price'] = new_tp_price
                    position_info.loc[0, 'stop_loss'] = (entry_price - new_sl_price) / entry_price
                    position_info.loc[0, 'sl_price'] = new_sl_price
                    position_info.to_csv(os.path.join(os.path.abspath(os.path.dirname('__file__')), 'records/current_position.csv'), index=False)

                if current_high >= new_tp_price:
                    exit_condition = 'Take Profit Long'
                    exit_price = current_high
                elif current_low <= new_sl_price:
                    exit_condition = 'Stop Loss Long'
                    exit_price = current_low

    return exit_condition, exit_price
