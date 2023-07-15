
import MetaTrader5 as mt5

from copy import copy, deepcopy
from time import time, sleep
import math
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import ta as ta_ta
from ta.volatility import AverageTrueRange
import pandas_ta as ta

from scipy.stats import rankdata

import ctypes

####################################################################################################

#-- print文字色変更
ENABLE_PROCESSED_OUTPUT = 0x0001
ENABLE_WRAP_AT_EOL_OUTPUT = 0x0002
ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
MODE = ENABLE_PROCESSED_OUTPUT + ENABLE_WRAP_AT_EOL_OUTPUT + ENABLE_VIRTUAL_TERMINAL_PROCESSING
 
kernel32 = ctypes.windll.kernel32
handle = kernel32.GetStdHandle(-11)
kernel32.SetConsoleMode(handle, MODE)

GREEN = '\033[32m'
BLUE = '\033[34m'
BOLD = '\033[1m'
BLUE_BOLD = '\033[34m' + '\033[1m'
END = '\033[0m'

####################################################################################################

#-- 現在の時刻を取得
def get_current_time():
  current_time = datetime.now()
  current_time = datetime.strptime("{0:%Y-%m-%d %H:%M:%S}".format(current_time), '%Y-%m-%d %H:%M:%S')
  return current_time


#-- 各市場の取引時間の設定
def get_trade_start_end_time(market):
  current_date_time = datetime.now()

  # 東京市場用
  if market == "tk":
    trade_start_time_hour = 8
    trade_start_time_minute = 30
    trade_end_time_hour = 16
    trade_end_time_minute = 0

  # ロンドン市場用
  elif market == "ld":
    trade_start_time_hour = 16
    trade_start_time_minute = 35
    trade_end_time_hour = 19
    trade_end_time_minute = 0

  # ニューヨーク市場用
  elif market == "ny":
    trade_start_time_hour = 22
    trade_start_time_minute = 00
    trade_end_time_hour = 0
    trade_end_time_minute = 0
    

  trade_start_time = datetime(current_date_time.year, current_date_time.month, current_date_time.day, trade_start_time_hour, trade_start_time_minute, 0)
  trade_end_time = datetime(current_date_time.year, current_date_time.month, current_date_time.day, trade_end_time_hour, trade_end_time_minute, 0)
  if market == "ny":
    trade_end_time = trade_end_time + timedelta(days=1)

  return trade_start_time, trade_end_time

####################################################################################################

#-- 各取引関連情報、トレンドフラグ予測lightgbm使用時の精度表示（外側ループ）
def show_info_outer(cnt, account_balance, account_margin_level, symbol, spread, flag_calc_data, select_features_list=None, score=None):
    print(f"・Balance                      : {account_balance}")
    print(f"・margin_level                 : {account_margin_level} %")
    print(f'・銘柄: {symbol}')
    print(f'・spread                       : {spread}')
    
    if type(select_features_list) != type(None):
        print()
        print(f'・len(select_features_list): {len(select_features_list)}')    
        print()

    if type(score) != type(None):    
        print(f'・accuracy : {score["accuracy"]}')
        print(f'・recall   : {score["recall"]}')
        print(f'・precision: {score["precision"]}')
        print(f'・f1_score : {score["f1"]}')


#-- 相場の状態、売買フラグ等を表示（内側ループ）
def show_info_inner(
    flag_calc_data, symbol, current_time, last_trade_time, last_graph_render_time_dic,
    bollinger_spread_range, low_under_line, high_under_line, atr, atr_under_line,
    trand_judgment_material_flag, buy_flag_type, sell_flag_type, account_margin_level, period_adj
):
    print()
    print(f'・bollinger_spread_range       : {bollinger_spread_range} / {low_under_line} / {high_under_line}')
    print(f'・atr                          : {atr} / {atr_under_line}')        
    print(f'・trend_flag                   : {flag_calc_data["time"].iloc[period_adj]} / {np.round(flag_calc_data["trend_clac_change_7_ema"].iloc[period_adj], decimals=5)} / {flag_calc_data["trend_flag"].iloc[period_adj]}')        
    if trand_judgment_material_flag != False:                    
        print(f'・trand_judgment_material_flag : {trand_judgment_material_flag}')
        print(f'・buy sell flg_type            : {(buy_flag_type, sell_flag_type)}')
    if trand_judgment_material_flag == False:                    
        print(f'・trand_judgment_material_flag : {trand_judgment_material_flag} のためパス')

####################################################################################################

#-- 過去分OHLCデータを取得
def get_mt5_data(symbol, time_frame, data_back_pos, bar_period):
    rates = mt5.copy_rates_from_pos(symbol, time_frame, data_back_pos, bar_period)
    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')  
    return data


#-- 全ポジション情報を取得
def get_all_positions(symbol_list):
  position_list = []
  for symbol in symbol_list:
    positions = mt5.positions_get(group='*' + symbol + '*')
    for position in positions:
      position_list.append(position)
  return position_list

  
#-- ポジション情報を取得（おそらく未使用）
def get_position(symbol):
    buy_position = None
    sell_position = None
    positions = mt5.positions_get(group='*' + symbol + '*')

    for position in positions:
        order_type = position[5]

        if order_type == 0:
            buy_position = position
        elif order_type == 1:
            sell_position = position
    return buy_position, sell_position


#-- 注文実行後、設定時刻経過後に約定しなかったものをキャンセル
def order_cancel(symbol, order_cancel_period=5):
    get_mt5_now_time = get_mt5_data(symbol, mt5.TIMEFRAME_M1, 0, 1)
    expiration_time = get_mt5_now_time["time"].max() + timedelta(minutes=order_cancel_period)
    expiration_time = int(expiration_time.timestamp())                
    return expiration_time

####################################################################################################

#-- 売買注文送信
def order_send(
    symbol, buy_flg, sell_flg, first_lot, buy_price_stop, buy_price_limit, sell_price_stop, sell_price_limit,
    sl, tp, slippage, magic_number, expiration_time, trand_judgment_material_flag
):

    result = None

    if buy_flg:            
        if trand_judgment_material_flag == "trand_market":                    
            request = {
                'symbol': symbol,
                'action': mt5.TRADE_ACTION_PENDING,
                'type': mt5.ORDER_TYPE_BUY_STOP,
                'volume': first_lot,
                'price': buy_price_stop,
                'sl': sl,
                'tp': tp,
                'deviation': slippage,
                'magic': magic_number,
                'type_time': mt5.ORDER_TIME_SPECIFIED,
                'expiration': expiration_time,
                'type_filling': mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)    

        else:
            request = {
                'symbol': symbol,
                'action': mt5.TRADE_ACTION_PENDING,
                'type': mt5.ORDER_TYPE_BUY_LIMIT,
                'volume': first_lot,
                'price': buy_price_limit,
                'sl': sl,
                'tp': tp,
                'deviation': slippage,
                'magic': magic_number,
                'type_time': mt5.ORDER_TIME_SPECIFIED,
                'expiration': expiration_time,
                'type_filling': mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)    

    if sell_flg:            
        if trand_judgment_material_flag == "trand_market":
            request = {
                'symbol': symbol,
                'action': mt5.TRADE_ACTION_PENDING,
                'type': mt5.ORDER_TYPE_SELL_STOP,
                'volume': first_lot,
                'price': sell_price_stop,
                'sl': sl,
                'tp': tp,
                'deviation': slippage,
                'magic': magic_number,
                'type_time': mt5.ORDER_TIME_SPECIFIED,
                'expiration': expiration_time,
                'type_filling': mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request) 

        else:
            request = {
                'symbol': symbol,
                'action': mt5.TRADE_ACTION_PENDING,
                'type': mt5.ORDER_TYPE_SELL_LIMIT,
                'volume': first_lot,
                'price': sell_price_limit,
                'sl': sl,
                'tp': tp,
                'deviation': slippage,
                'magic': magic_number,
                'type_time': mt5.ORDER_TIME_SPECIFIED,
                'expiration': expiration_time,
                'type_filling': mt5.ORDER_FILLING_IOC,
            }                    

            result = mt5.order_send(request)

    return result

####################################################################################################

#-- テクニカル分析関連。微調整する際に使用する関数
def val_set_def():
  return {
    "ema_preriod_fast_main": 3,
    "ema_preriod_slow_main": 20,

    "ema_window_1_main": 5,
    "ema_window_2_main": 6,
    "macd_line_window_main": 9,

    "sto_k_period_main": 30,
    "sto_d_period_main": 7,
    "sto_slowing_main": 10,

    "adx_period_main": 50,

    "trend_change": 7,
  }  


#-- 各種インジケーターのデータを作成するためのメインプロセス関数
def add_technical_data(data, val_set_dic, puttern_flag=0):
  ema_preriod_fast_main = val_set_dic["ema_preriod_fast_main"]
  ema_preriod_slow_main = val_set_dic["ema_preriod_slow_main"]
  ema_window_1_main = val_set_dic["ema_window_1_main"]
  ema_window_2_main = val_set_dic["ema_window_2_main"]
  macd_line_window_main = val_set_dic["macd_line_window_main"]
  sto_k_period_main = val_set_dic["sto_k_period_main"]
  sto_d_period_main = val_set_dic["sto_d_period_main"]
  sto_slowing_main = val_set_dic["sto_slowing_main"]
  adx_period_main = val_set_dic["adx_period_main"]

  if puttern_flag == 0:
    data = calculate_atr(data)
    data = calculate_sma(data)
    data = calculate_rsi(data)
    data = calculate_bollinger_band(data)
    calculate_hampel_bollinger(data)
    data = calculate_macd(data)
    data = calculate_ichimoku_kinnkouhyou(data)
  elif puttern_flag == 1:
    data = calculate_atr(data, period=14)
    data = calculate_ema(data, window=5)
    data = calculate_ema(data, window=10)        
    data = calculate_sma(data, period=9)
    data = calculate_sma(data, period=21)    
    data = calculate_rsi(data, period=9)
    data = calculate_bollinger_band(data, window=20)
    data = calculate_bollinger_band(data, window=50)    
    data = calculate_hampel_bollinger(data, window=20)
    data = calculate_macd(data)
    data = calculate_rci(data, interval=5)
    data = calculate_rci(data, interval=8)
    data = calculate_stochastics(data, k_period=5, d_period=3, slowing=3)    
    data = calculate_stochastics(data, k_period=30, d_period=10, slowing=10)
    data = calculate_adx(data, period=6)
    data = calculate_w_ma_cross(data, ema_preriod_fast=3, ema_preriod_slow=10, sma_preriod=200)
    data = calculate_w_ma_cross(data, ema_preriod_fast=1, ema_preriod_slow=10)
    data = calculate_rci(data, interval=15)
    data = calculate_rci(data, interval=30)
    data = calculate_w_ma_cross(data, ema_preriod_fast=1, ema_preriod_slow=12, sma_preriod=20)
    data = calculate_w_ma_cross(data, ema_preriod_fast=9, ema_preriod_slow=20, sma_preriod=20)
    data = calculate_signal(data, ema_window_1=5, ema_window_2=6, macd_line_window=9)  
    data = calculate_stochastics(data, k_period=3, d_period=3, slowing=12)
    data = calculate_adx_2(data, period=50)
    data = calculate_w_ma_cross(data, ema_preriod_fast=3, ema_preriod_slow=150, sma_preriod=20)

 # ------------------------------------------------------------------------------------------------
    data = calculate_w_ma_cross(data, ema_preriod_fast=ema_preriod_fast_main, ema_preriod_slow=ema_preriod_slow_main, sma_preriod=20)
    data = calculate_stochastics(data, k_period=sto_k_period_main, d_period=sto_d_period_main, slowing=sto_slowing_main)

  return data

####################################################################################################
#-- 各種インジケーター個別作成用

def calculate_atr(data, period=14):
    data[f'tr_{period}'] = np.maximum((data['high'] - data['low']), 
                            np.maximum(abs(data['high'] - data['close'].shift()), abs(data['low'] - data['close'].shift())))
    data[f'atr_{period}'] = data[f'tr_{period}'].rolling(window=period).mean()
    return data


def calculate_sma(data, period=15, price_type='close'):
    data[f'sma_{period}'] = data[price_type].rolling(window=period).mean()
    return data


def calculate_rsi(data, period=14, price_type='close'):
    delta = data[price_type].diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    avg_gain = gains.rolling(window=period).mean()
    avg_loss = losses.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    data[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    return data


def calculate_bollinger_band(data, window=20, price_type='close'): # ±1σ: 68.26%, ±2σ: 95.44%, ±3σ: 99.74%
  target_data = data[price_type]
  b_band_use_sma = target_data.rolling(window=window).mean()

  sigma_1 = target_data.rolling(window=window).std()
  sigma_2 = target_data.rolling(window=window).std() * 2
  sigma_3 = target_data.rolling(window=window).std() * 3

  upper_sigma_1 = b_band_use_sma + sigma_1
  upper_sigma_2 = b_band_use_sma + sigma_2
  upper_sigma_3 = b_band_use_sma + sigma_3
  lower_sigma_1 = b_band_use_sma - sigma_1
  lower_sigma_2 = b_band_use_sma - sigma_2
  lower_sigma_3 = b_band_use_sma - sigma_3

  data[f"sma_{window}"] = b_band_use_sma
  data[f"upper_sigma_1_{window}"] = upper_sigma_1
  data[f"upper_sigma_2_{window}"] = upper_sigma_2
  data[f"upper_sigma_3_{window}"] = upper_sigma_3
  data[f"lower_sigma_1_{window}"] = lower_sigma_1
  data[f"lower_sigma_2_{window}"] = lower_sigma_2
  data[f"lower_sigma_3_{window}"] = lower_sigma_3

  return data


def calculate_hampel_bollinger(data, window=20, price_type='close'):
  target_data = data[price_type]
  data[f"hampel_Mid"] = target_data.rolling(window=window).median()
            
  #中央値±3×1.4826×中央絶対偏差
  data[f"medad"] = (target_data-target_data.rolling(window=window).median()).abs().rolling(window=window).median()

  data[f"hampel_upper_1"] = data[f"hampel_Mid"] + 1.4826 * data[f"medad"] * 1           
  data[f"hampel_upper_2"] = data[f"hampel_Mid"] + 1.4826 * data[f"medad"] * 2                      
  data[f"hampel_upper_3"] = data[f"hampel_Mid"] + 1.4826 * data[f"medad"] * 3
  data[f"hampel_lower_1"] = data[f"hampel_Mid"] - 1.4826 * data[f"medad"] * 1           
  data[f"hampel_lower_2"] = data[f"hampel_Mid"] - 1.4826 * data[f"medad"] * 2           
  data[f"hampel_lower_3"] = data[f"hampel_Mid"] - 1.4826 * data[f"medad"] * 3                       

  return data


def calculate_ema(data, window=5, price_type='close'):
  data[f"ema_{window}"] = data[price_type].ewm(span=window, adjust=False).mean()
  return data


def calculate_macd(data, price_type='close'):  
  ema_12 = calculate_ema(data, window=12)
  ema_26 = calculate_ema(data, window=26)

  macd_line = pd.DataFrame(ema_12["ema_12"] - ema_26["ema_26"], columns=["close"])
  signal = calculate_ema(macd_line, window=9)[["ema_9"]]
  signal.columns = ["signal"]
  macd_line = macd_line[["close"]]
  macd_line.columns = ["macd"]

  osci = pd.DataFrame(macd_line["macd"] - signal["signal"], columns=["osci"])

  data["ema_12"] = ema_12[["ema_12"]]
  data["ema_26"] = ema_26[["ema_26"]]
  data["osci"] = osci  
  data["macd_line"] = macd_line
  data["signal"] = signal

  return data


def calculate_signal(data, ema_window_1=12, ema_window_2=26, macd_line_window=9, price_type='close'):  
  ema_1 = calculate_ema(data, window=ema_window_1)
  ema_2 = calculate_ema(data, window=ema_window_2)

  macd_line = pd.DataFrame(ema_1[f"ema_{ema_window_1}"] - ema_2[f"ema_{ema_window_2}"], columns=["close"])
  signal = calculate_ema(macd_line, window=macd_line_window)[[f"ema_{macd_line_window}"]]
  signal.columns = [f"signal_{macd_line_window}"]
  macd_line = macd_line[["close"]]
  macd_line.columns = [f"macd_{ema_window_1}_{ema_window_2}"]

  data[f"ema_{ema_window_1}"] = ema_1[[f"ema_{ema_window_1}"]]
  data[f"ema_{ema_window_2}"] = ema_2[[f"ema_{ema_window_2}"]]
  data[f"macd_line_{ema_window_1}_{ema_window_2}"] = macd_line
  data[f"signal_{macd_line_window}"] = signal

  return data


def calculate_ichimoku_kinnkouhyou(data, mode=None):
  process_data = deepcopy(data)

  # 基準線
  high26 = process_data["high"].rolling(window=26).max()
  low26 = process_data["low"].rolling(window=26).min()
  process_data["base_line"] = (high26 + low26) / 2
  # 転換線
  high9 = process_data["high"].rolling(window=9).max()
  low9 = process_data["low"].rolling(window=9).min()
  process_data["conversion_line"] = (high9 + low9) / 2
  # 先行スパン1
  leading_span1 = (process_data["base_line"] + process_data["conversion_line"]) / 2
  process_data["leading_span1"] = leading_span1.shift(25)
  # 先行スパン2
  high52 = process_data["high"].rolling(window=52).max()
  low52 = process_data["low"].rolling(window=52).min()
  leading_span2 = (high52 + low52) / 2
  process_data["leading_span2"] = leading_span2.shift(25)

  if mode == "full":
  # 遅行スパン
    process_data["lagging_span"] = process_data["close"].shift(-25)

  return process_data


def calculate_rci(data, interval=9, price_type='close'):
  close_list = data[price_type]
  rci_list = [None for _ in range(interval)]

  nb_close = len(close_list)
  for idx in range(nb_close):
    if (idx + interval > nb_close):
      break

    y = close_list[idx:idx + interval]
    x_rank = np.arange(len(y))
    y_rank = rankdata(y, method='ordinal') - 1
    sum_diff = sum((x_rank - y_rank)**2)
    rci = (1 - ((6 * sum_diff) / (interval**3 - interval))) * 100
    rci_list.append(rci)

  rci_list = pd.DataFrame(rci_list)
  rci_list.columns=["rci"]

  data[f'rci_{interval}'] = rci_list
  return data


def calculate_stochastics(data, k_period=5, d_period=3, slowing=3):
  close = data["close"]
  high = data["high"]
  low  = data["low"]
  data[f"%K_{k_period}"] = ((close - low.rolling(k_period).min()) / (high.rolling(k_period).max() - low.rolling(k_period).min())) * 100
  data[f"%D_{d_period}"] = data[f"%K_{k_period}"].rolling(d_period).mean()
  data[f"Slow%D_{slowing}"] = data[f"%D_{d_period}"].rolling(slowing).mean()

  return data


def calculate_adx(data, period=14):
  adx = ta.trend.adx(data['high'], data['low'], data['close'], length=period)
  adx.columns = ["adx", "plus_di", "minus_di"]
  data['adx'] = adx["adx"]
  data['plus_di'] = adx["plus_di"]
  data['minus_di'] = adx["minus_di"]
  return data


def calculate_adx_2(data, period=14):
  adx = ta.trend.adx(data['high'], data['low'], data['close'], length=period)
  adx.columns = ["adx", "plus_di", "minus_di"]
  data[f'adx_{period}'] = adx["adx"] * 5
  data[f'plus_di_{period}'] = adx["plus_di"] * 5
  data[f'minus_di_{period}'] = adx["minus_di"] * 5
  return data


def calculate_w_ma_cross(data, ema_preriod_fast=5, ema_preriod_slow=10, sma_preriod=200):
  data[f"ema_close_{ema_preriod_fast}"] = data["close"].ewm(span=ema_preriod_fast, adjust=False).mean()
  data[f"ema_high_{ema_preriod_fast}"] = data['high'].ewm(span=ema_preriod_fast, adjust=False).mean()
  data[f"ema_low_{ema_preriod_fast}"] = data['low'].ewm(span=ema_preriod_fast, adjust=False).mean() 
  data[f"ema_close_{ema_preriod_slow}"] = data["close"].ewm(span=ema_preriod_slow, adjust=False).mean()
  data[f"ema_high_{ema_preriod_slow}"] = data['high'].ewm(span=ema_preriod_slow, adjust=False).mean()
  data[f"ema_low_{ema_preriod_slow}"] = data['low'].ewm(span=ema_preriod_slow, adjust=False).mean() 
  data[f"sma_close_{sma_preriod}"] = data["close"].rolling(sma_preriod).mean() 
  data[f"sma_high_{sma_preriod}"] = data['high'].rolling(sma_preriod).mean() 
  data[f"sma_low_{sma_preriod}"] = data['low'].rolling(sma_preriod).mean() 
  return data

####################################################################################################

#-- グラフ表示用メインプロセス関数
def graph_render_main_process(data, val_set_dic, act_data=None, graph_time_range=None, graph_set_time_start=None, graph_set_time_end=None):
  graph_render(data, "chart", val_set_dic, act_data=act_data, graph_time_range=graph_time_range, graph_set_time_start=graph_set_time_start, graph_set_time_end=graph_set_time_end)                
  graph_render(data, "ema_cross", val_set_dic, graph_time_range=graph_time_range, graph_set_time_start=graph_set_time_start, graph_set_time_end=graph_set_time_end)                    
  graph_render(data, "rsi", val_set_dic, graph_time_range=graph_time_range, graph_set_time_start=graph_set_time_start, graph_set_time_end=graph_set_time_end)
#  graph_render(flag_calc_data, "signal", val_set_dic, graph_time_range=graph_time_range, graph_set_time_start=graph_set_time_start, graph_set_time_end=graph_set_time_end)
  graph_render(data, "macd", val_set_dic, graph_time_range=graph_time_range, graph_set_time_start=graph_set_time_start, graph_set_time_end=graph_set_time_end)
  graph_render(data, "rci", val_set_dic, graph_time_range=graph_time_range, graph_set_time_start=graph_set_time_start, graph_set_time_end=graph_set_time_end)
  graph_render(data, "adx", val_set_dic, graph_time_range=graph_time_range, graph_set_time_start=graph_set_time_start, graph_set_time_end=graph_set_time_end)            
  graph_render(data, "stochastics", val_set_dic, graph_time_range=graph_time_range, graph_set_time_start=graph_set_time_start, graph_set_time_end=graph_set_time_end)    
  graph_render(data, "trend_change", val_set_dic, graph_time_range=graph_time_range, graph_set_time_start=graph_set_time_start, graph_set_time_end=graph_set_time_end)    
  graph_render(data, "ichimoku", val_set_dic, act_data=act_data, graph_time_range=graph_time_range, graph_set_time_start=graph_set_time_start, graph_set_time_end=graph_set_time_end)         

####################################################################################################

#-- 個別グラフ表示用関数。引数により表示内容を変更（kinds引数、”chart”等）
def graph_render(data, kinds, val_set_dic, act_data=None, graph_time_range=None, graph_set_time_start=None, graph_set_time_end=None):
    x_axis_interval = 2

    ema_preriod_fast_main = val_set_dic["ema_preriod_fast_main"]
    ema_preriod_slow_main = val_set_dic["ema_preriod_slow_main"]
    ema_window_1_main = val_set_dic["ema_window_1_main"]
    ema_window_2_main = val_set_dic["ema_window_2_main"]
    macd_line_window_main = val_set_dic["macd_line_window_main"]
    sto_k_period_main = val_set_dic["sto_k_period_main"]
    sto_d_period_main = val_set_dic["sto_d_period_main"]
    sto_slowing_main = val_set_dic["sto_slowing_main"]
    adx_period_main = val_set_dic["adx_period_main"]

    trend_change = val_set_dic["trend_change"]

    graph_use_df = deepcopy(data)

    if type(graph_time_range) != type(None):
      graph_use_df = graph_use_df[-int(len(graph_use_df)*graph_time_range):]
      graph_use_df = graph_use_df.reset_index(drop=True)

    if type(graph_set_time_start) != type(None):
      graph_use_df = graph_use_df[graph_set_time_start<=graph_use_df["time"]]
      graph_use_df = graph_use_df.reset_index(drop=True) 

    if type(graph_set_time_end) != type(None):
      graph_use_df = graph_use_df[graph_set_time_end>=graph_use_df["time"]]
      graph_use_df = graph_use_df.reset_index(drop=True)       

    if type(act_data) != type(None):
      act_data = deepcopy(act_data)

      if type(graph_time_range) != type(None):
        act_data = act_data[-int(len(act_data)*graph_time_range):]
        act_data = act_data.reset_index(drop=True)    

      if type(graph_set_time_start) != type(None):
        act_data = act_data[graph_set_time_start<=act_data["time"]]        
        act_data = act_data.reset_index(drop=True)    

      if type(graph_set_time_end) != type(None):      
        act_data = act_data[graph_set_time_end>=act_data["time"]]        
        act_data = act_data.reset_index(drop=True)              

      act_df = deepcopy(act_data)[["time", "close"]]
      act_df = act_df.rename({"close": "y"}, axis=1)
      graph_use_df = graph_use_df.rename({"close": "yhat"}, axis=1)


    if kinds == "chart":
      fig = plt.figure(figsize=(24, 8))      
      ax1 = fig.subplots()

      idx1 = act_data.index[act_data["close"] >= act_data["open"]]
      idx0 = act_data.index[act_data["close"] < act_data["open"]]
      act_data["body"] = act_data["close"] - act_data["open"]
      act_data["body"] = act_data["body"].abs()
      ax1.bar(idx1, act_data.loc[idx1, "body"], width = 1, bottom = act_data.loc[idx1, "open"], linewidth = 1, color = "#33b066", zorder = 2, alpha=0.8)
      ax1.bar(idx0, act_data.loc[idx0, "body"], width = 1, bottom = act_data.loc[idx0, "close"], linewidth = 1, color = "#ff5050", zorder = 2, alpha=0.8)
      ax1.vlines(act_data.index, act_data["low"], act_data["high"], linewidth = 1, color="#666666", zorder=1)

      join_data = pd.merge(graph_use_df[[
        "time", "yhat", "sma_20", "ema_close_3", "ema_high_3", "ema_low_3", "ema_close_10", "ema_high_10", "ema_low_10", "sma_close_200", "sma_high_200", "sma_low_200",
        "upper_sigma_2_20", "lower_sigma_2_20", "upper_sigma_2_50", "lower_sigma_2_50",
        "ema_close_1"
      ]], act_df[["time", "y"]], how="left", on="time") 
      join_data = join_data.set_index("time")
      join_data = join_data.reindex(columns=[
        'y', "yhat", "sma_20", "ema_close_3", "ema_high_3", "ema_low_3", "ema_close_10", "ema_high_10", "ema_low_10", "sma_close_200", "sma_high_200", "sma_low_200",
        "upper_sigma_2_20", "lower_sigma_2_20", "upper_sigma_2_50", "lower_sigma_2_50",
        "ema_close_1"
      ])      

      ax1.plot(range(len(join_data.index)), join_data.y, color="black", alpha=1, label="y")
      ax1.plot(range(len(join_data.index)), join_data.yhat, color="red", alpha=1, label="yhat")
      ax1.plot(range(len(join_data.index)), join_data.sma_20, ls="-", color="green", alpha=0.3, label="sma_20")

      ax1.plot(range(len(join_data.index)), join_data.ema_close_10, ls="--", color="blue", alpha=1.0, label="ema_close_10")      
      ax1.plot(range(len(join_data.index)), join_data.ema_high_10, ls="--", color="blue", alpha=0.5, label="ema_high_10")            
      ax1.plot(range(len(join_data.index)), join_data.ema_low_10, ls="-.", color="blue", alpha=0.5, label="ema_low_10")      

      ax1.plot(range(len(join_data.index)), join_data.sma_close_200, ls="--", color="green", alpha=0.5, label="sma_close_200")
      ax1.plot(range(len(join_data.index)), join_data.sma_high_200, ls="--", color="green", alpha=0.5, label="sma_high_200")            
      ax1.plot(range(len(join_data.index)), join_data.sma_low_200, ls="-.", color="green", alpha=0.5, label="sma_low_200")      
      ax1.plot(range(len(join_data.index)), join_data.upper_sigma_2_20, ls="--", color="orange", alpha=0.5, label="upper_sigma_2_20")
      ax1.plot(range(len(join_data.index)), join_data.lower_sigma_2_20, ls="--", color="orange", alpha=0.5, label="lower_sigma_2_20")           
      ax1.plot(range(len(join_data.index)), join_data.upper_sigma_2_50, ls="--", color="red", alpha=0.5, label="upper_sigma_2_50")
      ax1.plot(range(len(join_data.index)), join_data.lower_sigma_2_50, ls="--", color="red", alpha=0.5, label="lower_sigma_2_50")

      plt.title('chart')
      plt.xlim(0, len(join_data))
      ax1.grid(color='black', alpha=0.2)
      ax1.grid(color='gray', alpha=0.2)
      plt.xticks(np.arange(0, len(join_data), x_axis_interval))
      plt.legend(loc='upper left')      

    elif kinds == "rsi":
      upper_line = 80
      lower_line = 20
      graph_use_df_ma_rsi =  deepcopy(graph_use_df)[["time", "rsi_9"]]
      graph_use_df_ma_rsi[f"line_{upper_line}"] = upper_line
      graph_use_df_ma_rsi[f"line_{lower_line}"] = lower_line        
      graph_use_df_ma_rsi = graph_use_df_ma_rsi.set_index("time")

      fig = plt.figure(figsize=(20, 5))
      ax1 = fig.subplots()
      ax1.plot(range(len(graph_use_df_ma_rsi)), graph_use_df_ma_rsi.rsi_9, color="blue", alpha=0.8, label="rsi")
      ax1.plot(range(len(graph_use_df_ma_rsi)), graph_use_df_ma_rsi[f"line_{upper_line}"], ls="--", color="red", alpha=0.5, label=f"{upper_line}")
      ax1.plot(range(len(graph_use_df_ma_rsi)), graph_use_df_ma_rsi[f"line_{lower_line}"], ls="--", color="red", alpha=0.5, label=f"{lower_line}")                    
              
      plt.title('rsi')
      ax1.set_ylim([0, 100])
      plt.xlim(0, len(graph_use_df_ma_rsi))
      ax1.grid(color='black', alpha=0.2)
      ax1.grid(color='gray', alpha=0.2)
      plt.xticks(np.arange(0, len(graph_use_df_ma_rsi), x_axis_interval))
      plt.legend(loc='upper left')

    elif kinds == "macd":
      graph_use_df = graph_use_df.set_index("time")
      graph_use_df = graph_use_df[["close", "osci", "macd_line", "signal"]]

      fig = plt.figure(figsize=(20, 5))
      ax1 = fig.subplots()
      ax2 = ax1.twinx()
      plot = ax1.bar(range(len(graph_use_df)), graph_use_df.osci, color="gray", alpha=0.5, label="osci")
      plot = ax2.plot(range(len(graph_use_df)), graph_use_df.macd_line, "-", color="blue", alpha=1, label="macd_line")
      plot = ax2.plot(range(len(graph_use_df)), graph_use_df.signal, "-", color="green", alpha=1, label="signal")

      h1, l1 = ax1.get_legend_handles_labels()
      h2, l2 = ax2.get_legend_handles_labels()
      plt.title('macd')
      plt.xlim(0, len(graph_use_df))
      ax1.grid(color='black', alpha=0.2)
      ax1.grid(color='gray', alpha=0.2)
      plt.xticks(np.arange(0, len(graph_use_df), x_axis_interval))            
      ax1.legend(h1+h2, l1+l2, loc='upper left')

    elif kinds == "signal":
      ema_window_1 = ema_window_1_main
      ema_window_2 = ema_window_2_main
      macd_line_window = macd_line_window_main

      graph_use_df = graph_use_df.set_index("time")
      graph_use_df = graph_use_df[[f"macd_line_{ema_window_1}_{ema_window_2}", f"signal_{macd_line_window}"]]

      fig = plt.figure(figsize=(20, 5))
      ax1 = fig.subplots()
      ax2 = ax1.twinx()

      for index, line in enumerate(graph_use_df):
        if index == 0:
          plot = ax1.plot(range(len(graph_use_df)), graph_use_df[line], "-", color="blue", alpha=1, label=f"macd_line_{ema_window_1}_{ema_window_2}")
        elif index == 1:
          plot = ax2.plot(range(len(graph_use_df)), graph_use_df[line], "-", color="green", alpha=1, label=f"signal_{macd_line_window}")
      h1, l1 = ax1.get_legend_handles_labels()
      h2, l2 = ax2.get_legend_handles_labels()

      plt.xlim(0, len(graph_use_df))
      ax1.grid(color='black', alpha=0.2)
      ax1.grid(color='gray', alpha=0.2)
      plt.xticks(np.arange(0, len(graph_use_df), x_axis_interval))            
      ax1.legend(h1+h2, l1+l2, loc='upper left')

    elif kinds == "rci":
      graph_use_df = graph_use_df.set_index("time")

      fast_line = 5
      slow_line = 8
      graph_use_df = graph_use_df[[f"rci_{fast_line}", f"rci_{slow_line}"]]
      graph_use_df_col = list(graph_use_df.columns)      

      fig = plt.figure(figsize=(24, 6))
      ax1 = fig.subplots()
      for index, line in enumerate(graph_use_df):      
        if index == 0: 
          plot = ax1.plot(range(len(graph_use_df)), graph_use_df[graph_use_df_col[index]], "-", color="blue", alpha=1, label=graph_use_df_col[index])
        else:
          plot = ax1.plot(range(len(graph_use_df)), graph_use_df[graph_use_df_col[index]], "-", color="green", alpha=1, label=graph_use_df_col[index])

      h1, l1 = ax1.get_legend_handles_labels()
      plt.title('rci')
      plt.xlim(0, len(graph_use_df))
      ax1.grid(color='black', alpha=0.2)
      ax1.grid(color='gray', alpha=0.2)
      plt.xticks(np.arange(0, len(graph_use_df), x_axis_interval))
      plt.yticks(np.arange(-100, 110, 10))
      ax1.legend(h1, l1, loc='upper left')      

    elif kinds == "ema_cross":
      graph_use_df = graph_use_df.set_index("time")
      fast_line = ema_preriod_fast_main
      slow_line = ema_preriod_slow_main
      graph_use_df = graph_use_df[[f"ema_close_{fast_line}", f"ema_close_{slow_line}"]]
      graph_use_df_col = list(graph_use_df.columns)

      fig = plt.figure(figsize=(24, 4))
      ax1 = fig.subplots()
      for index, line in enumerate(graph_use_df):
        if len(graph_use_df_col) == 6:
          if index <= 2: 
            plot = ax1.plot(range(len(graph_use_df)), graph_use_df[graph_use_df_col[index]], "-", color="blue", alpha=1, label=graph_use_df_col[index])
          else:
            plot = ax1.plot(range(len(graph_use_df)), graph_use_df[graph_use_df_col[index]], "-", color="green", alpha=1, label=graph_use_df_col[index])
        elif len(graph_use_df_col) == 4:
          if index <= 1: 
            plot = ax1.plot(range(len(graph_use_df)), graph_use_df[graph_use_df_col[index]], "-", color="blue", alpha=1, label=graph_use_df_col[index])
          else:
            plot = ax1.plot(range(len(graph_use_df)), graph_use_df[graph_use_df_col[index]], "-", color="green", alpha=1, label=graph_use_df_col[index])          
        elif len(graph_use_df_col) == 2:
          if index == 0: 
            plot = ax1.plot(range(len(graph_use_df)), graph_use_df[graph_use_df_col[index]], "-", color="blue", alpha=1, label=graph_use_df_col[index])
          else:
            plot = ax1.plot(range(len(graph_use_df)), graph_use_df[graph_use_df_col[index]], "-", color="green", alpha=1, label=graph_use_df_col[index])                   
        elif len(graph_use_df_col) == 3:
          if index == 0: 
            plot = ax1.plot(range(len(graph_use_df)), graph_use_df[graph_use_df_col[index]], "-", color="blue", alpha=1, label=graph_use_df_col[index])
          else:
            plot = ax1.plot(range(len(graph_use_df)), graph_use_df[graph_use_df_col[index]], "-", color="green", alpha=1, label=graph_use_df_col[index])                                

      plt.title('ema cross')
      h1, l1 = ax1.get_legend_handles_labels()
      plt.xlim(0, len(graph_use_df))
      ax1.grid(color='black', alpha=0.2)
      ax1.grid(color='gray', alpha=0.2)
      plt.xticks(np.arange(0, len(graph_use_df), x_axis_interval))
      ax1.legend(h1, l1, loc='upper left')

    elif kinds == "stochastics":
      k_period = sto_k_period_main
      d_period = sto_d_period_main
      slowing = sto_slowing_main

      graph_use_df = graph_use_df.set_index("time")
      graph_use_df = graph_use_df[[f"%K_{k_period}", f"%D_{d_period}", f"Slow%D_{slowing}"]]

      upper_line = 80
      lower_line = 20
      graph_use_df[upper_line] = upper_line
      graph_use_df[lower_line] = lower_line      

      fig = plt.figure(figsize=(20, 4))
      ax1 = fig.subplots()
      plot = ax1.plot(range(len(graph_use_df)), graph_use_df[f"%D_{d_period}"], "-", color="blue", alpha=1, label=f"%D_{d_period}")
      plot = ax1.plot(range(len(graph_use_df)), graph_use_df[f"Slow%D_{slowing}"], "--", color="limegreen", alpha=1, label=f"Slow%D_{slowing}")      

      plt.title('stochastics')
      h1, l1 = ax1.get_legend_handles_labels()
      plt.xlim(0, len(graph_use_df))
      ax1.grid(color='black', alpha=0.2)
      ax1.grid(color='gray', alpha=0.2)
      plt.xticks(np.arange(0, len(graph_use_df), x_axis_interval))
      plt.yticks(np.arange(0, 110, 10))
      ax1.legend(h1, l1, loc='upper left')      

    elif kinds == "adx":
      graph_use_df = graph_use_df.set_index("time")
      graph_use_df = graph_use_df[["close", "plus_di", "minus_di", "adx"]]      

      lower_line = 40
      graph_use_df[lower_line] = lower_line      

      fig = plt.figure(figsize=(20, 4))
      ax1 = fig.subplots()
      plot = ax1.plot(range(len(graph_use_df)), graph_use_df.adx, "-", color="black", alpha=0.5, label="ADX")            
      plot = ax1.plot(range(len(graph_use_df)), graph_use_df[lower_line], ls="--", color="purple", alpha=0.5, label=f"{lower_line}")      

      plt.title('adx')
      h1, l1 = ax1.get_legend_handles_labels()
      plt.xlim(0, len(graph_use_df))
      ax1.grid(color='black', alpha=0.2)
      ax1.grid(color='gray', alpha=0.2)
      plt.xticks(np.arange(0, len(graph_use_df), x_axis_interval))
      plt.yticks(np.arange(0, 110, 10))
      ax1.legend(h1, l1, loc='upper left')

    elif kinds == "trend_change":
      graph_use_df = graph_use_df.set_index("time")
      graph_use_df = graph_use_df[[f'trend_clac_change_{val_set_dic["trend_change"]}_ema', f'trend_clac_change_{val_set_dic["trend_change"]}_sma']]      
      graph_use_df_col = list(graph_use_df.columns)      

      graph_use_df["zero_line"] = 0
      graph_use_df["upper_line_1"] = 0.015
      graph_use_df["lower_line_1"] = -0.015
      graph_use_df["upper_line_2"] = 0.02
      graph_use_df["lower_line_2"] = -0.02
      graph_use_df["upper_line_3"] = 0.03
      graph_use_df["lower_line_3"] = -0.03      

      fig = plt.figure(figsize=(24, 8))
      ax1 = fig.subplots()
      for index, line in enumerate(graph_use_df):      
        if index == 0: 
          plot = ax1.plot(range(len(graph_use_df)), graph_use_df[graph_use_df_col[index]], "-", color="blue", alpha=1, label=graph_use_df_col[index])
        elif index == 1:
          plot = ax1.plot(range(len(graph_use_df)), graph_use_df[graph_use_df_col[index]], "-", color="green", alpha=1, label=graph_use_df_col[index])

      plot = ax1.plot(range(len(graph_use_df)), graph_use_df["zero_line"], "-", color="purple", alpha=0.5, label="zero_line")
      plot = ax1.plot(range(len(graph_use_df)), graph_use_df["upper_line_1"], "-", color="green", alpha=0.3, label="upper_line_1_0.0015")      
      plot = ax1.plot(range(len(graph_use_df)), graph_use_df["lower_line_1"], "-", color="green", alpha=0.3, label="lower_line_1_0.0015")                  
      plot = ax1.plot(range(len(graph_use_df)), graph_use_df["upper_line_2"], "-", color="orange", alpha=0.5, label="upper_line_2_0.0020")      
      plot = ax1.plot(range(len(graph_use_df)), graph_use_df["lower_line_2"], "-", color="orange", alpha=0.5, label="lower_line_2_0.0020")                       
      plot = ax1.plot(range(len(graph_use_df)), graph_use_df["upper_line_3"], "-", color="red", alpha=0.3, label="upper_line_3_0.0030")      
      plot = ax1.plot(range(len(graph_use_df)), graph_use_df["lower_line_3"], "-", color="red", alpha=0.3, label="lower_line_3_0.0030")           

      plt.title('trend')
      h1, l1 = ax1.get_legend_handles_labels()
      plt.xlim(0, len(graph_use_df))
      ax1.grid(color='black', alpha=0.2)
      ax1.grid(color='gray', alpha=0.2)
      plt.xticks(np.arange(0, len(graph_use_df), x_axis_interval))
      plt.yticks(np.arange(-0.05, 0.06, 0.01))      
      ax1.legend(h1, l1, loc='upper left')

    elif kinds == "ichimoku":
      graph_use_df = deepcopy(data)
      graph_use_df = calculate_ichimoku_kinnkouhyou(graph_use_df, mode="full")

      if type(graph_time_range) != type(None):
        graph_use_df = graph_use_df[-int(len(graph_use_df)*graph_time_range):]
        graph_use_df = graph_use_df.reset_index(drop=True)

      if type(graph_set_time_start) != type(None):
        graph_use_df = graph_use_df[graph_set_time_start<=graph_use_df["time"]]
        graph_use_df = graph_use_df.reset_index(drop=True) 

      if type(graph_set_time_end) != type(None):
        graph_use_df = graph_use_df[graph_set_time_end>=graph_use_df["time"]]
        graph_use_df = graph_use_df.reset_index(drop=True)  

      graph_use_df = graph_use_df.set_index("time")
      graph_use_df = graph_use_df[["close", "conversion_line", "base_line", "leading_span1", "leading_span2", "lagging_span"]]

      fig = plt.figure(figsize=(20, 6))
      ax = fig.subplots()

      idx1 = act_data.index[act_data["close"] >= act_data["open"]]
      idx0 = act_data.index[act_data["close"] < act_data["open"]]
      act_data["body"] = act_data["close"] - act_data["open"]
      act_data["body"] = act_data["body"].abs()
      ax.bar(idx1, act_data.loc[idx1, "body"], width = 1, bottom = act_data.loc[idx1, "open"], linewidth = 1, color = "#33b066", zorder = 2, alpha=0.8)
      ax.bar(idx0, act_data.loc[idx0, "body"], width = 1, bottom = act_data.loc[idx0, "close"], linewidth = 1, color = "#ff5050", zorder = 2, alpha=0.8)
      ax.vlines(act_data.index, act_data["low"], act_data["high"], linewidth = 1, color="#666666", zorder=1)      

      plot_1 = ax.plot(range(len(graph_use_df)), graph_use_df["conversion_line"], "-", label="conversion_line", color="blue", alpha=0.5, linewidth=2)
      plot_2 = ax.plot(range(len(graph_use_df)), graph_use_df["base_line"], "-", label="base_line", color="green", alpha=0.5, linewidth=2)        
      plot_3 = ax.plot(range(len(graph_use_df)), graph_use_df["leading_span1"], "-", label="leading_span1", color="orange", alpha=0.1)
      plot_4 = ax.plot(range(len(graph_use_df)), graph_use_df["leading_span2"], "-", label="leading_span2", color="purple", alpha=0.1)
      plot_5 = ax.plot(range(len(graph_use_df)), graph_use_df["lagging_span"], "-", label="lagging_span", color="red", alpha=0.5, linewidth=2)

      ax.fill_between(range(len(graph_use_df)), graph_use_df["leading_span1"], graph_use_df["leading_span2"], color="black", alpha=0.1)

      plt.title('ichimoku')
      h1, l1 = ax.get_legend_handles_labels()

      plt.xlim(0, len(graph_use_df))
      ax.grid(color='black', alpha=0.2)
      ax.grid(color='gray', alpha=0.2)
      plt.xticks(np.arange(0, len(graph_use_df), x_axis_interval))
      ax.legend(h1, l1, loc='upper left')      

    plt.show()

####################################################################################################
#-- 周期性確認用の関数


#-- リサンプリング
def data_resampling(data, freq):
  data = deepcopy(data)
  data = data.resample(freq).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
  data.dropna(how="any", axis=0, inplace=True)
  data.reset_index(inplace=True)
  return data


#-- 第何週か取得
def get_nth_week2_datetime_dt(dt, firstweekday=6):
    first_dow = dt.replace(day=1).weekday()
    offset = (first_dow - firstweekday) % 7
    return (dt.day + offset - 1) // 7 + 1


#-- freqごとにtime列を処理
def time_data_fix(data, freq):
  data = deepcopy(data)
  if freq == "Min" or freq == "H":
    data["time"] = data["time"].apply(lambda x: str(x)[str(x).find(" "):])
  elif freq == "D":
    data["time"] = data["time"].apply(lambda x: str(x)[str(x).find("-")+1:str(x).find(" ")])    
  elif freq == "Week_day":
    data["week_num"] = data["time"].apply(lambda x: x.weekday())  
    data["time"] = data["time"].apply(lambda x: str(x)[:str(x).find(" ")])    
  elif freq == "Week":
    data["nth_week_num"] = data["time"].apply(lambda x: get_nth_week2_datetime_dt(x))  
    data["time"] = data["time"].apply(lambda x: str(x)[:str(x).find(" ")]) 
  elif freq == "M":
    data["time"] = data["time"].apply(lambda x: str(x)[str(x).find("-")+1:str(x).rfind("-")])    
  return data


#-- freqごとにグルーピング&平均で集計処理
def grouping_aggregation(data, freq):
  data = deepcopy(data)
  if freq == "Min" or freq == "H":
    group_data_mean = data.groupby("time").mean()
  elif freq == "D":
    group_data_mean = data.groupby("time").mean()
  elif freq == "Week_day":
    data.drop("time", axis=1, inplace=True)
    data.rename({"week_num": "time"}, axis=1, inplace=True)
    data = data.reindex(columns=["time", "open", "high", "low", "close"])
    group_data_mean = data.groupby("time").mean()
    group_data_mean.reset_index(inplace=True)
    group_data_mean["time"] = group_data_mean["time"].apply(lambda x:
                                                              "月曜" if x == 0 else \
                                                              "火曜" if x == 1 else \
                                                              "水曜" if x == 2 else \
                                                              "木曜" if x == 3 else \
                                                              "金曜" if x == 4 else \
                                                              "土曜" if x == 5 else "日曜")
    group_data_mean.set_index("time", inplace=True)    
  elif freq == "Week":
    data.drop("time", axis=1, inplace=True)
    data.rename({"nth_week_num": "time"}, axis=1, inplace=True)
    data = data.reindex(columns=["time", "open", "high", "low", "close"])
    group_data_mean = data.groupby("time").mean()
    group_data_mean.reset_index(inplace=True)
    group_data_mean["time"] = group_data_mean["time"].apply(lambda x:
                                                              "第1週" if x == 1 else \
                                                              "第2週" if x == 2 else \
                                                              "第3週" if x == 3 else \
                                                              "第4週" if x == 4 else \
                                                              "第5週" if x == 5 else "第6週")
    group_data_mean.set_index("time", inplace=True)
  elif freq == "M":
    group_data_mean = data.groupby("time").mean()
    group_data_mean.reset_index(inplace=True)
    group_data_mean["time"] = group_data_mean["time"].apply(lambda x: str(int(x)) + "月")
    group_data_mean.set_index("time", inplace=True)   
    
  group_data_mean["oc_diff"] = group_data_mean["close"] - group_data_mean["open"]
    
  return group_data_mean


#-- freqごとにグラフ表示処理。close及び、oc_diff（open_closeの差分）
def periodicity_check_graph(data, freq="Min", render_mode="oc_diff"):
  data = deepcopy(data)
  data["zero_line"] = 0

  fig = plt.figure(figsize=(24, 6))
  ax1 = fig.subplots()
  ax1.get_xticklabels()
  plt.setp(ax1.get_xticklabels(), rotation=75, ha="right")
  plt.rcParams["font.size"] = 14
  ax1.grid(color='black', alpha=0.2)
  ax1.grid(color='gray', alpha=0.2)    

  if render_mode == "close_only":
    data = data[["close"]]
    for index, line in enumerate(data): 
      if freq == "D":
        plot = ax1.plot(data.index, data, "-", color="blue", alpha=1, label="close")
        locator = mdates.DayLocator(interval=1)
        ax1.xaxis.set_major_locator(locator)        
      else:
        plot = ax1.plot(data.index, data, "-", color="blue", alpha=1, label="close")

    plt.title('price')

  elif render_mode == "oc_diff":
    data = data[["oc_diff", "zero_line"]]
    if freq == "D":
      plot = ax1.plot(data.index, data["oc_diff"], "-", color="blue", alpha=1, label="oc_diff")
      locator = mdates.DayLocator(interval=1)
      ax1.xaxis.set_major_locator(locator)        
    else:
      plot = ax1.plot(data.index, data["oc_diff"], "-", color="blue", alpha=1, label="oc_diff")
      
    if freq == "D":      
      plt.yticks(np.arange(-3, 4, 1))
    elif freq == "Week_day":
      plt.yticks(np.arange(-0.20, 0.30, 0.04))        
    elif freq == "Week":
      plt.yticks(np.arange(-2.0, 3.0, 0.4))
    elif freq == "M":
      plt.yticks(np.arange(-5, 6, 1))        
    else:
      plt.yticks(np.arange(-0.05, 0.06, 0.01))
      
    plot = ax1.plot(data.index, data["zero_line"], "--", color="purple", alpha=1, label="0")        

    plt.title('trend')
        
  plt.show()


#-- 周期性確認のメインプロセス関数
def periodicity_analysis_main_process(data, start_periods=None):
  # 元のOHLCデータ
  data = deepcopy(data)
  data["time"] = data["time"].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))

  if type(start_periods) != type(None):
    data = data[data["time"] >= start_periods]
  data.set_index('time', inplace=True)

  resampled_data_15min = data_resampling(data, "15Min")
  resampled_data_30min = data_resampling(data, "30Min")
  resampled_data_1h = data_resampling(data, "1H")
  resampled_data_1d = data_resampling(data, "1d")
  resampled_data_7d = data_resampling(data, "7D")
  resampled_data_1m = data_resampling(data, "1M")

  fixed_resampled_data_15min = time_data_fix(resampled_data_15min, freq="Min")
  fixed_resampled_data_30min = time_data_fix(resampled_data_30min, freq="Min")    
  fixed_resampled_data_1h = time_data_fix(resampled_data_1h, freq="H")
  fixed_resampled_data_1d = time_data_fix(resampled_data_1d, freq="D")
  fixed_resampled_data_week_day = time_data_fix(resampled_data_1d, freq="Week_day")  
  fixed_resampled_data_7d = time_data_fix(resampled_data_7d, freq="Week")
  fixed_resampled_data_1m = time_data_fix(resampled_data_1m, freq="M")

  group_data_mean_15min = grouping_aggregation(fixed_resampled_data_15min, "Min")
  group_data_mean_30min = grouping_aggregation(fixed_resampled_data_30min, "Min")    
  group_data_mean_1h = grouping_aggregation(fixed_resampled_data_1h, freq="H")
  group_data_mean_1d = grouping_aggregation(fixed_resampled_data_1d, freq="D")
  group_data_mean_week_day = grouping_aggregation(fixed_resampled_data_week_day, freq="Week_day")  
  group_data_mean_7d = grouping_aggregation(fixed_resampled_data_7d, freq="Week")
  group_data_mean_1m = grouping_aggregation(fixed_resampled_data_1m, freq="M")

  print("●"+BLUE_BOLD+"周期性確認"+END+"\n")

  print("・15分単位")
  periodicity_check_graph(group_data_mean_15min, freq="Min", render_mode="close_only")
  periodicity_check_graph(group_data_mean_15min, freq="Min", render_mode="oc_diff")

  print("\n・30分単位")
  periodicity_check_graph(group_data_mean_30min, freq="Min", render_mode="close_only")
  periodicity_check_graph(group_data_mean_30min, freq="Min", render_mode="oc_diff")

  print("\n・1時間単位")
  periodicity_check_graph(group_data_mean_1h, freq="H", render_mode="close_only")
  periodicity_check_graph(group_data_mean_1h, freq="H", render_mode="oc_diff")

  print("\n・1日単位")
  periodicity_check_graph(group_data_mean_1d, freq="D", render_mode="close_only")
  periodicity_check_graph(group_data_mean_1d, freq="D", render_mode="oc_diff")

  print("\n・曜日単位")
  periodicity_check_graph(group_data_mean_week_day, freq="Week_day", render_mode="close_only")
  periodicity_check_graph(group_data_mean_week_day, freq="Week_day", render_mode="oc_diff")  

  print("\n・第何週単位")
  periodicity_check_graph(group_data_mean_7d, freq="Week", render_mode="close_only")
  periodicity_check_graph(group_data_mean_7d, freq="Week", render_mode="oc_diff")

#  periodicity_check_graph(group_data_mean_1m, freq="M", render_mode="close_only")
#  periodicity_check_graph(group_data_mean_1m, freq="M", render_mode="oc_diff")

####################################################################################################
