
from copy import copy, deepcopy
from time import time, sleep
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import json

import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import lightgbm as lgb

from keras.models import Model, Sequential
from keras.layers import GRU, SimpleRNN, LSTM
from keras_ordered_neurons import ONLSTM
from keras.layers import Dense, Dropout, Input, BatchNormalization, Activation, LeakyReLU
from keras.optimizers import Adam, SGD, Adamax, RMSprop
from keras.callbacks import EarlyStopping
from keras import regularizers

import tensorflow as tf
from tensorflow.keras.layers import Activation

from keras.models import model_from_json
from keras.utils import custom_object_scope
from tensorflow.keras.utils import get_custom_objects

####################################################################################################

#-- 回帰用rnnモデルインスタンス生成用クラス
class CreateRnnModel:
  def __init__(
    self,
    *,
    use_model=None,
    epochs=300,
    batch_size=32, 
    layer_num=None,
    units=None,
    chunk_size=None,
    activation=None,
    batch_normalization_use=False,
    dropout_rate=None,
    optimizer=None,
    loss=None,
    min_delta=None,
    restore_best_weights=None
  ):
    self.use_model = use_model
    self.epochs = epochs
    self.batch_size = batch_size
    self.layer_num = layer_num
    self.units = units
    self.chunk_size = chunk_size
    self.activation = activation
    self.batch_normalization_use = batch_normalization_use
    self.dropout_rate = dropout_rate
    self.optimizer = optimizer
    self.loss = loss
    self.min_delta = min_delta
    self.restore_best_weights = restore_best_weights


  #-- モデル構造について、クラス継承した際に編集しやすくするために分けた部分
  def layer_composition(self, i, input_shape):
    if self.use_model == "onlstm":
      composition_1 = ONLSTM(
        units=self.units, activation=self.activation, chunk_size=self.chunk_size,
        return_sequences=False, input_shape=input_shape
      )
      composition_2 = ONLSTM(
        units=self.units, activation=self.activation, chunk_size=self.chunk_size,
        return_sequences=True, input_shape=input_shape
      )
      composition_3 = ONLSTM(
        units=self.units, activation=self.activation, chunk_size=self.chunk_size,
        return_sequences=True
      )      
      composition_4 = ONLSTM(
        units=self.units, activation=self.activation, chunk_size=self.chunk_size,
      )

    elif self.use_model == "lstm":
      composition_1 = LSTM(
        units=self.units, activation=self.activation,
        return_sequences=False, input_shape=input_shape
      )
      composition_2 = LSTM(
        units=self.units, activation=self.activation,
        return_sequences=True, input_shape=input_shape
      )
      composition_3 = LSTM(
        units=self.units, activation=self.activation,
        return_sequences=True
      )      
      composition_4 = LSTM(
        units=self.units, activation=self.activation
      )
    elif self.use_model == "gru":
      composition_1 = GRU(
        units=self.units, activation=self.activation,
        return_sequences=False, input_shape=input_shape
      )
      composition_2 = GRU(
        units=self.units, activation=self.activation,
        return_sequences=True, input_shape=input_shape
      )
      composition_3 = GRU(
        units=self.units, activation=self.activation,
        return_sequences=True
      )      
      composition_4 = GRU(
        units=self.units, activation=self.activation
      )                   

    if i == 0:
      if self.layer_num == 1:
        self.model.add(composition_1)
      else:
        self.model.add(composition_2)
      if self.batch_normalization_use == True:
        self.model.add(BatchNormalization())
      if type(self.dropout_rate) != type(None):
        self.model.add(Dropout(self.dropout_rate))

    elif i == self.layer_num - 1:
      self.model.add(composition_4)      
      if self.batch_normalization_use == True:
        self.model.add(BatchNormalization())
      if type(self.dropout_rate) != type(None):
        self.model.add(Dropout(self.dropout_rate))      

    else:
      self.model.add(composition_3)  
      if self.batch_normalization_use == True:
        self.model.add(BatchNormalization())
      if type(self.dropout_rate) != type(None):
        self.model.add(Dropout(self.dropout_rate))

    return self.model


  #-- モデル構築の起点関数。Sequentialからcompileの手前まで
  def create_model(self, input_shape, output_shape):
    self.model = Sequential()

    for i in range(self.layer_num):
        self.model = self.layer_composition(i, input_shape)

    self.model.add(Dense(units=output_shape[1], activation='linear'))
    return self.model


  #-- fit
  def fit(self, train_x, train_y, validation_data=None):
    if type(validation_data) != type(None):
      validation_data = (validation_data[0], validation_data[1])
    else:
      validation_data = None

    self.history = self.model.fit(
      train_x, train_y, 
      epochs = self.epochs, batch_size = self.batch_size,
      validation_data=validation_data,
      shuffle=False
      , callbacks = EarlyStopping(
        patience = 0,
        monitor = "val_loss",
        min_delta = self.min_delta,
        restore_best_weights = self.restore_best_weights
  #      verbose = self.verbose
      )
    )

    return self.history


  #-- predict
  def predict(self, X):
    return self.model.predict(X)


  #-- hparamsをまとめて渡すために必要な関数
  def set_params(self, **params):
    for param, value in params.items():
      setattr(self, param, value)
    return self


  #-- 設定したhparamsを取得する関数
  def get_params(self, deep = True):
    return {
      self.use_model: {
        "epochs": self.epochs,
        "batch_size": self.batch_size,
        "layer_num": self.layer_num,
        "units": self.units,
#        "chunk_size": self.chunk_size,
        "activation": self.activation,
        "batch_normalization_use": self.batch_normalization_use,
        "dropout_rate": self.dropout_rate,
        "optimizer": self.optimizer,
        "loss": self.loss,
        "min_delta": self.min_delta,
        "restore_best_weights": self.restore_best_weights,
      }
    }


  #-- ファインチューニングを行う際に使用
  def create_fine_tuning_model(self, load_model, output_shape):
    model_1_index = None
    dropout_1_index = None
    model_2_index = None
    for i, layer in enumerate(load_model.layers):
      if layer.name == 'gru_81':
          model_1_index = i

      if layer.name == 'dropout_20':
          dropout_1_index = i

      if layer.name == 'gru_87':
          model_2_index = i

    if model_1_index is not None:
        load_model.layers[model_1_index].trainable = False

    if dropout_1_index is not None:
        load_model.layers[dropout_1_index].trainable = False      

    if model_2_index is not None:
        load_model.layers[model_2_index].return_sequences = True

    model = Sequential()
    model.add(load_model)

    composition_1 = GRU(
      units=self.units, activation=self.activation,
      return_sequences=True
    )
    composition_2 = GRU(
      units=self.units, activation=self.activation
    )
    composition_3 = LSTM(
      units=self.units, activation=self.activation
    )

    model.add(composition_1)
    model.add(Dropout(self.dropout_rate))
    
    model.add(composition_2)
    model.add(Dropout(self.dropout_rate))     

    model.add(Dense(units=output_shape[1], activation='linear'))

    model.compile(optimizer=self.optimizer, loss=self.loss)
    model.summary()

    return model


  #-- rnnインスタンス作成 メインプロセス関数
  def rnn_main_process(self, input_shape, output_shape, train_x, train_y, validation_data=None):
    self.model = self.create_model(input_shape, output_shape)
    self.model.compile(optimizer = self.optimizer, loss = self.loss)
    self.model.summary()
    self.history = self.fit(train_x, train_y, validation_data=validation_data)
    return [self.model, self.history]
  
###########################################################################

#-- 活性化関数 Mishを使用するためのもの
class Mish(Activation):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X = Activation('Mish', name="conv1_act")(X_input)
    '''

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'


def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))

get_custom_objects().update({'Mish': Mish(mish)})

###########################################################################

#-- モデルロード等でループでMishを使用するために必要な関数
def reset_mish():
  class Mish(Activation):
    def __init__(self, activation, **kwargs):
      super(Mish, self).__init__(activation, **kwargs)
      self.__name__ = 'mish'

  def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))

  get_custom_objects().update({'mish': Mish(mish)})    

####################################################################################################

#-- モデルのセーブ、ロードを行うためのもの
def create_model_save_and_load(target_model_path, mode="load", model_add_info="", model=None):
  if mode == "save":
    # モデルの構造を保存
    model_architecture = model.to_json()
    with open(f'{target_model_path}test_model_architecture{model_add_info}.json', 'w') as json_file:
        json_file.write(model_architecture)

    # 学習済みの重みを保存
    model.save_weights(f'{target_model_path}test_model_weights{model_add_info}.h5')

  elif mode == "load":
    custom_objects = {'ONLSTM': ONLSTM, 'Mish': Mish(mish)}

    # モデルの構造を読み込む
    with open(f'{target_model_path}test_model_architecture{model_add_info}.json', 'r') as json_file:
        loaded_model_architecture = json_file.read()

    # 新しいモデルのインスタンスを作成し、学習済みの重みをロードする
    with custom_object_scope(custom_objects):
        loaded_model = model_from_json(loaded_model_architecture)
        loaded_model.load_weights(f'{target_model_path}test_model_weights{model_add_info}.h5')
        loaded_model.compile(optimizer=Adam(), loss='mean_squared_error')
    return loaded_model

####################################################################################################

#-- 回帰用lightgbm
def future_pred_use_lightgbm(
    data, params, sc, df, col_name_list, test_rate = 0.1, train_rate = 0.9,
    future_period=5, freq_num=1, freq="Min"
):      
    bk_data = deepcopy(data)            
    use_data = deepcopy(data)
    time_data = use_data[["time"]]
    use_data = use_data.set_index("time")            
    col_list = list(use_data.columns)

    model_name = "lgb"
    sc_use = params['scaler_use']

    if type(params) != type(None):
        if 'epochs' in params:
            epochs = params['epochs']
            del params['epochs']
        elif "n_estimators" in params:
            epochs = params["n_estimators"]
            del params["n_estimators"]                    
        if 'scaler_use' in params:
            del params['scaler_use']            

    if sc_use == "minmax":
        sc = MinMaxScaler(feature_range=(0, 1))        
        use_data = sc.fit_transform(use_data)        
    elif sc_use == "standard":
        sc = StandardScaler()
        use_data = sc.fit_transform(use_data)
    else:
        use_data = np.array(use_data)
    if type(params) != type(None):
        if 'epochs' in params:
            epochs = params['epochs']
            del params['epochs']
        elif "n_estimators" in params:
            epochs = params["n_estimators"]            
        if 'scaler_use' in params:
            del params['scaler_use']            


    if sc_use == "minmax":
        sc = MinMaxScaler(feature_range=(0, 1))        
        use_data = sc.fit_transform(use_data)        
    elif sc_use == "standard":
        sc = StandardScaler()
        use_data = sc.fit_transform(use_data)
    else:
        use_data = np.array(use_data)       

    train_df, test_df = train_test_split(use_data, test_size=test_rate, shuffle=False)
    train_df, valid_df = train_test_split(train_df, train_size=train_rate, shuffle=False)

    models = []

    for i in range(use_data.shape[1]):
        if model_name == "lgb":
            model = lgb.LGBMRegressor()      
            train_data = lgb.Dataset(
                data=train_df[:-1], 
                label=train_df[1:, i], 
                feature_name=col_list
            )
            valid_data = lgb.Dataset(
                data=valid_df[:-1], 
                label=valid_df[1:, i], 
                feature_name=col_list
            )            

            model = lgb.train(
                params=params,
                train_set=train_data,
                valid_sets=valid_data,
                num_boost_round=epochs,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(False),
                ],
            )            

        models.append(model)

    future_data = []

    next_X = [test_df[-1]]
    for i in range(future_period):
        # 最後の時刻のデータを使って予測
        if model_name == "lgb":
            next_X = np.array([model.predict(next_X, num_iteration=model.best_iteration) for i, model in enumerate(models)]).reshape(1, -1)
        elif model_name == "lr":
            next_X = np.array([model.predict(next_X) for i, model in enumerate(models)]).reshape(1, -1)
        future_data.append(next_X[0].tolist())

    test_pred_data = future_data

    if sc_use == "minmax" or sc_use == "standard":
        test_pred_data = sc.inverse_transform(test_pred_data)

    test_pred_data = pd.DataFrame(test_pred_data)
    test_pred_data.columns = col_list

    test_time_data = bk_data.time.max() + timedelta(minutes=freq_num)
    test_time_data = pd.DataFrame(
          pd.date_range(start=test_time_data, periods=future_period, freq=f'{freq_num}{freq}')
        , columns=["time"])
    test_pred_data = pd.concat([test_time_data, test_pred_data], axis=1)

    future_ohlc_data = pd.concat([bk_data, test_pred_data], axis=0)[["time", "close", "open", "high", "low"]]
    future_ohlc_data = future_ohlc_data.reset_index()
    future_ohlc_data = future_ohlc_data.drop("index", axis=1)            

    test_pred_data = test_pred_data.rename({"close": "yhat"}, axis=1)

    all_join_data = pd.concat([bk_data, test_pred_data], axis=0)
    all_join_data = all_join_data.rename({"close": "y"}, axis=1)        
    all_join_data = all_join_data.reset_index()
    all_join_data = all_join_data.drop("index", axis=1)

    tmp_col_list = list(all_join_data.columns)
    tmp_col_list.remove("yhat")
    tmp_col_list.insert(2, "yhat")
    all_join_data = all_join_data.reindex(columns=tmp_col_list)

    return [future_ohlc_data, all_join_data]            

####################################################################################################

#-- optunaで作成したjsonを読み込むための関数。スコアでソートし良い順で出力
def json_params_loader(data, load_mode = 0, select_score="mse_calc"):
  json_open = open('./json/'+data+'.json', 'r')
  json_load = json.load(json_open)

  if load_mode == 0:
    return json_load
  elif load_mode == 1 or load_mode == 2:
    score_check_dic = {}
    for index, (key, val) in enumerate(json_load["all_data"].items()):
        score_check_dic[key] = val["score"][select_score]
    if (select_score == 'r2_calc') or (select_score == 'accuracy') or (select_score == 'recall') or (select_score == 'precision') or (select_score == 'f1'):
        sort_score_check_dic = sorted(score_check_dic.items(), key=lambda x:x[1], reverse=True)
    else:
        sort_score_check_dic = sorted(score_check_dic.items(), key=lambda x:x[1], reverse=False)
    best_prams_number = sort_score_check_dic[0][0]
    best_prams = json_load["all_data"][best_prams_number]['params']
    if load_mode == 1:
        return [best_prams, json_load["all_data"][best_prams_number]["score"], best_prams_number, sort_score_check_dic]
    elif load_mode == 2:
        score_sorted_list = [key[0] for key in sort_score_check_dic]
        params_score_sorted_dic = {}
        for key in score_sorted_list:
            params_score_sorted_dic[key] = json_load["all_data"][key]        
        return [params_score_sorted_dic, score_sorted_list]        

####################################################################################################

#-- データセット作成メインプロセス関数
def create_pred_data(
  data, rnn_batch_size, train_rate, test_rate, 
  variable_delete_rate_min=0.3, variable_delete_rate_max=None, 
  outlier_handling_use=True, multiplier=2,
  select_mode=0, dim_quantity=1, reverse=False
):
    pred_use_data = deepcopy(data)
    pred_use_data = pred_use_data.drop("real_volume", axis=1)
    pred_use_data = pred_use_data.dropna(how="any", axis=0)
    pred_use_data = pred_use_data.reset_index()
    pred_use_data = pred_use_data.set_index("time")    
    copy_target_col = pred_use_data[["close"]]
    pred_use_data = pred_use_data.drop(["index", "close"], axis=1)
    pred_use_data = pd.concat([copy_target_col, pred_use_data], axis=1)

    if outlier_handling_use == True:
      pred_use_data = outlier_handling(pred_use_data, multiplier)

    corr = pred_use_data.corr()

    pred_use_data, delete_cols_dic = explanatory_variable(
      pred_use_data, corr, variable_delete_rate_min, variable_delete_rate_max, select_mode=select_mode, dim_quantity=dim_quantity, reverse=reverse
    )
    pred_use_data = pred_use_data.reset_index()

    use_variable_check_df = deepcopy(pred_use_data)    
    use_variable_check_df = use_variable_check_df.set_index("time")

    train_x, train_y, valid_x, valid_y, test_x, test_y, input_shape, output_shape, sc, df, col_name_list = \
    rnn_data_processing(pred_use_data, rnn_batch_size, train_rate, test_rate)

    return [pred_use_data, use_variable_check_df, delete_cols_dic, train_x, train_y, valid_x, valid_y, test_x, test_y, input_shape, output_shape, sc, df, col_name_list]

###########################################################################

#-- 外れ値除外。Hampel Identifier（平均値では無く中央値を使用）
def outlier_handling(data, multiplier):
  data.loc[
      (data[f"hampel_upper_{multiplier}"] < data["close"])                
  , "close"] = data[f"hampel_upper_{multiplier}"]

  data.loc[
      (data[f"hampel_lower_{multiplier}"] > data["close"])                
  , "close"] = data[f"hampel_lower_{multiplier}"]             
  return data

###########################################################################

#-- create_pred_data関数から呼び出される関数。正規化、rnn用にデータを作成した後にtrain_test_split
def rnn_data_processing(data, rnn_batch_size, train_rate, test_rate):
  df = deepcopy(data)

  copy_df = df.drop(["time"], axis=1)
  col_name_list = list(copy_df.columns)
  col_num = len(copy_df.columns)

  x = deepcopy(copy_df)
  sc = MinMaxScaler(feature_range=(0, 1))
  x = sc.fit_transform(x)  

  X, Y = [], []
  for i in range(len(x)):
      if (i+rnn_batch_size+1 >= len(x)):
          break
      X.append(x[i:i+rnn_batch_size])
      Y.append(x[i+rnn_batch_size+1])

  train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=test_rate, shuffle=False)
  train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, train_size=train_rate, shuffle=False)

  train_x = np.array(train_x).reshape(len(train_x), rnn_batch_size, col_num)
  test_x = np.array(test_x).reshape(len(test_x), rnn_batch_size, col_num)

  train_y = np.array(train_y).reshape(-1, col_num)
  test_y = np.array(test_y).reshape(-1, col_num) 

  valid_x = np.array(valid_x).reshape(len(valid_x), rnn_batch_size, col_num)
  valid_y = np.array(valid_y).reshape(-1, col_num)

  input_shape = (rnn_batch_size, col_num)
  output_shape = (1, col_num)

  return  [train_x, train_y, valid_x, valid_y, test_x, test_y, input_shape, output_shape, sc, df, col_name_list]

###########################################################################

#-- 相関係数ベースで説明変数を選定する際に使用する関数
def explanatory_variable(
  data, corr_data, variable_delete_rate_min=None, variable_delete_rate_max=None,
  select_mode=0, dim_quantity=1, reverse=False
):
  copy_data = deepcopy(data)
  corr_data = corr_data.iloc[0]

  if select_mode == 0:
    if type(variable_delete_rate_min) != type(None):
      delete_cols_dic = {}
      for index, (col_name, corr_val) in enumerate(corr_data.items()):
        if index >= 1:
          if variable_delete_rate_min > abs(corr_val):
            delete_cols_dic[col_name] = corr_val
          elif type(variable_delete_rate_max) != type(None) and variable_delete_rate_max < abs(corr_val):
            delete_cols_dic[col_name] = corr_val
          elif math.isnan(corr_val) == True:
            delete_cols_dic[col_name] = corr_val

      delete_name_list = list(delete_cols_dic.keys())
      copy_data.drop(delete_name_list, axis=1, inplace=True)
    else:
      delete_cols_dic = None
      copy_data = copy_data[[
        'close', 'low', 'high', 'open',
        'ema_6', 'ema_close_9', 'sma_7', 'hampel_Mid',
        'lower_sigma_2_20', 'upper_sigma_2_20', 'lower_sigma_3_20', 'upper_sigma_3_20'
      ]]

  elif select_mode == 1:
    corr_data = pd.DataFrame(abs(corr_data))
    if reverse == True:
      corr_data = corr_data.sort_values('close', ascending=[False])
    else:
      corr_data = corr_data.sort_values('close', ascending=[True])      
    corr_data = corr_data.reset_index()

    use_col_source = corr_data[["index"]]
    use_col = use_col_source.iloc[:dim_quantity]
    use_col = use_col.rename({"index": "col_name"}, axis=1)
    delete_cols_dic = use_col_source.iloc[dim_quantity:]
    delete_cols_dic = delete_cols_dic.rename({"index": "col_name"}, axis=1)    
    use_col = use_col["col_name"].tolist()
    delete_cols_dic = delete_cols_dic["col_name"].tolist()
    copy_data = copy_data[use_col]
  
  return copy_data, delete_cols_dic

###########################################################################

#-- 回帰用スコアを返す関数
def return_result_score(y_data, yhat_data):
    return_score = {}
    return_score["mae_calc"] = mean_absolute_error(y_data, yhat_data)
    return_score["mape_calc"] = mean_absolute_percentage_error(y_data, yhat_data)
    return_score["mse_calc"] = mean_squared_error(y_data, yhat_data)
    return_score["rmse_calc"] = np.sqrt(mean_squared_error(y_data, yhat_data))
    return_score["r2_calc"] = r2_score(y_data, yhat_data)
    return return_score

###########################################################################

#-- 主に通常学習させた際に結果を確認するためのもの
def rnn_result_display(
  model, train_x, train_y, valid_x, valid_y, test_x, test_y, 
  sc, df, col_name_list, batch_size=32, history=None
):
  if type(history) != type(None):
    history_df = pd.DataFrame(history.history)

    history_df.plot(figsize=(14, 4), color=["blue", "orange"])
    plt.legend()
    plt.show()

  print()
  score = model.evaluate(train_x, train_y, batch_size=batch_size, verbose=1)
  print(f'・train evaluate loss : {score}')  

  score = model.evaluate(valid_x, valid_y, batch_size=batch_size, verbose=1)
  print(f'・valid evaluate loss : {score}')  

  score = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=1)
  print(f'・test  evaluate loss : {score}')

  predict_y = model.predict(test_x, batch_size=batch_size, verbose=0)

  sc_inverse = sc.inverse_transform(test_y)
  predict_y = sc.inverse_transform(predict_y)

  sc_inverse_df = pd.DataFrame(sc_inverse)
  sc_inverse_df.columns = col_name_list
  sc_inverse_df = sc_inverse_df[["close"]]
  sc_inverse_df.rename({"close": "y"}, axis=1, inplace=True)  

  predict_y_df = pd.DataFrame(predict_y)
  predict_y_df.columns = col_name_list
  predict_y_df = predict_y_df[["close"]]  
  predict_y_df.rename({"close": "yhat"}, axis=1, inplace=True)

  x_datetime = deepcopy(df[["time"]])
  x_datetime = x_datetime.iloc[len(x_datetime)-len(predict_y):]
  x_datetime = x_datetime.iloc[len(x_datetime)-len(predict_y):]
  x_datetime.reset_index(inplace=True)
  x_datetime.drop("index", axis=1, inplace=True)

  result_df = x_datetime
  result_df = pd.concat([result_df, sc_inverse_df], axis=1)
  result_df = pd.concat([result_df, predict_y_df], axis=1)      

  result_df = result_df.rename({"close": "y"}, axis=1)

  graph_result_inner_df = deepcopy(result_df)
  graph_result_inner_df.set_index("time", inplace=True)

  return_score = return_result_score(graph_result_inner_df.y, graph_result_inner_df.yhat)
  return_score_keys = list(return_score.keys())

  print()
  print("・range test")
  print(f'  mae  : {return_score[return_score_keys[0]]}')
  print(f'  mape : {return_score[return_score_keys[1]]}')
  print(f'  mse  : {return_score[return_score_keys[2]]}')
  print(f'  rmse : {return_score[return_score_keys[3]]}')
  print(f'  r2   : {return_score[return_score_keys[4]]}')

  print()
  graph_result_inner_df.plot(figsize=(20, 6), color=["blue", "orange"])
  plt.show()

###########################################################################

#-- こちらはモデルロードした際に結果を確認するためのもの
def load_model_score_check(
  model, test_x, test_y,
  sc, df, col_name_list, batch_size=32,
  render_graph=False, test_mode=0
):
  print("\n・精度確認")

  score = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=0)
  print(f'・test  evaluate loss : {score}')

  if test_mode == 0:
    predict_y = model.predict(test_x, batch_size=batch_size, verbose=0)

  elif test_mode == 1:
    predict_y = []

    loop_cnt=0
    for step in range(len(test_x)):
      y_future = model.predict(test_x[step:step+1], batch_size=1, verbose=0)    
      y_future = y_future[0].tolist()
      predict_y.append(y_future)
      print("\r"+f'{loop_cnt}/{len(test_x)}',end="")
      loop_cnt+=1

    predict_y = np.array(predict_y)

  elif test_mode == 2:
    x_future = test_x[0:1]
    predict_y = []

    loop_cnt=0
    for step in range(len(test_x)):
      y_future = model.predict(x_future, batch_size=1, verbose=0)    
      x_future = x_future[0][1:].tolist()
      y_future = y_future[0].tolist()
      x_future.append(y_future)
      x_future = np.array([x_future])
      predict_y.append(y_future)
      print("\r"+f'{loop_cnt}/{len(test_x)}',end="")
      loop_cnt+=1

    predict_y = np.array(predict_y)
    score = model.evaluate(test_x, predict_y, batch_size=1, verbose=0)
    print(); print(f'・test  evaluate loss : {score}')


  sc_inverse = sc.inverse_transform(test_y)
  predict_y = sc.inverse_transform(predict_y)

  sc_inverse_df = pd.DataFrame(sc_inverse)
  sc_inverse_df.columns = col_name_list
  sc_inverse_df = sc_inverse_df[["close"]]
  sc_inverse_df.rename({"close": "y"}, axis=1, inplace=True)  

  predict_y_df = pd.DataFrame(predict_y)
  predict_y_df.columns = col_name_list
  predict_y_df = predict_y_df[["close"]]  
  predict_y_df.rename({"close": "yhat"}, axis=1, inplace=True)

  x_datetime = deepcopy(df[["time"]])
  x_datetime = x_datetime.iloc[len(x_datetime)-len(predict_y):]
  x_datetime = x_datetime.iloc[len(x_datetime)-len(predict_y):]
  x_datetime.reset_index(inplace=True)
  x_datetime.drop("index", axis=1, inplace=True)

  result_df = x_datetime
  result_df = pd.concat([result_df, sc_inverse_df], axis=1)
  result_df = pd.concat([result_df, predict_y_df], axis=1)      

  result_df = result_df.rename({"close": "y"}, axis=1)

  graph_result_inner_df = deepcopy(result_df)
  graph_result_inner_df.set_index("time", inplace=True)

  return_score = return_result_score(graph_result_inner_df.y, graph_result_inner_df.yhat)
  return_score_keys = list(return_score.keys())

  print(f'・mae: {np.round(return_score[return_score_keys[0]], decimals=4)}', end="")
  print(f', mape: {np.round(return_score[return_score_keys[1]], decimals=4)}', end="")
  print(f', mse: {np.round(return_score[return_score_keys[2]], decimals=4)}', end="")
  print(f', rmse: {np.round(return_score[return_score_keys[3]], decimals=4)}', end="")
  print(f', r2: {np.round(return_score[return_score_keys[4]], decimals=4)}')

  if render_graph == True:
    print()
    graph_result_inner_df.plot(figsize=(20, 6), color=["blue", "orange"])
    plt.show()

  return score

####################################################################################################

#-- 設定した分の未来予測を行う関数。predictの結果を説明変数として使用し、ループさせる。完了後結果を出力、グラフ表示等
def future_predict_on_test(
  model, test_x, test_y, sc, df, col_name_list,
  future_period=None, freq_num=1, freq="Min", batch_size=32, render_graph=False, time_range=None
):

  x_future = test_x[-1:]
  list_test_x = test_x.tolist()

  for step in range(future_period):
    y_future = model.predict(x_future, batch_size=1, verbose=0)    
    x_future = x_future[0][1:].tolist()
    y_future = y_future[0].tolist()
    x_future.append(y_future)
    x_future = np.array([x_future])
    y_future = x_future[0].tolist()
    list_test_x.append(y_future)

  test_x = np.array(list_test_x)
  predict_y = model.predict(test_x, batch_size=1, verbose=0)

  sc_inverse = sc.inverse_transform(test_y)
  predict_y = sc.inverse_transform(predict_y)

  sc_inverse_df = pd.DataFrame(sc_inverse)
  sc_inverse_df.columns = col_name_list
  sc_inverse_df = sc_inverse_df[["close", "open", "high", "low"]]
  sc_inverse_df.rename({"close": "y"}, axis=1, inplace=True)  

  predict_y_df = pd.DataFrame(predict_y)
  predict_y_df.columns = col_name_list
  predict_y_df = predict_y_df[["close", "open", "high", "low"]] 
  predict_y_df.rename({"close": "yhat"}, axis=1, inplace=True)

  x_datetime = deepcopy(df[["time"]])
  x_datetime = x_datetime.iloc[len(x_datetime)-len(predict_y):]
  x_datetime = x_datetime.iloc[len(x_datetime)-len(predict_y):]
  x_datetime.reset_index(inplace=True)
  x_datetime.drop("index", axis=1, inplace=True)

  if freq == "Min":
    x_datetime = x_datetime.iloc[future_period:]
    x_datetime = x_datetime.reset_index()
    x_datetime = x_datetime.drop("index", axis=1)      
    
    foward_time = x_datetime["time"].max() + timedelta(minutes=freq_num)      
    future_datetime_df = pd.DataFrame(
      pd.date_range(start=foward_time, periods=future_period, freq=f'{freq_num}{freq}')
    , columns=["time"])

  future_datetime_df = future_datetime_df.reset_index()
  future_datetime_df = future_datetime_df.drop("index", axis=1)
  future_datetime_df["time"] = future_datetime_df["time"].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))    
  x_datetime["time"] = x_datetime["time"].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
  x_datetime = pd.concat([x_datetime, future_datetime_df], axis=0)
  x_datetime = x_datetime.reset_index()
  x_datetime = x_datetime.drop("index", axis=1)

  result_df = x_datetime
  result_df = pd.concat([result_df, sc_inverse_df], axis=1)
  result_df = pd.concat([result_df, predict_y_df], axis=1)      
  result_df = result_df.set_index("time")

  if render_graph == True:
    tmp_graph_df = deepcopy(result_df)[["y", "yhat"]]
    tmp_graph_df = tmp_graph_df[-300:]
    tmp_graph_df.reset_index(drop=True, inplace=True)    
    print("\n・predict 結果")
    display(tmp_graph_df)

    print("\n・精度確認")
    tmp_graph_df.plot(figsize=(20, 6), color=["blue", "orange"])

    tmp_graph_df = tmp_graph_df.dropna(how="any", axis=0)
    tmp_graph_df = tmp_graph_df.dropna(how="any", axis=1)    
    return_score = return_result_score(tmp_graph_df.y, tmp_graph_df.yhat)
    return_score_keys = list(return_score.keys())

    print(f'・mae: {np.round(return_score[return_score_keys[0]], decimals=4)}', end="")
    print(f', mape: {np.round(return_score[return_score_keys[1]], decimals=4)}', end="")
    print(f', mse: {np.round(return_score[return_score_keys[2]], decimals=4)}', end="")
    print(f', rmse: {np.round(return_score[return_score_keys[3]], decimals=4)}', end="")
    print(f', r2: {np.round(return_score[return_score_keys[4]], decimals=4)}')

    plt.show()

  sc_inverse_df = sc_inverse_df.rename({"y": "close"}, axis=1)
  predict_y_df = predict_y_df.rename({"yhat": "close"}, axis=1)  

  predict_y_df = predict_y_df.iloc[-future_period:]
  predict_y_df = predict_y_df.reset_index()
  predict_y_df = predict_y_df.drop("index", axis=1)

  join_df = pd.concat([sc_inverse_df, predict_y_df], axis=0)
  join_df = join_df.reset_index()
  join_df = join_df.drop("index", axis=1)  
  result_df = result_df.reset_index()
  join_df = pd.concat([result_df["time"], join_df], axis=1)

  return join_df

####################################################################################################

#-- lightgbmでマルチクラス分類を行った際に使用したスコア評価用関数
def return_classification_result_score(y_data, yhat_data, average='macro'):
    return {
        "accuracy": accuracy_score(y_data, yhat_data),
        "recall": recall_score(y_data, yhat_data, average=average),
        "precision": precision_score(y_data, yhat_data, average=average),
        "f1": f1_score(y_data, yhat_data, average=average),
    }

####################################################################################################

#-- 予測結果を確認するためのもの
def pred_data_process(test_pred_data, data_bk, target_col, score_display=False, graph_render=False):
  score = return_classification_result_score(test_pred_data.y, test_pred_data.yhat)
  if score_display == True:
    print()
    print(f'・accuracy : {score["accuracy"]}')
    print(f'・recall   : {score["recall"]}')
    print(f'・precision: {score["precision"]}')
    print(f'・f1_score : {score["f1"]}')
    print()    

  if graph_render == True:
    range_rate=1
    graph_render_df = test_pred_data[-int(len(test_pred_data)*range_rate):]    

    graph_render_df.plot(figsize=(20, 6))  
    plt.show()


  predict_data = deepcopy(test_pred_data)[["yhat"]]
  predict_data.columns = [target_col]
  predict_data.reset_index(inplace=True)
  tmp_target_time = predict_data["time"].min() - timedelta(minutes=5)

  data_bk.reset_index(inplace=True)

  tmp_y_data = data_bk[data_bk["time"]<=tmp_target_time][["time", target_col]]

  predict_data = pd.concat([tmp_y_data, predict_data],axis=0)
  predict_data.reset_index(drop=True, inplace=True)

  result_predict_data = deepcopy(data_bk)
  result_predict_data[target_col] = predict_data[target_col]
  result_predict_data.set_index("time", inplace=True)  

  if graph_render == True:
    check_df = pd.merge(data_bk, result_predict_data, on="time", how="inner")[["time", "trend_flag_x", "trend_flag_y"]]
    check_df = check_df[check_df["time"]>="2023/06/27"]
    check_df.set_index("time", inplace=True)

    check_df.plot(figsize=(24, 8))
    plt.show()

  return result_predict_data, score

####################################################################################################

'''
  params = {
    'objective': 'multiclass',
    'boosting_type': 'gbdt',
    'learning_rate': 0.02858892054144408,
    'lambda_l1': 5.9282478192424635e-05,
    'lambda_l2': 6.539869218696517e-06,
    'num_leaves': 69,
    'min_data_in_leaf': 25,
    'max_depth': 35,
    'min_sum_hessian_in_leaf': 0.073329100561061,
    'feature_fraction': 0.6599410656015101,
    'feature_fraction_bynode': 0.2732505383665931,
    'bagging_fraction': 0.4385903139779751,
    'bagging_freq': 33,
    'verbose': -1,
    'force_col_wise': True,
    'early_stopping_rounds': 200,
    'num_class': 3,
    'metric': 'multi_logloss'
  }
'''

#-- マルチクラス分類用のlightgbm。optunaで試行するも、結果があまり安定しなかったためhparamsはほぼ触れず
def predict_trend_flag(data, target_col='trend_flag', score_display=True, graph_render=True):
#  model_path = './create_model/'+"mt5_usdjpy_202307041026_lgb_x18_1500"+'.pkl'
#  model = pickle.load(open(model_path, 'rb'))  

  params = {
    'objective': 'multiclass',
    'verbose': -1,
    'num_class': 3,
    'metric': 'multi_logloss'
  }

  epochs = 300
  test_rate = 0.1
  train_rate = 0.9
  sc_use = "None"
  #sc_use = "minmax"
  #sc_use = "standard"

  data_bk = deepcopy(data)

  Y = data[[target_col]]
  X = data.drop(target_col, axis=1)
  X_col_list = list(X.columns)

  if sc_use == "minmax":
      sc = MinMaxScaler(feature_range=(0, 1))        
      X = sc.fit_transform(X)        
  elif sc_use == "standard":
      sc = StandardScaler()
      X = sc.fit_transform(X)

  X = np.array(X)

  train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=test_rate, shuffle=False)
  train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, train_size=train_rate, shuffle=False)            

  train_data = lgb.Dataset(
      data=train_x[:-1], 
      label=train_y[1:],       
      feature_name=X_col_list
  )

  valid_data = lgb.Dataset(
      data=valid_x[:-1], 
      label=valid_y[1:],       
      feature_name=X_col_list
  )            

  model = lgb.train(
      params=params,
      train_set=train_data,
      valid_sets=valid_data,
      num_boost_round=epochs,
      callbacks=[
          lgb.early_stopping(stopping_rounds=50, verbose=False),                           
          lgb.log_evaluation(False),
      ],
    )            
      
  test_pred_data = np.array(model.predict(test_x, num_iteration=model.best_iteration))

  test_pred_data = np.argmax(test_pred_data, axis=1)
  test_pred_data = pd.DataFrame(test_pred_data)
  test_pred_data.columns = ["yhat"]
  test_y.reset_index(inplace=True)
  test_pred_data = pd.concat([pd.DataFrame(test_y), test_pred_data], axis=1)
  test_pred_data.rename({target_col: "y"}, axis=1, inplace=True)
  test_pred_data.set_index("time", inplace=True)

  result_predict_data, score = pred_data_process(test_pred_data, data_bk, target_col, score_display=score_display, graph_render=graph_render)

  return result_predict_data[[target_col]], score

####################################################################################################
