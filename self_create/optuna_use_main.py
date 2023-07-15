
#-- optuna関連 クラス&関数
#-- 他にも使用関数がありましたが、内容がほぼ重複しているため抜粋したものです。

from self_create.mt5_model_functions import * 

import pandas as pd
import numpy as np
from time import time, sleep
from datetime import datetime, timezone, timedelta
from copy import copy, deepcopy

import math
import itertools
import random
import json
import os

import pickle

import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.dates as mdates

from importlib import reload

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import optuna

from keras.models import model_from_json
from keras.utils import custom_object_scope
from tensorflow.keras.utils import get_custom_objects

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import lightgbm as lgb
from sklearn.linear_model import LinearRegression

####################################################################################################

#-- 回帰 rnn用optunaオブジェクトクラス
class RnnObjective:
    def __init__(
        self, data_set, try_model, sc, df, col_name_list, freq_num, freq, source_data, 
        rnn_batch_size, epochs=300, batch_size=32, future_period=None, test_mode=0, json_create_flag=False
    ):
        self.train_x = data_set[0][0]
        self.train_y = data_set[0][1]        
        self.valid_x = data_set[1][0]
        self.valid_y = data_set[1][1]        
        self.test_x = data_set[2][0]
        self.test_y = data_set[2][1]
        
        self.sc = sc
        self.df = df
        self.col_name_list = col_name_list
        self.freq_num = freq_num
        self.freq = freq
        self.source_data = source_data
        
        self.try_model = try_model
        self.rnn_batch_size = rnn_batch_size
        self.output_dence_units = train_x.shape[2]
        self.epochs = epochs
        self.batch_size = batch_size
        self.future_period = future_period
        
        self.test_mode = test_mode
        self.json_create_flag = json_create_flag
        
        self.current_num = 1
        self.result_dic = {}

    #-- 指定回数パラメータの試行
    def __call__(self, trial):        
        print(f'・試行回数: {self.current_num} 回目')

        params = {            
            "layer_num" : trial.suggest_categorical("layer_num", [2]),       
            "units" : trial.suggest_int("units", 5, 50),
            "activation" : trial.suggest_categorical("activation", ["Mish", "relu", None]),            
            "dropout_rate" : trial.suggest_float("dropout_rate", 0.005, 0.05),
            "lr" : trial.suggest_float("lr", 0.00005, 0.0001), 
#            "loss" : trial.suggest_categorical("loss", ["mse", "rmse", "mape"]), 
            "min_delta" : trial.suggest_categorical("min_delta", [1.0e-8]),
            "restore_best_weights" : trial.suggest_categorical("restore_best_weights", [True]),             
        }
        
        choice_params = {
            "rnn_batch_size": self.rnn_batch_size, "epochs": self.epochs, "batch_size": self.batch_size, 
            "layer_num": params["layer_num"], "units": params["units"], "activation": params["activation"],
            "batch_normalization": False, "dropout_rate": params["dropout_rate"], "lr": params["lr"], 
            "min_delta": params["min_delta"], "restore_best_weights": params["restore_best_weights"],
            "early_stopping": 0,
        }
        
        model = Sequential()
        if self.try_model == "gru":
            if params["layer_num"] == 1:
                model.add(GRU(
                    units=params["units"], activation=params["activation"],
                    return_sequences=False, input_shape=input_shape
                ))         
                model.add(Dropout(params["dropout_rate"]))                 
            else:
                model.add(GRU(
                    units=params["units"], activation=params["activation"],
                    return_sequences=True, input_shape=input_shape
                ))
                model.add(Dropout(params["dropout_rate"]))                                 
                model.add(GRU(
                    units=params["units"], activation=params["activation"]
                ))                    
                model.add(Dropout(params["dropout_rate"]))                         
        elif self.try_model == "lstm":
            if params["layer_num"] == 1:
                model.add(LSTM(
                    units=params["units"], activation=params["activation"],
                    return_sequences=False, input_shape=input_shape
                ))                                
                model.add(Dropout(params["dropout_rate"]))                                 
            else:
                model.add(LSTM(
                    units=params["units"], activation=params["activation"],
                    return_sequences=True, input_shape=input_shape
                ))
                model.add(Dropout(params["dropout_rate"]))                                 
                model.add(LSTM(
                    units=params["units"], activation=params["activation"]
                ))
                model.add(Dropout(params["dropout_rate"]))                         

        model.add(Dense(units=self.output_dence_units, activation='linear'))
        model.compile(optimizer=Adam(params["lr"]), loss='mean_squared_error')
        
        history = model.fit(
            self.train_x, self.train_y, 
            epochs = self.epochs, batch_size = self.batch_size,
            validation_data=(self.valid_x, self.valid_y),
            shuffle=False
            , callbacks = EarlyStopping(
                patience = 0,
                monitor = "val_loss",
                min_delta = params["min_delta"],
                restore_best_weights = params["restore_best_weights"],
                verbose = -1
            )
        )
                
        if self.test_mode == 0:
            predict_y = model.predict(self.test_x, batch_size=self.batch_size, verbose=0)
            score = return_result_score(self.test_y, predict_y)
        
        elif self.test_mode == 1:
            x_future = self.test_x[-1:]
            pred_y_list = []

            for i, step in enumerate(range(self.future_period)):
                y_future = model.predict(x_future, batch_size=1, verbose=0)    
                x_future = x_future[0][1:].tolist()
                y_future = y_future[0].tolist()
                x_future.append(y_future)
                x_future = np.array([x_future])
                pred_y_list.append(y_future)

            pred_y_list = np.array(pred_y_list)
                       
            predict_y = self.sc.inverse_transform(pred_y_list)  
            predict_y_df = pd.DataFrame(predict_y)
            predict_y_df.columns = self.col_name_list
            predict_y_df = predict_y_df[["close", "open", "high", "low"]]  
            predict_y_df.rename({"close": "yhat"}, axis=1, inplace=True)

            x_datetime = deepcopy(self.df[["time"]])
            x_datetime = x_datetime.iloc[len(x_datetime)-len(self.test_x)-self.future_period:]
            x_datetime.reset_index(inplace=True)
            x_datetime.drop("index", axis=1, inplace=True)
            
            if freq == "Min":
                foward_time = x_datetime["time"].max() + timedelta(minutes=self.freq_num)      
                future_datetime_df = pd.DataFrame(
                  pd.date_range(start=foward_time, periods=self.future_period, freq=f'{self.freq_num}{self.freq}')
                , columns=["time"])
                future_datetime_df = future_datetime_df.reset_index()
                future_datetime_df = future_datetime_df.drop("index", axis=1)
                future_datetime_df["time"] = future_datetime_df["time"].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))    

            pred_time_join = pd.concat([future_datetime_df, predict_y_df], axis=1)

            result_df = self.source_data[["time", "close"]]
            result_df = result_df.rename({"close": "y"}, axis=1)
            
            result_df = pd.merge(result_df[["time", "y"]], pred_time_join[["time", "yhat"]], how="inner", on="time")
            result_df = result_df.set_index("time")
            
            score = return_result_score(result_df.y, result_df.yhat)
                
        self.result_dic[f"{self.try_model}_{self.current_num}"] = {"score": score, "params": choice_params}
        
        if self.current_num >= 2 and self.current_num % 20 == 0:
            json_create(self.result_dic, json_create_flag = self.json_create_flag, indent = 4, ensure_ascii = False, other_1 = f"_usdjpy_{self.try_model}", other_2 = f"_x{len(data.columns)-1}_{len(data.close)}_{self.current_num}_normal_15m")

        model_add_info = \
        f"_20230616_usdjpy_{self.try_model}_x{len(data.columns)-1}_{len(data.close)}_{self.current_num}_normal"
        create_model_save_and_load(target_model_path, mode="save", model_add_info=model_add_info, model=model)
        
        self.current_num += 1
        return score["mse_calc"]
        

#-- rnn用optunaオブジェクトクラスのメインプロセス関数
def rnn_optuna_parameter(
    data_set, try_model, sc, df, col_name_list, freq_num, freq, source_data,
    rnn_batch_size, epochs=300, batch_size=32, future_period=None, test_mode=0, n_trials=100, json_create_flag=False
):
    rnn_objective = RnnObjective(
        data_set, try_model, sc, df, col_name_list, freq_num, freq, source_data,
        rnn_batch_size, epochs, batch_size, future_period=future_period, test_mode=test_mode, json_create_flag=json_create_flag
    )
    study = optuna.create_study(sampler = optuna.samplers.RandomSampler(), direction="minimize")
    study.optimize(rnn_objective, n_trials=n_trials)
    result_params = {"best_params": study.best_params, "all_data": rnn_objective.result_dic}
    json_create(result_params, json_create_flag = json_create_flag, indent = 4, ensure_ascii = False, other_1 = f"_usdjpy_{try_model}", other_2 = f"_x{len(data.columns)-1}_{len(data.close)}_complete_normal_15m")    

    return [study, result_params]


#-- 呼び出し例
study, result_params = rnn_optuna_parameter(
    fit_use_data_set, "gru", sc, df, col_name_list, freq_num, freq, source_data,
    rnn_batch_size, epochs=300, batch_size=128, future_period=future_period, test_mode=0, n_trials=1000, json_create_flag=True
)

####################################################################################################

#-- 分類スコア取得関数
def return_classification_result_score(y_data, yhat_data, average='macro'):
    return {
        "accuracy": accuracy_score(y_data, yhat_data),
        "recall": recall_score(y_data, yhat_data, average=average),
        "precision": precision_score(y_data, yhat_data, average=average),
        "f1": f1_score(y_data, yhat_data, average=average),
    }


#-- マルチクラス分類用 lightgbm用optunaオブジェクトクラス ※ 回帰の物をそのまま流用したので名前がRegressorのままです
class RegressorObjective:
    def __init__(
        self, data, try_model, epochs=300, test_mode=0, json_create_flag=False
    ):
        self.data = data
        
        self.try_model = try_model
        self.epochs = epochs
        
        self.test_mode = test_mode
        self.json_create_flag = json_create_flag
        
        self.current_num = 1
        self.result_dic = {}


    #-- 指定回数パラメータの試行
    def __call__(self, trial):
        
        print(f'・試行回数: {self.current_num} 回目')        

        params = {
            'objective' : trial.suggest_categorical('objective', ['multiclass']),

            'boosting_type' : trial.suggest_categorical('boosting_type', ['gbdt', 'rf']),
            
            'learning_rate' : trial.suggest_float('learning_rate', 0.00001, 0.1),
            
            'lambda_l1' : trial.suggest_float('lambda_l1', 2.0e-08, 2.0e-04),
            'lambda_l2' : trial.suggest_float('lambda_l2', 1.0e-09, 1.0e-05),
            
            'num_leaves' : trial.suggest_int('num_leaves', 2, 100),
            'min_data_in_leaf' : trial.suggest_int('min_data_in_leaf', 2, 50),
            'max_depth' : trial.suggest_int('max_depth', -1, 50),
            
            'min_sum_hessian_in_leaf' : trial.suggest_float('min_sum_hessian_in_leaf', 1.0e-5, 1.0e-1),

            'feature_fraction' : trial.suggest_float('feature_fraction', 0.1, 0.9),
            'feature_fraction_bynode' : trial.suggest_float('feature_fraction_bynode', 0.1, 0.9),
            'bagging_fraction' : trial.suggest_float('bagging_fraction', 0.1, 0.9),

            'bagging_freq' : trial.suggest_int('bagging_freq', 1, 50),
            
            'verbose' : trial.suggest_categorical('verbose', [-1]), 
            
            'force_col_wise' : trial.suggest_categorical('force_col_wise', [True, False]),             
        }

        early_stopping_rounds = 200            
        sc_use = trial.suggest_categorical('scaler_use', ["minmax", "standard", None])    

        choice_params = {
            'scaler_use': sc_use,
            'objective': params['objective'],
            "n_estimators": self.epochs, 'boosting_type': params['boosting_type'], 'learning_rate': params['learning_rate'],
            'lambda_l1': params['lambda_l1'], 'lambda_l2': params['lambda_l2'],             
            'num_leaves': params['num_leaves'],
            'min_data_in_leaf': params['min_data_in_leaf'], 'max_depth': params['max_depth'], 

            'min_sum_hessian_in_leaf': params['min_sum_hessian_in_leaf'],
            'feature_fraction': params['feature_fraction'], 'feature_fraction_bynode': params['feature_fraction_bynode'],
            'bagging_fraction': params['bagging_fraction'], 'bagging_freq': params['bagging_freq'], 
            
            'verbose': params['verbose'],
            'force_col_wise': params['force_col_wise'], 
            "early_stopping_rounds": early_stopping_rounds,
        
        }

        #-- 本当はself.test_mode == 0とself.test_mode == 1が回帰でした。mt5_model_functions.pyの中身とほぼ変わらないため省略

        if self.test_mode == 2:        
            
            target_col = "trend_flag"
            test_rate = 0.1
            train_rate = 0.9        
            
            params["num_class"] = 3
            params['metric'] = 'multi_logloss'           
            choice_params["num_class"] = 3           
            choice_params['metric'] = 'multi_logloss'           

            Y = self.data[[target_col]]
            X = self.data.drop(target_col, axis=1)
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
                data=train_x, 
                label=train_y, 
                feature_name=X_col_list
            )
            valid_data = lgb.Dataset(
                data=valid_x, 
                label=valid_y, 
                feature_name=X_col_list
            )            

            model = lgb.train(
                params=params,
                train_set=train_data,
                valid_sets=valid_data,
                num_boost_round=self.epochs,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
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
            
            score = return_classification_result_score(test_pred_data.y, test_pred_data.yhat)
        
        self.result_dic[f"{self.try_model}_{self.current_num}"] = {"score": score, "params": choice_params}
        
        if self.current_num >= 2 and self.current_num % 1000 == 0:
            json_create(self.result_dic, json_create_flag = self.json_create_flag, indent = 4, ensure_ascii = False, other_1 = f"_usdjpy_{self.try_model}", other_2 = f"_x{len(data.columns)-1}_{len(data)}_{self.current_num}")
            
        self.current_num += 1

        if self.test_mode == 2:  
            return score["f1"]


#-- lightgbm用optunaオブジェクトクラスのメインプロセス関数
def regressor_optuna_parameter(
    data, try_model, epochs=300, n_trials=100, test_mode=0, json_create_flag=False
):
    regressor_objective = \
        RegressorObjective(data, try_model, epochs, test_mode, json_create_flag)
    study = optuna.create_study(sampler = optuna.samplers.RandomSampler(), direction="minimize")
    optuna.logging.disable_default_handler()
    study.optimize(regressor_objective, n_trials=n_trials)
    result_params = {"best_params": study.best_params, "all_data": regressor_objective.result_dic}
    json_create(result_params, json_create_flag = json_create_flag, indent = 4, ensure_ascii = False, other_1 = f"_usdjpy_{try_model}", other_2 = f"_x{len(data.columns)-1}_{len(data)}_complete")    

    return [study, result_params]


#-- 呼び出し例
study, result_params = regressor_optuna_parameter(
    data, model_name, epochs=500, n_trials=10000, test_mode=2, json_create_flag=True    
)

####################################################################################################
#-- lightgbm セーブ&ロード

#model_path = './create_model/'+"mt5_usdjpy_202307031551_lgb_x24_1498"+'.pkl'
#pickle.dump(model, open(model_path, 'wb'))

#model = pickle.load(open(model_path, 'rb'))

####################################################################################################

#-- マルチクラス分類についての説明変数選別用関数
def trend_flag_explanatory_variable(source_data, target_col, source_data_2=None, data_range_adj=None, data_range_per=None, exclusion_list=None, select_threshold_val=None):
    source_data["time"] = source_data["time"].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))
    source_data_2["time"] = source_data_2["time"].apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S'))

    data = deepcopy(source_data)
    if type(data_range_adj) != type(None):
        data = data[data["time"]>=data_range_adj]
    if type(data_range_per) != type(None):        
        data = data.iloc[-int(len(data)*data_range_per):]                
    data.reset_index(drop=True, inplace=True)
    data = data.dropna(how="any", axis=1)
    data = data.dropna(how="any", axis=0)

    data_2 = deepcopy(source_data_2)
    if type(data_range_adj) != type(None):    
        data_2 = data_2[data_2["time"]>=data_range_adj]
    if type(data_range_per) != type(None):        
        data_2 = data_2.iloc[-int(len(data_2)*data_range_per):]                
    data_2.reset_index(drop=True, inplace=True)
    data_2 = data_2.dropna(how="any", axis=1)
    data_2 = data_2.dropna(how="any", axis=0)
    
    data[target_col] = data[target_col].apply(
        lambda x:
            2 if (x >= 0.02) else \
            1 if (0.02 > x) and (x >= 0.01) else \
           -1 if (-0.01 >= x) and (x > -0.02) else \
           -2 if (-0.02 >= x) else \
            0
    )
        
    corr_calc_use_data = pd.merge(data[["time", target_col]], data_2, how="inner", on="time")
    corr_calc_use_data.set_index("time", inplace=True)

    corr_calc_use_data = pd.DataFrame(corr_calc_use_data.corr().iloc[0]).sort_values(target_col, ascending=[False]).T
    corr_calc_use_data.drop(exclusion_list, axis=1, inplace=True)

    select_features_list = []
    for col in corr_calc_use_data:
        corr_val = corr_calc_use_data.iloc[0][col]
        if abs(corr_val) >= select_threshold_val:
            select_features_list.append(col)

    select_features_list.insert(0, "time")        

    data_2 = data_2[select_features_list]

    data = data.reindex(columns=["time", target_col, "open", "high", "low", "close"])
    data = pd.merge(data[["time", target_col]], data_2, how="inner", on="time")
    data.set_index("time", inplace=True)

    data_cols = list(data.columns)

    return data, data_cols, corr_calc_use_data

####################################################################################################

#-- 現在の日時を取得
def current_day():
  return datetime.now(timezone(timedelta(hours=9)))


#-- 現在の日時を文字列に変換、フォーマット変更
def get_str_current_day(flag):
  tmp_day_str = str(current_day())
  tmp_day_str = tmp_day_str[:tmp_day_str.find(".")].replace(" ", "-").replace(":", "-")
  tmp_day_str = tmp_day_str[:tmp_day_str.rfind("-")]

  if flag == 0:
    return tmp_day_str

  elif flag == 1:
    return tmp_day_str.replace("-", "")


#-- jsonファイルを作成
def json_create(data, json_create_flag = False, indent = 4, ensure_ascii = False, other_1 = "", other_2 = ""):
  if json_create_flag == True:
    now_time_str = get_str_current_day(1)
    with open('./json/'+f'optuna_{now_time_str}{other_1}{other_2}'+'.json', 'w') as f:
        json.dump(data, f, ensure_ascii = ensure_ascii , indent =indent)


#-- csvファイルを作成
def csv_create(target_brand, data, data_name, csv_create_flag = 0, other = ""): 
  if csv_create_flag == 1:
    now_time_str = get_str_current_day(1)
    data.to_csv('./csv/'+f'{target_brand}_{data_name}_{now_time_str}_{other}'+'.csv', header=True, index=False, encoding="utf-8")


#-- jsonファイルからhparamをロード
def json_params_loader(data, load_mode = 0, select_score="mse_calc"):
  json_open = open('./json/'+data+'.json', 'r')
  json_load = json.load(json_open)
    
  if load_mode == 0:
    return json_load
  elif load_mode == 1 or load_mode == 2:
    score_check_dic = {}
    for index, (key, val) in enumerate(json_load["all_data"].items()):
        score_check_dic[key] = val["score"][select_score]
    if select_score == 'r2_calc':
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


#-- データセットから使用する範囲を切り出し
def use_data_range_trimming(data, mode=0, range_rate=0.1, start_time=None, end_time=None):    
    copy_data = deepcopy(data)
    
    if mode == 0:
        copy_data = copy_data.iloc[-int(len(copy_data)*range_rate):]
        copy_data = copy_data.reset_index()
        copy_data = copy_data.drop("index", axis=1)
    elif mode == 1:
        if (type(start_time) != type(None)) & (type(end_time) != type(None)):
            copy_data = copy_data[copy_data["time"]>=start_time]
            copy_data = copy_data[copy_data["time"]<=end_time] 
            copy_data = copy_data.reset_index()
            copy_data = copy_data.drop("index", axis=1)
        elif type(start_time) != type(None):
            pass            
        elif type(end_time) != type(None):            
            copy_data = copy_data[copy_data["time"]<=end_time]
            
    elif mode == 2:
        copy_data = copy_data[copy_data["time"]<=end_time]  
        copy_data = copy_data.iloc[-int(len(copy_data)*range_rate):]
        copy_data = copy_data.reset_index()
        copy_data = copy_data.drop("index", axis=1)        
        
    return copy_data


#-- データセットの開始位置、終了位置を任意に変更
def slide_data_range(data, slide_data_range_rate=None, slide_data_row_back=None):
    data = deepcopy(data)
    if type(slide_data_range_rate) != type(None) and type(slide_data_row_back) != type(None):
        data = data[-int(len(data)*slide_data_range_rate):]
        data = data.reset_index()
        data = data.drop("index", axis=1)
        data = data[:-slide_data_row_back]        
        
    elif type(slide_data_range_rate) != type(None):
        data = data[-int(len(data)*slide_data_range_rate):]
        data = data.reset_index()
        data = data.drop("index", axis=1)
        
    elif type(slide_data_row_back) != type(None):
        data = data[:-slide_data_row_back]

    return data


#-- ディレクトリ内のファイル名をまとめて変更
def file_name_edit():
  dir_path = "./create_model/"
  dir_list = os.listdir(dir_path)
  for i, file_name in enumerate(dir_list):
      if ".h5" == os.path.splitext(dir_list[i])[1] or ".json" == os.path.splitext(dir_list[i])[1]:
          name = os.path.splitext(dir_list[i])[0]
          tmp_1 = name[:name.find("gru_")+3]
          tmp_2 = name[name.find("gru_")+4:]
          tmp_3 = tmp_2[tmp_2.find("_"):]
          new_file_name = tmp_1 + tmp_3 + "_future"
          new_file_name = new_file_name + os.path.splitext(dir_list[i])[1]
          os.rename(os.path.join(dir_path, dir_list[i]), os.path.join(dir_path, new_file_name))

####################################################################################################
