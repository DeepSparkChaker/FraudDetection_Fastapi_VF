# -*- coding: utf-8 -*-
"""Model config in json format"""
cfg = {
    "data": {
        "path": "C:/Users/rzouga/Desktop/ALLINHERE/ALLINHERE/FraudDetection/transactions_train.csv"
    },
    # "data_test": {
    #   "path": "../input/ventilator-pressure-prediction/test.csv"
    # },
    # "data_submission": {
    #   "path": "../input/ventilator-pressure-prediction/test.csv"
    # },
    "train": {
        'fit_params': {'early_stopping_rounds': 100, 'verbose': 55000},
        'n_fold': 5,
        'seeds': [2021],
        'target_col': "isFraud",
        'debug': False

    },
    "model": {'n_estimators': 11932, 
                    'max_depth': 16, 
                    'learning_rate': 0.005352340588475586,
                    'lambda_l1': 1.4243404105489683e-06,
                    'lambda_l2': 0.04777178032735788,
                    'num_leaves': 141, 
                    'feature_fraction': 0.6657626611307914, 
                    'bagging_fraction': 0.9115997498937961,
                    'bagging_freq': 1,
                    'min_child_samples': 51,
                     "objective": "binary",
                     #"metric": "binary_logloss",
                     "verbosity": -1,
                     "boosting_type": "gbdt",
                     #"random_state": 228,
                     "metric": "auc",
                     #"device": "gpu",
                     'tree_method': "gpu_hist"
                    }
}