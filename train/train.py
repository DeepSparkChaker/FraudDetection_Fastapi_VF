# Importing the required Python libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import Pipeline
import joblib
from configs_json.config import cfg
from utils.config import  Config
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from dataloader.dataloader  import  DataLoader
from utils.preparation import *

def create_train_test_data(dataset,train_size=0.5):
    # load and split the data
    data_train = dataset.sample(frac=train_size, random_state=30).reset_index(drop=True)
    data_test = dataset.drop(data_train.index).reset_index(drop=True)
    # save the data
    data_train.to_csv('data/train.csv', index=False)
    data_test.to_csv('data/test.csv', index=False)
    print(f"Train data for modeling: {data_train.shape}")
    print(f"Test data for predictions: {data_test.shape}")

def train_model(x_train, y_train):
     
    print("Training the model ...")
    # Model defined 
    params={"learning_rate": Config.from_json(cfg).model.learning_rate,
        #'device': Config.from_json(cfg).model.device,
        'metric':Config.from_json(cfg).model.metric,
        'objective':Config.from_json(cfg).model.objective,
        'n_estimators':  Config.from_json(cfg).model.n_estimators,
        'num_leaves': Config.from_json(cfg).model.num_leaves,
        'min_child_samples':  Config.from_json(cfg).model.min_child_samples,
        'feature_fraction': Config.from_json(cfg).model.feature_fraction,
        'bagging_fraction': Config.from_json(cfg).model.bagging_fraction,
        'bagging_freq':  Config.from_json(cfg).model.bagging_freq,
        #'reg_alpha':  Config.from_json(cfg).model.reg_alpha,
        #'reg_lambda':  Config.from_json(cfg).model.reg_lambda,
       # 'gpu_platform_id':  Config.from_json(cfg).model.gpu_platform_id
            }
    params_optuna={'n_estimators': 11932, 
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
    model_lgbm  = LGBMRegressor(**params_optuna)
    # Pipline preprocess
    pipeline_model_lgbm = Pipeline([ 
        ('pre', data_preparing),
        ('lgbm', model_lgbm)
    ])
    pipeline_model_lgbm.fit(x_train, y_train)

    return pipeline_model_lgbm

def accuracy(model, x_test, y_test):
    print("Testing the model ...")
    predictions = model.predict(x_test)
    roc_auc = roc_auc_score(y_test, predictions)
    return roc_auc

def export_model(model):
    # Save the model
    joblib_path = 'models/model_test.joblib'
    with open(joblib_path, 'wb') as file:
        joblib.dump(model, file)
        print(f"Model saved at {joblib_path}")

def main():
    #mlops data 
    # data =
    # Load the whole data
    dataloder=DataLoader()
    data = dataloder.load_data(Config.from_json(cfg).data)
  
    # Split train/test
    # Creates train.csv and test.csv
    create_train_test_data(data,train_size=0.5)
    # Loads the data for the model training
    train = pd.read_csv('data/train.csv', keep_default_na=False)
    #num_columns=train.drop(['isFraud'], axis=1).select_dtypes(include=['int64','float64']).columns
    #cat_columns= train.drop(['isFraud'], axis=1).select_dtypes(exclude=['int64','float64']).columns
    x_train = train.drop(['isFraud'], axis=1)
    y_train =  train['isFraud'].to_numpy()
     
    # Loads the data for the model testing
    test = pd.read_csv('data/test.csv', keep_default_na=False)
    x_test = test.drop(['isFraud'], axis=1)
    y_test = test['isFraud'].to_numpy()

    # Train and Test
    model = train_model(x_train, y_train)
    auc_test = accuracy(model, x_test, y_test)
   
    print(f"auc_test: {auc_test}")

    # Save the model
    export_model(model)

if __name__ == '__main__':
    main()