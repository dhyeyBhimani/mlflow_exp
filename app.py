import os
import warnings
import pandas as pd 
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
from mlflow.models import infer_signature
import sys
from urllib.parse import urlparse



import logging

logging.basicConfig(level=logging.WARN)
logger =  logging.getLogger(__name__)

def evalute(real,pred):
    mae = mean_absolute_error(real, pred)
    mse = mean_squared_error(real, pred)
    r2 = r2_score(real, pred)
    return mae, mse, r2

if __name__ =="__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Read the wine quality data
    try:
        data = pd.read_csv('Data\cleaned_wine_data.csv')
    except Exception as e:
        logging.exception("Data not found")

    train_data,test_data = train_test_split(data)
    X_train = train_data.drop(["quality"], axis=1)
    y_train = train_data[["quality"]]
    X_test = test_data.drop(["quality"], axis=1)
    y_test = test_data[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)


        mae,mse,r2 = evalute(y_test,y_pred)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  MSE: %s" % mse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)


        prediction  = model.predict(X_train)
        signature = infer_signature(X_train, prediction)


        tracking_yrl_pass = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_yrl_pass != 'file':
            mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetWineModel",signature=signature)
        else:
            mlflow.sklearn.log_model(model, "model")
    
            



