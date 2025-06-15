import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
from mlflow.models import infer_signature

data_path = "ml-loan-demo/data/loan_data_prod.csv"
df = pd.read_csv(data_path)
df = pd.get_dummies(df, drop_first=True)
df.dropna(inplace=True)

X = df.drop("LoanAmount", axis=1)
y = df["LoanAmount"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run() as run:
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2",  r2)

    mlflow.sklearn.log_model(
        sk_model    = model,
        artifact_path = "model",
        input_example = X_train.iloc[:1],
        signature     = infer_signature(X_train, y_train)
    )

    print(f"✅ Run ID: {run.info.run_id}")
