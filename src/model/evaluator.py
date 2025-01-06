from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import xgboost
import pandas as pd

def evaluate_model(model: xgboost.XGBRegressor, test_data: pd.DataFrame, y_test: pd.Series) -> tuple[float, float, float]:
    """Evalúa el modelo usando métricas de regresión."""
    y_pred = model.predict(test_data)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return rmse, mae, r2