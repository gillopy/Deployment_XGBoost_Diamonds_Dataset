import xgboost as xgb
import pandas as pd

def train_model(X_train: pd.DataFrame, y_train: pd.Series, params: dict = None) -> xgb.XGBRegressor:
    """Entrena el modelo XGBoost para predicción de precios."""
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'enable_categorical': True
        }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    return model