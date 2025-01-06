from datetime import datetime
import joblib
import xgboost

def save_model(model: xgboost.XGBRegressor, model_path: str) -> None:
    """Guarda el modelo en un archivo usando joblib."""
    current_date = datetime.now().strftime("%Y-%m-%d")
    full_path = f"{model_path}_{current_date}.joblib"
    joblib.dump(model, full_path)
    print(f"Modelo guardado en {full_path}")