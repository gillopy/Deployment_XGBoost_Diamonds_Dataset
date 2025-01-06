import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """Carga los datos desde un archivo CSV."""
    return pd.read_csv(file_path)

# src/data/data_processor.py
import pandas as pd
import numpy as np

def process_data(df: pd.DataFrame, columns_to_impute: list, target_column: str = None) -> tuple[pd.DataFrame, pd.Series]:
    """Procesa los datos para el modelo de predicción de precios de diamantes."""
    # Convertir características categóricas
    categorical_columns = ['cut', 'color', 'clarity']
    for col in categorical_columns:
        df[col] = df[col].astype('category')
    
    # Imputar valores faltantes
    df[columns_to_impute] = df[columns_to_impute].replace(0, np.nan)
    
    # Extraer target
    target = df[target_column] if target_column else None
    
    return df, target