# src/data/data_processor.py
import pandas as pd
import numpy as np

def process_data(df: pd.DataFrame, target_column: str = None) -> tuple[pd.DataFrame, pd.Series]:
    """Procesa los datos para el modelo de predicción de precios de diamantes."""
    # Convertir características categóricas
    categorical_columns = ['cut', 'color', 'clarity']
    for col in categorical_columns:
        df[col] = df[col].astype('category')
    
    # Extraer target
    target = df[target_column] if target_column else None
    df_processed = df.drop(columns=[target_column]) if target_column else df
    
    return df_processed, target