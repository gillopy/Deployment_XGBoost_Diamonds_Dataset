from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(data: pd.DataFrame, target_column: str, test_size=0.2, random_state=42, stratify: bool = False) -> tuple:
    """Divide los datos en conjuntos de entrenamiento y prueba."""
    X = data.drop(columns=target_column, axis=1)
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test