# src/main.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.data_loader import load_data
from src.data.data_processor import process_data
from src.data.data_splitter import split_data
from src.model.trainer import train_model
from src.model.evaluator import evaluate_model
from src.model.saver import save_model
import pandas as pd

def main():
    # Cargar y dividir datos primero
    diamonds = load_data(file_path="data/raw/diamonds.csv")
    X_train, X_test, y_train, y_test = split_data(diamonds, target_column='price', stratify=False)
    
    # Luego procesar los datos de entrenamiento
    X_train_processed, _ = process_data(df=X_train)
    X_test_processed, _ = process_data(df=X_test)
    
    # Entrenar modelo
    model = train_model(X_train=X_train_processed, y_train=y_train)
    
    # Evaluar modelo
    rmse, mae, r2 = evaluate_model(model, test_data=X_test_processed, y_test=y_test)
    
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.2f}")
    
    # Guardar modelo
    save_model(model, model_path="models/trained_model")

if __name__ == "__main__":
    main()