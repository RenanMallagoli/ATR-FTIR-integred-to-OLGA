import pandas as pd
import numpy as np
from ftir_prep import PipelineBuilder

def load_and_clean_real_data(train_csv_path, test_csv_path):
    """
    Carrega os dados reais, extrai a classe alvo correta, isola APENAS 
    os comprimentos de onda (espectros) e aplica o pipeline de suavização.
    """
    # 1. Leitura e união dos CSVs
    df_train = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)
    df_total = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    
    df_total = df_total.dropna(subset=["class"])
    
    # 2. Definição do Alvo (Target)
    # Baseado no código original, a coluna se chama "class"
    y = df_total["class"].values
    
    # 3. Isolamento EXCLUSIVO da Matriz Espectral
    # Mapeamos todas as colunas que não são frequências de infravermelho
    colunas_metadados = ["id", "GENDER", "AGE", "HEMOGLOBIN", "GLUCOSE", "class"]
    
    # Removemos os metadados (apenas se eles existirem no dataframe para evitar erros)
    colunas_remover = [col for col in colunas_metadados if col in df_total.columns]
    X_df = df_total.drop(columns=colunas_remover)
    
    # Agora X contém APENAS os valores dos espectros purificados
    X = X_df.values
    
    # Como removemos as strings, os cabeçalhos restantes são os wavenumbers
    wavenumbers = X_df.columns.astype(float).values
    
    # 4. Pipeline de Limpeza (FTIR-Prep)
    print("Aplicando Savitzky-Golay e Correção de Baseline na matriz de espectros...")
    pipeline = (PipelineBuilder()
                .add_truncation('fingerprint')         # Mantém apenas a região de interesse biológico
                .add_baseline('rubberband')            # Endireita a linha de base da onda
                .add_smoothing('savgol', polyorder=2)  # Remove os micro-ruídos
                .add_normalization('minmax')           # Escala de 0 a 1
                .build())
                
    X_processed, wavenumbers_processed = pipeline.process(X, wavenumbers)
    X_processed = X_processed[:, ::2] 
    wavenumbers_processed = wavenumbers_processed[::2]
    
    print(f"Sucesso! {len(X_processed)} espectros limpos (resolução reduzida para velocidade).")
    
    return X_processed, y, wavenumbers_processed