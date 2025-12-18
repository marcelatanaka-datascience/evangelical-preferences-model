"""
Script para padronizar a variável V80 em z-score
e adicionar a coluna padronizada ao CSV com z-scores da wave 2.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuração dos caminhos
BASE_DIR = Path(__file__).parent
WAVE2_DIR = BASE_DIR / 'wave-2-1991-br'
INPUT_FILE = 'WV2_Data_Brazil_Csv_v1.6.1_com_zscore.csv'
OUTPUT_FILE = 'WV2_Data_Brazil_Csv_v1.6.1_com_zscore.csv'

def load_data():
    """Carrega os dados do arquivo com z-scores."""
    file_path = WAVE2_DIR / INPUT_FILE
    
    if not file_path.exists():
        print(f"Arquivo não encontrado: {file_path}")
        return None
    
    print(f"Carregando dados de {file_path}...")
    
    try:
        # Lê o arquivo CSV (separado por ponto-e-vírgula)
        df = pd.read_csv(file_path, sep=';', encoding='utf-8', low_memory=False)
        print(f"Dados carregados: {len(df)} linhas, {len(df.columns)} colunas")
        return df
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None

def check_variable(df, variable):
    """Verifica se a variável existe no DataFrame."""
    if variable not in df.columns:
        print(f"\n⚠️  Variável não encontrada: {variable}")
        print(f"\nVariáveis disponíveis que começam com 'V8':")
        v8_vars = [col for col in df.columns if col.startswith('V8')]
        print(v8_vars[:20])  # Mostra as primeiras 20
        return False
    
    return True

def standardize_variable(df, variable):
    """Padroniza a variável em z-score e adiciona coluna ao DataFrame."""
    print(f"\n{'='*60}")
    print(f"PADRONIZAÇÃO EM Z-SCORE: {variable}")
    print(f"{'='*60}\n")
    
    # Verifica se a coluna z-score já existe
    z_col = f"{variable}_z"
    if z_col in df.columns:
        print(f"⚠️  A coluna {z_col} já existe. Será recalculada.")
    
    # Calcula z-score (subtrai média e divide pelo desvio padrão)
    # Usa apenas valores válidos (não missing) para calcular média e desvio
    valid_values = df[variable].dropna()
    
    if len(valid_values) == 0:
        print(f"⚠️  Variável {variable} não tem valores válidos.")
        return df
    
    mean = valid_values.mean()
    std = valid_values.std()
    
    if std == 0:
        print(f"⚠️  Variável {variable} tem desvio padrão zero. Não é possível padronizar.")
        return df
    
    # Calcula z-score: (x - mean) / std
    # Valores missing permanecem como missing
    df[z_col] = (df[variable] - mean) / std
    
    print(f"{variable} -> {z_col}:")
    print(f"  Média original: {mean:.4f}")
    print(f"  Desvio padrão original: {std:.4f}")
    print(f"  Média z-score: {df[z_col].mean():.6f}")
    print(f"  Desvio padrão z-score: {df[z_col].std():.6f}")
    print(f"  Valores válidos: {df[z_col].notna().sum()}")
    print()
    
    return df

def save_output(df, output_path):
    """Salva o DataFrame com a nova coluna."""
    try:
        # Salva mantendo o separador original (ponto-e-vírgula)
        df.to_csv(output_path, sep=';', index=False, encoding='utf-8-sig')
        print(f"✓ Arquivo salvo com sucesso: {output_path}")
        print(f"  Total de colunas: {len(df.columns)}")
        print(f"  Total de linhas: {len(df)}")
        return True
    except Exception as e:
        print(f"❌ Erro ao salvar arquivo: {e}")
        return False

def main():
    """Função principal."""
    print("="*60)
    print("PADRONIZAÇÃO EM Z-SCORE: V80 - Wave 2 (1991)")
    print("="*60)
    
    # Carrega dados
    df = load_data()
    if df is None:
        return
    
    # Variável de interesse
    variable = 'V80'
    
    # Verifica se a variável existe
    if not check_variable(df, variable):
        return
    
    # Padroniza a variável
    df_with_zscore = standardize_variable(df, variable)
    
    # Salva o arquivo
    output_path = WAVE2_DIR / OUTPUT_FILE
    if save_output(df_with_zscore, output_path):
        print(f"\n{'='*60}")
        print("Processo concluído com sucesso!")
        print(f"{'='*60}")
        print(f"\nArquivo atualizado: {output_path}")
        print(f"Coluna adicionada: {variable}_z")

if __name__ == "__main__":
    main()








